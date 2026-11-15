"""Return forecasting, walk-forward validation, evaluation, and artifacts."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.baselines import arima_forecast, xgboost_forecast, yesterday_price_forecast, zero_return_forecast
from src.backtest import backtest_long_flat
from src.data_loader import load_ticker_data
from src.dataset import SequenceDataset, chronological_split
from src.evaluate import compare_models, plot_equity_curves, plot_predictions, regression_metrics
from src.features import FEATURE_COLUMNS, add_features
from src.models import GRUModel, LSTMModel, TransformerModel


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    scheduler_factor: float,
    scheduler_patience: int,
    checkpoint_path: str,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, list[float]], bool]:
    """Train a return model and restore its best validation checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    history = {"train": [], "validation": []}
    best_loss, stale_epochs = float("inf"), 0
    early_stopped = False
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    for _ in range(epochs):
        model.train()
        train_losses = []
        for x, target, direction in train_loader:
            x, target, direction = x.to(device), target.to(device), direction.to(device)
            optimizer.zero_grad()
            predicted_return, predicted_direction = model(x)
            loss = mse(predicted_return, target) + bce(predicted_direction, direction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for x, target, direction in validation_loader:
                predicted_return, predicted_direction = model(x.to(device))
                validation_losses.append(
                    (
                        mse(predicted_return, target.to(device))
                        + bce(predicted_direction, direction.to(device))
                    ).item()
                )
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                early_stopped = True
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    return history, early_stopped


def plot_training_history(history: dict[str, list[float]], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["validation"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Return plus direction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def walk_forward_splits(
    n_samples: int, initial_train_size: int, validation_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window train/validation indices without future leakage."""
    splits = []
    train_end = initial_train_size
    while train_end + validation_size <= n_samples:
        splits.append((np.arange(train_end), np.arange(train_end, train_end + validation_size)))
        train_end += validation_size
    return splits


def _scaled_arrays(
    train_frame: pd.DataFrame, evaluation_frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    feature_scaler = StandardScaler().fit(train_frame[FEATURE_COLUMNS])
    train_features = feature_scaler.transform(train_frame[FEATURE_COLUMNS])
    evaluation_features = feature_scaler.transform(evaluation_frame[FEATURE_COLUMNS])
    # Returns are closer to stationary than raw prices and do not grow with
    # the price level, reducing non-stationarity and out-of-range forecasts.
    target_mean = float(train_frame["log_return"].mean())
    target_std = float(train_frame["log_return"].std())
    if target_std == 0:
        raise ValueError("Training log returns must have non-zero variance.")
    train_targets = (train_frame["log_return"].to_numpy() - target_mean) / target_std
    evaluation_targets = (evaluation_frame["log_return"].to_numpy() - target_mean) / target_std
    return train_features, evaluation_features, target_mean, target_std, train_targets, evaluation_targets


def _loaders(
    train_frame: pd.DataFrame, evaluation_frame: pd.DataFrame, window: int, batch_size: int
) -> tuple[dict[str, DataLoader], float, float]:
    train_features, evaluation_features, target_mean, target_std, train_targets, evaluation_targets = (
        _scaled_arrays(train_frame, evaluation_frame)
    )
    loaders = {
        "train": DataLoader(
            SequenceDataset(train_features, train_targets, window),
            batch_size=batch_size,
            shuffle=False,
        ),
        "evaluation": DataLoader(
            SequenceDataset(evaluation_features, evaluation_targets, window),
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    return loaders, target_mean, target_std


def _constructors(training: dict, window: int) -> dict[str, Callable[[], nn.Module]]:
    return {
        "LSTM": lambda: LSTMModel( len(FEATURE_COLUMNS), training["hidden_size"], training["num_layers"], training["dropout"]),
        "GRU": lambda: GRUModel( len(FEATURE_COLUMNS), training["hidden_size"], training["num_layers"], training["dropout"]),
        "Transformer": lambda: TransformerModel(
            len(FEATURE_COLUMNS), training["hidden_size"], training["transformer_heads"],
            training["transformer_layers"], training["dropout"], window,
        ),
    }


def _predict_model(
    model: nn.Module, loader: DataLoader, target_mean: float, target_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    predicted_returns, logits, actual_returns = [], [], []
    with torch.no_grad():
        for x, target, _ in loader:
            predicted_return, predicted_direction = model(x.to(device))
            predicted_returns.extend(predicted_return.cpu().numpy())
            logits.extend(predicted_direction.cpu().numpy())
            actual_returns.extend(target.numpy())
    return (
        np.asarray(predicted_returns) * target_std + target_mean,
        np.asarray(logits),
        np.asarray(actual_returns) * target_std + target_mean,
    )


def _metrics_for_predictions(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    previous_prices: np.ndarray,
    direction_logits: np.ndarray | None,
) -> dict[str, float]:
    actual_prices = previous_prices * np.exp(actual_returns)
    predicted_prices = previous_prices * np.exp(predicted_returns)
    return regression_metrics(
        actual_returns, predicted_returns, actual_prices, predicted_prices, direction_logits
    )


def _train_one(
    name: str,
    constructor: Callable[[], nn.Module],
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: dict,
    checkpoint_path: str,
) -> tuple[dict[str, float], dict[str, list[float]], bool, np.ndarray, np.ndarray]:
    training = config["training"]
    loaders, target_mean, target_std = _loaders(
        train_frame, evaluation_frame, config["window_size"], training["batch_size"]
    )
    history, early_stopped = train_model(
        constructor(), loaders["train"], loaders["evaluation"], training["epochs"],
        training["learning_rate"], training["weight_decay"], training["patience"],
        training["scheduler_factor"], training["scheduler_patience"], checkpoint_path,
        target_mean, target_std,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = constructor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predicted_returns, logits, actual_returns = _predict_model(
        model, loaders["evaluation"], target_mean, target_std
    )
    previous_prices = evaluation_frame["Close"].to_numpy()[config["window_size"] - 1:-1]
    metrics = _metrics_for_predictions(actual_returns, predicted_returns, previous_prices, logits)
    return metrics, history, early_stopped, predicted_returns, actual_returns


def _log_run(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def run_experiment(ticker: str, config: dict) -> None:
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = results_dir / ticker.replace(".", "_")
    ticker_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"
    data = add_features(load_ticker_data(ticker, config["data_dir"]))
    train_frame, validation_frame, test_frame = chronological_split(
        data, tuple(config["split_ratios"])
    )
    _log_run(
        log_path,
        f"{ticker} rows={len(data)} train={len(train_frame)} validation={len(validation_frame)} "
        f"test={len(test_frame)} train_dates={train_frame.index[0].date()}..{train_frame.index[-1].date()} "
        f"validation_dates={validation_frame.index[0].date()}..{validation_frame.index[-1].date()} "
        f"test_dates={test_frame.index[0].date()}..{test_frame.index[-1].date()}",
    )
    window = config["window_size"]
    constructors = _constructors(config["training"], window)

    development = pd.concat([train_frame, validation_frame])
    initial = len(development) // 2
    validation_size = (len(development) - initial) // config["walk_forward_folds"]
    folds = walk_forward_splits(len(development), initial, validation_size)
    if len(folds) < config["walk_forward_folds"]:
        raise ValueError("Configured walk-forward setup produced fewer than five folds.")
    fold_metrics: dict[str, list[dict[str, float]]] = {
        **{name: [] for name in constructors},
        "ZeroReturn": [],
        "XGBoost": [],
    }
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_train = development.iloc[train_indices]
        fold_validation = development.iloc[validation_indices]
        fold_config = dict(config)
        fold_config["training"] = dict(config["training"])
        fold_config["training"]["epochs"] = config["training"].get(
            "walk_forward_epochs", config["training"]["epochs"]
        )
        _log_run(
            log_path,
            f"{ticker} fold={fold_number} rows_train={len(fold_train)} "
            f"rows_validation={len(fold_validation)} "
            f"train_dates={fold_train.index[0].date()}..{fold_train.index[-1].date()} "
            f"validation_dates={fold_validation.index[0].date()}..{fold_validation.index[-1].date()}",
        )
        for name, constructor in constructors.items():
            checkpoint = ticker_dir / f"fold_{fold_number}_{name.lower()}.pt"
            metrics, history, early_stopped, _, _ = _train_one(
                name, constructor, fold_train, fold_validation, fold_config, str(checkpoint)
            )
            fold_metrics[name].append(metrics)
            _log_run(
                log_path,
                f"{ticker} fold={fold_number} model={name} epochs_trained={len(history['train'])} "
                f"early_stopping={early_stopped} final_train_loss={history['train'][-1]:.8f} "
                f"final_validation_loss={history['validation'][-1]:.8f}",
            )
        fold_actual_returns = fold_validation["log_return"].to_numpy()[window:]
        fold_previous_prices = fold_validation["Close"].to_numpy()[window - 1:-1]
        zero_returns = zero_return_forecast(len(fold_actual_returns))
        fold_metrics["ZeroReturn"].append(
            _metrics_for_predictions(
                fold_actual_returns, zero_returns, fold_previous_prices, None
            )
        )
        xgb_returns = xgboost_forecast(
            fold_train, fold_validation.iloc[window - 1:], FEATURE_COLUMNS, "log_return"
        )
        fold_metrics["XGBoost"].append(
            _metrics_for_predictions(
                fold_actual_returns, xgb_returns, fold_previous_prices, None
            )
        )
    walk_forward_rows = []
    for name, metrics_list in fold_metrics.items():
        averaged = {"Model": name}
        averaged.update({key: float(np.mean([m[key] for m in metrics_list])) for key in metrics_list[0]})
        walk_forward_rows.append(averaged)
    pd.DataFrame(walk_forward_rows).to_csv(ticker_dir / "walk_forward_metrics.csv", index=False)

    results: dict[str, dict[str, float]] = {}
    validation_results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    test_actual_returns = test_frame["log_return"].to_numpy()[window:]
    previous_prices = test_frame["Close"].to_numpy()[window - 1:-1]
    for name, constructor in constructors.items():
        checkpoint = ticker_dir / f"{name.lower()}.pt"
        metrics, history, early_stopped, predicted_returns, actual_returns = _train_one(
            name, constructor, train_frame, validation_frame, config, str(checkpoint)
        )
        validation_results[name] = metrics
        # Refit the final model on the training split and validate on validation split;
        # test data remains untouched until this evaluation.
        test_loaders, target_mean, target_std = _loaders(
            train_frame, test_frame, window, config["training"]["batch_size"]
        )
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        model = constructor()
        model.load_state_dict(checkpoint_data["state_dict"])
        predicted_returns, logits, actual_returns = _predict_model(
            model, test_loaders["evaluation"], target_mean, target_std
        )
        results[name] = _metrics_for_predictions(
            actual_returns, predicted_returns, previous_prices, logits
        )
        predictions[name] = previous_prices * np.exp(predicted_returns)
        plot_training_history(history, str(ticker_dir / f"{name.lower()}_loss.png"))
        _log_run(
            log_path,
            f"{ticker} final model={name} epochs_trained={len(history['train'])} "
            f"early_stopping={early_stopped} final_train_loss={history['train'][-1]:.8f} "
            f"final_validation_loss={history['validation'][-1]:.8f}",
        )

    zero_returns = zero_return_forecast(len(test_actual_returns))
    results["ZeroReturn"] = _metrics_for_predictions(
        test_actual_returns, zero_returns, previous_prices, None
    )
    results["YesterdayPrice"] = _metrics_for_predictions(
        test_actual_returns, zero_returns, previous_prices, None
    )
    arima_returns = arima_forecast(train_frame["log_return"], len(test_actual_returns))
    results["ARIMA"] = _metrics_for_predictions(
        test_actual_returns, arima_returns, previous_prices, None
    )
    xgb_returns = xgboost_forecast(
        train_frame, test_frame.iloc[window - 1:], FEATURE_COLUMNS, "log_return"
    )
    results["XGBoost"] = _metrics_for_predictions(
        test_actual_returns, xgb_returns, previous_prices, None
    )
    validation_actual_returns = validation_frame["log_return"].to_numpy()[window:]
    validation_previous_prices = validation_frame["Close"].to_numpy()[window - 1:-1]
    validation_zero_returns = zero_return_forecast(len(validation_actual_returns))
    validation_results["ZeroReturn"] = _metrics_for_predictions(
        validation_actual_returns,
        validation_zero_returns,
        validation_previous_prices,
        None,
    )
    validation_xgb_returns = xgboost_forecast(
        train_frame, validation_frame.iloc[window - 1:], FEATURE_COLUMNS, "log_return"
    )
    validation_results["XGBoost"] = _metrics_for_predictions(
        validation_actual_returns,
        validation_xgb_returns,
        validation_previous_prices,
        None,
    )
    predictions["ZeroReturn"] = previous_prices
    predictions["YesterdayPrice"] = yesterday_price_forecast(previous_prices)
    predictions["ARIMA"] = previous_prices * np.exp(arima_returns)
    predictions["XGBoost"] = previous_prices * np.exp(xgb_returns)
    comparison = compare_models(results)
    comparison.to_csv(ticker_dir / "test_metrics.csv", index=False)
    plot_predictions(
        test_frame.index[window:], test_frame["Close"].to_numpy()[window:],
        predictions, str(ticker_dir / "predictions.png"),
    )
    selected_models = {"LSTM", "GRU", "Transformer", "XGBoost", "ZeroReturn"}
    best_name = min(
        selected_models,
        key=lambda model_name: validation_results[model_name]["ReturnRMSE"],
    )
    _log_run(log_path, f"{ticker} backtest_selected_model={best_name} selection_metric=validation_ReturnRMSE")
    backtest_metrics, curves = backtest_long_flat(
        test_frame["Close"].iloc[window - 1:],
        np.r_[previous_prices[0], predictions[best_name]],
        config["backtest"]["transaction_cost"], config["backtest"]["annualization_factor"],
    )
    pd.DataFrame([backtest_metrics]).to_csv(ticker_dir / "backtest_metrics.csv", index=False)
    curves.to_csv(ticker_dir / "equity_curve.csv", index=False)
    plot_equity_curves(
        test_frame.index[window - 1:], curves["strategy_equity"].to_numpy(),
        curves["buy_and_hold_equity"].to_numpy(), str(ticker_dir / "equity_curve.png"),
    )


if __name__ == "__main__":
    with open("config.yaml", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    set_seed(settings["seed"])
    results_path = Path(settings["results_dir"])
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "run_log.txt").write_text("", encoding="utf-8")
    for configured_ticker in settings["tickers"]:
        run_experiment(configured_ticker, settings)
