"""Training, early stopping, checkpointing, and walk-forward utilities."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.baselines import arima_forecast, xgboost_forecast
from src.data_loader import load_ticker_data
from src.dataset import SequenceDataset, chronological_split, scale_splits
from src.evaluate import compare_models, plot_predictions, regression_metrics
from src.features import FEATURE_COLUMNS, add_features
from src.models import GRUModel, LSTMModel, TransformerModel


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for repeatable experiments."""
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
) -> dict[str, list[float]]:
    """Train with combined price/direction loss and restore the best checkpoint."""
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
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    for _ in range(epochs):
        model.train()
        train_losses = []
        for x, price, direction in train_loader:
            x, price, direction = x.to(device), price.to(device), direction.to(device)
            optimizer.zero_grad()
            predicted_price, predicted_direction = model(x)
            loss = mse(predicted_price, price) + bce(predicted_direction, direction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for x, price, direction in validation_loader:
                predicted_price, predicted_direction = model(x.to(device))
                validation_losses.append(
                    (mse(predicted_price, price.to(device)) + bce(predicted_direction, direction.to(device))).item()
                )
        train_loss, validation_loss = float(np.mean(train_losses)), float(np.mean(validation_losses))
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def plot_training_history(history: dict[str, list[float]], output_path: str) -> None:
    """Save train-versus-validation loss, useful for spotting overfitting."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["validation"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Combined loss")
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


def predict_model(
    model: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate price, direction-logit, and actual-price arrays."""
    device = next(model.parameters()).device
    model.eval()
    prices, logits, actuals = [], [], []
    with torch.no_grad():
        for x, price, _ in loader:
            predicted_price, predicted_direction = model(x.to(device))
            prices.extend(predicted_price.cpu().numpy())
            logits.extend(predicted_direction.cpu().numpy())
            actuals.extend(price.numpy())
    return np.asarray(prices), np.asarray(logits), np.asarray(actuals)


def run_experiment(ticker: str, config: dict) -> None:
    """Run all neural models for one ticker and save honest test artifacts."""
    data = add_features(load_ticker_data(ticker, config["data_dir"]))
    train_frame, validation_frame, test_frame = chronological_split(
        data, tuple(config["split_ratios"])
    )
    train_values, validation_values, test_values, _ = scale_splits(
        train_frame, validation_frame, test_frame, FEATURE_COLUMNS
    )
    window = config["window_size"]
    # Scale the regression target using training prices only. This keeps the
    # neural loss numerically well-conditioned without leaking future prices.
    close_mean = float(train_frame["Close"].mean())
    close_std = float(train_frame["Close"].std())
    if close_std == 0:
        raise ValueError("Training close prices must have non-zero variance.")
    scaled_closes = {
        "train": (train_frame["Close"].to_numpy() - close_mean) / close_std,
        "validation": (validation_frame["Close"].to_numpy() - close_mean) / close_std,
        "test": (test_frame["Close"].to_numpy() - close_mean) / close_std,
    }
    loaders = {
        "train": DataLoader(
            SequenceDataset(train_values, scaled_closes["train"], window),
            batch_size=config["training"]["batch_size"], shuffle=False,
        ),
        "validation": DataLoader(
            SequenceDataset(validation_values, scaled_closes["validation"], window),
            batch_size=config["training"]["batch_size"], shuffle=False,
        ),
        "test": DataLoader(
            SequenceDataset(test_values, scaled_closes["test"], window),
            batch_size=config["training"]["batch_size"], shuffle=False,
        ),
    }
    training = config["training"]
    constructors: dict[str, Callable[[], nn.Module]] = {
        "LSTM": lambda: LSTMModel(len(FEATURE_COLUMNS), training["hidden_size"], training["num_layers"], training["dropout"]),
        "GRU": lambda: GRUModel(len(FEATURE_COLUMNS), training["hidden_size"], training["num_layers"], training["dropout"]),
        "Transformer": lambda: TransformerModel(
            len(FEATURE_COLUMNS), training["hidden_size"], training["transformer_heads"],
            training["transformer_layers"], training["dropout"],
        ),
    }
    output_dir = Path(config["artifacts_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, constructor in constructors.items():
        model = constructor()
        history = train_model(
            model, loaders["train"], loaders["validation"], training["epochs"],
            training["learning_rate"], training["weight_decay"], training["patience"],
            training["scheduler_factor"], training["scheduler_patience"],
            str(output_dir / f"{ticker.replace('.', '_')}_{name.lower()}.pt"),
        )
        plot_training_history(history, str(output_dir / f"{ticker.replace('.', '_')}_{name.lower()}_loss.png"))
        predicted, _, actual = predict_model(model, loaders["test"])
        predicted = predicted * close_std + close_mean
        actual = actual * close_std + close_mean
        previous = test_frame["Close"].to_numpy()[window - 1:-1]
        results[name] = regression_metrics(actual, predicted, previous)
        predictions[name] = predicted
    test_actual = test_frame["Close"].to_numpy()[window:]
    naive = test_frame["Close"].shift(1).to_numpy()[window:]
    results["Naive"] = regression_metrics(test_actual, naive, test_frame["Close"].to_numpy()[window - 1:-1])
    arima = arima_forecast(train_frame["Close"], test_frame["Close"])
    results["ARIMA"] = regression_metrics(
        test_frame["Close"].to_numpy(), arima, test_frame["Close"].shift(1).bfill().to_numpy()
    )
    xgb = xgboost_forecast(train_frame, test_frame, FEATURE_COLUMNS)
    results["XGBoost"] = regression_metrics(
        test_frame["Close"].to_numpy()[1:], xgb, test_frame["Close"].to_numpy()[:-1]
    )
    table = compare_models(results)
    table.to_csv(output_dir / f"{ticker.replace('.', '_')}_metrics.csv", index=False)
    plot_predictions(test_frame.index[window:], test_actual, predictions, str(output_dir / f"{ticker.replace('.', '_')}_predictions.png"))


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    set_seed(settings["seed"])
    for configured_ticker in settings["tickers"]:
        run_experiment(configured_ticker, settings)
