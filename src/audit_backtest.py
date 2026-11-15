"""Audit alignment, model selection, exposure controls, and backtest sensitivity."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.baselines import arima_forecast, xgboost_forecast, zero_return_forecast
from src.data_loader import load_ticker_data
from src.dataset import chronological_split
from src.features import FEATURE_COLUMNS, add_features
from src.models import GRUModel, LSTMModel, TransformerModel
from src.train import _loaders, _metrics_for_predictions, _predict_model


MODEL_NAMES = ["LSTM", "GRU", "Transformer", "XGBoost", "ZeroReturn"]


def _model_from_checkpoint(ticker: str, model_name: str, config: dict, window: int):
    training = config["training"]
    constructors = {
        "LSTM": lambda: LSTMModel(
            len(FEATURE_COLUMNS), training["hidden_size"], training["num_layers"], training["dropout"]
        ),
        "GRU": lambda: GRUModel(
            len(FEATURE_COLUMNS), training["hidden_size"], training["num_layers"], training["dropout"]
        ),
        "Transformer": lambda: TransformerModel(
            len(FEATURE_COLUMNS), training["hidden_size"], training["transformer_heads"],
            training["transformer_layers"], training["dropout"], window,
        ),
    }
    checkpoint_path = (
        Path(config["results_dir"]) / ticker.replace(".", "_") / f"{model_name.lower()}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = constructors[model_name]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["target_mean"], checkpoint["target_std"]


def _predictions(
    ticker: str, model_name: str, train_frame: pd.DataFrame, evaluation_frame: pd.DataFrame, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    window = config["window_size"]
    actual_returns = evaluation_frame["log_return"].to_numpy()[window:]
    previous_prices = evaluation_frame["Close"].to_numpy()[window - 1:-1]
    if model_name in {"LSTM", "GRU", "Transformer"}:
        loaders, target_mean, target_std = _loaders(
            train_frame, evaluation_frame, window, config["training"]["batch_size"]
        )
        model, target_mean, target_std = _model_from_checkpoint(
            ticker, model_name, config, window
        )
        predicted_returns, _, _ = _predict_model(
            model, loaders["evaluation"], target_mean, target_std
        )
    elif model_name == "ZeroReturn":
        predicted_returns = zero_return_forecast(len(actual_returns))
    elif model_name == "XGBoost":
        predicted_returns = xgboost_forecast(
            train_frame, evaluation_frame.iloc[window - 1:], FEATURE_COLUMNS, "log_return"
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return predicted_returns, previous_prices


def _backtest(
    prices: np.ndarray,
    predicted_closes: np.ndarray,
    transaction_cost: float,
    delay: int = 0,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    if len(predicted_closes) != len(prices):
        raise ValueError("Prediction and price lengths must match.")
    signal_for_next_day = predicted_closes[1:] > prices[:-1]
    position = np.zeros(len(prices), dtype=float)
    if delay == 0:
        position[1:] = signal_for_next_day
    elif delay == 1:
        position[2:] = signal_for_next_day[:-1]
    else:
        raise ValueError("Only zero-delay and one-extra-day-delay audits are supported.")
    returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    strategy_returns = position * returns - turnover * transaction_cost
    strategy_equity = (1 + strategy_returns).cumprod()
    buy_hold_equity = (1 + returns).cumprod()
    def sharpe(values: np.ndarray) -> float:
        return float(np.sqrt(252) * values.mean() / (values.std() + 1e-12))
    def drawdown(equity: np.ndarray) -> float:
        return float((equity / np.maximum.accumulate(equity) - 1).min())
    result = {
        "strategy_total_return": float(strategy_equity[-1] - 1),
        "strategy_sharpe_ratio": sharpe(strategy_returns),
        "strategy_max_drawdown": drawdown(strategy_equity),
        "trade_count": float(np.count_nonzero(turnover)),
        "market_exposure_pct": float(position.mean() * 100),
        "buy_and_hold_total_return": float(buy_hold_equity[-1] - 1),
        "buy_and_hold_sharpe_ratio": sharpe(returns),
        "buy_and_hold_max_drawdown": drawdown(buy_hold_equity),
    }
    curves = pd.DataFrame(
        {"strategy_equity": strategy_equity, "buy_and_hold_equity": buy_hold_equity}
    )
    return result, curves, position


def _positive_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 0:
        return []
    cuts = np.sort(rng.choice(np.arange(1, total), parts - 1, replace=False))
    return np.diff(np.r_[0, cuts, total]).tolist()


def _matched_position(
    length: int, ones: int, trades: int, rng: np.random.Generator
) -> np.ndarray:
    if trades == 0:
        return np.zeros(length, dtype=float)
    one_runs = (trades + 1) // 2
    zero_runs = trades // 2 + 1
    one_lengths = _positive_composition(ones, one_runs, rng)
    zero_lengths = _positive_composition(length - ones, zero_runs, rng)
    position = []
    position.extend([0.0] * zero_lengths[0])
    for index in range(one_runs):
        position.extend([1.0] * one_lengths[index])
        if index + 1 < zero_runs:
            position.extend([0.0] * zero_lengths[index + 1])
    return np.asarray(position)


def _out_of_market_periods(
    dates: pd.Index, prices: np.ndarray, position: np.ndarray
) -> tuple[list[dict[str, object]], float]:
    returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
    target_dates = dates[1:]
    out = position[1:] == 0
    periods = []
    start = None
    for index, is_out in enumerate(out):
        if is_out and start is None:
            start = index
        if start is not None and (not is_out or index == len(out) - 1):
            end = index - 1 if not is_out else index
            period_return = float(np.prod(1 + returns[start + 1:end + 2]) - 1)
            periods.append(
                {
                    "start": str(target_dates[start].date()),
                    "end": str(target_dates[end].date()),
                    "avoided_return": period_return,
                }
            )
            start = None
    return periods, float(sum(item["avoided_return"] for item in periods))


def main() -> None:
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    results_dir = Path(config["results_dir"])
    selection_rows = []
    all_model_rows = []
    sensitivity_rows = []
    delay_rows = []
    equity_rows = []
    random_rows = []
    sample_rows = []
    avoided_rows = []
    for ticker in config["tickers"]:
        data = add_features(load_ticker_data(ticker, config["data_dir"]))
        train_frame, validation_frame, test_frame = chronological_split(
            data, tuple(config["split_ratios"])
        )
        window = config["window_size"]
        validation_metrics = {}
        test_predictions = {}
        for model_name in MODEL_NAMES:
            validation_returns, validation_previous = _predictions(
                ticker, model_name, train_frame, validation_frame, config
            )
            validation_actual = validation_frame["log_return"].to_numpy()[window:]
            validation_metrics[model_name] = float(
                np.sqrt(np.mean((validation_returns - validation_actual) ** 2))
            )
            test_predictions[model_name], _ = _predictions(
                ticker, model_name, train_frame, test_frame, config
            )
        selected_model = min(validation_metrics, key=validation_metrics.get)
        selection_rows.extend(
            {"Ticker": ticker, "Model": name, "ValidationReturnRMSE": value,
             "Selected": name == selected_model}
            for name, value in validation_metrics.items()
        )
        prices = test_frame["Close"].to_numpy()[window - 1:]
        equity_dates = test_frame.index[window - 1:]
        test_dates = test_frame.index[window:]
        for model_name, predicted_returns in test_predictions.items():
            predicted_closes = np.r_[prices[0], prices[:-1] * np.exp(predicted_returns)]
            result, _, position = _backtest(
                prices, predicted_closes, config["backtest"]["transaction_cost"]
            )
            all_model_rows.append(
                {"Ticker": ticker, "Model": model_name, "SelectedByValidation": model_name == selected_model, **result}
            )
            for cost in [0.0, 0.001, 0.003]:
                cost_result, _, _ = _backtest(prices, predicted_closes, cost)
                sensitivity_rows.append(
                    {"Ticker": ticker, "SelectedModel": selected_model, "TransactionCost": cost, **cost_result}
                ) if model_name == selected_model else None
            if model_name == selected_model:
                delayed, _, _ = _backtest(
                    prices, predicted_closes, config["backtest"]["transaction_cost"], delay=1
                )
                delay_rows.append({"Ticker": ticker, "Model": selected_model, **delayed})
                periods, avoided_total = _out_of_market_periods(
                    equity_dates, prices, position
                )
                if ticker == "AAPL":
                    avoided_rows.extend(
                        {"Ticker": ticker, **period, "TotalAvoidedReturn": avoided_total}
                        for period in periods
                    )
                for sample_index in [0, 1, len(test_dates) // 2, len(test_dates) - 2, len(test_dates) - 1]:
                    sample_rows.append(
                        {
                            "Ticker": ticker,
                            "Date": str(test_dates[sample_index].date()),
                            "WindowStart": str(test_frame.index[sample_index].date()),
                            "WindowEnd": str(test_frame.index[sample_index + window - 1].date()),
                            "PredictedReturn": float(predicted_returns[sample_index]),
                            "ActualReturn": float(test_frame["log_return"].to_numpy()[window + sample_index]),
                            "Position": float(position[sample_index + 1]),
                        }
                    )
                target_ones = int(position.sum())
                target_trades = int(np.count_nonzero(np.abs(np.diff(np.r_[0.0, position]))))
                random_returns = []
                rng = np.random.default_rng(config["seed"])
                for _ in range(1000):
                    random_position = _matched_position(
                        len(position), target_ones, target_trades, rng
                    )
                    random_turnover = np.abs(np.diff(np.r_[0.0, random_position]))
                    actual_returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
                    random_strategy_returns = (
                        random_position * actual_returns
                        - random_turnover * config["backtest"]["transaction_cost"]
                    )
                    random_returns.append(float((1 + random_strategy_returns).prod() - 1))
                random_returns = np.asarray(random_returns)
                random_rows.append(
                    {
                        "Ticker": ticker,
                        "Model": selected_model,
                        "RandomRuns": 1000,
                        "MatchedTrades": target_trades,
                        "MatchedExposurePct": float(position.mean() * 100),
                        "RandomMeanReturn": float(random_returns.mean()),
                        "RandomStdReturn": float(random_returns.std()),
                        "RandomMinReturn": float(random_returns.min()),
                        "RandomMaxReturn": float(random_returns.max()),
                        "StrategyReturnPercentile": float(
                            np.mean(random_returns <= result["strategy_total_return"]) * 100
                        ),
                    }
                )
                equity_rows.append(
                    {
                        "Ticker": ticker,
                        "StrategyFirstDate": str(equity_dates[0].date()),
                        "StrategyLastDate": str(equity_dates[-1].date()),
                        "StrategyEquityDays": len(equity_dates),
                        "BuyHoldFirstDate": str(equity_dates[0].date()),
                        "BuyHoldLastDate": str(equity_dates[-1].date()),
                        "BuyHoldEquityDays": len(equity_dates),
                    }
                )
    pd.DataFrame(selection_rows).to_csv(results_dir / "validation_model_selection.csv", index=False)
    pd.DataFrame(all_model_rows).to_csv(results_dir / "all_model_backtests.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "backtest_cost_sensitivity.csv", index=False)
    pd.DataFrame(delay_rows).to_csv(results_dir / "delayed_backtest.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(results_dir / "equity_date_audit.csv", index=False)
    pd.DataFrame(random_rows).to_csv(results_dir / "matched_random_baseline.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(results_dir / "prediction_alignment_samples.csv", index=False)
    pd.DataFrame(avoided_rows).to_csv(results_dir / "aapl_avoided_returns.csv", index=False)


if __name__ == "__main__":
    main()
