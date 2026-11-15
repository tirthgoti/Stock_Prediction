"""Evaluation metrics and prediction plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def regression_metrics(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    direction_logits: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate return, reconstructed-price, and direct direction-head metrics."""
    error = predicted_returns - actual_returns
    predicted_direction = predicted_returns > 0 if direction_logits is None else direction_logits >= 0
    actual_direction = actual_returns > 0
    tn, fp, fn, tp = confusion_matrix(actual_direction, predicted_direction, labels=[False, True]).ravel()
    return {
        "ReturnRMSE": float(np.sqrt(np.mean(error**2))),
        "ReturnMAE": float(np.mean(np.abs(error))),
        "PriceRMSE": float(np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))),
        "PriceMAE": float(np.mean(np.abs(predicted_prices - actual_prices))),
        "DirectionAccuracy": float(accuracy_score(actual_direction, predicted_direction) * 100),
        "DirectionPrecision": float(precision_score(actual_direction, predicted_direction, zero_division=0)),
        "DirectionRecall": float(recall_score(actual_direction, predicted_direction, zero_division=0)),
        "DirectionF1": float(f1_score(actual_direction, predicted_direction, zero_division=0)),
        "DirectionTN": float(tn),
        "DirectionFP": float(fp),
        "DirectionFN": float(fn),
        "DirectionTP": float(tp),
    }


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert metric dictionaries into one readable comparison table."""
    return pd.DataFrame(results).T.rename_axis("Model").reset_index()


def plot_predictions(
    dates: pd.Index, actual: np.ndarray, predictions: dict[str, np.ndarray], output_path: str
) -> None:
    """Save actual-versus-predicted price curves."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual", linewidth=2)
    for name, values in predictions.items():
        plt.plot(dates, values, label=name, alpha=0.8)
    plt.title("Next-day close predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_equity_curves(
    dates: pd.Index,
    strategy_equity: np.ndarray,
    buy_and_hold_equity: np.ndarray,
    output_path: str,
) -> None:
    """Save strategy and buy-and-hold equity curves."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, strategy_equity, label="Strategy")
    plt.plot(dates, buy_and_hold_equity, label="Buy and hold")
    plt.title("Backtest equity curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

