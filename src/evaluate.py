"""Evaluation metrics and prediction plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, previous: np.ndarray) -> dict[str, float]:
    """Calculate price errors and next-day directional accuracy."""
    error = predicted - actual
    nonzero = np.where(actual != 0, actual, np.finfo(float).eps)
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(np.abs(error / nonzero)) * 100),
        "DirectionalAccuracy": float(np.mean((predicted > previous) == (actual > previous)) * 100),
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

