"""Learning-only long/flat backtesting utilities."""

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Return the largest peak-to-trough percentage decline."""
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def backtest_long_flat(
    actual_closes: pd.Series,
    predicted_closes: np.ndarray,
    transaction_cost: float = 0.001,
    annualization_factor: int = 252,
) -> dict[str, float]:
    """Backtest buy-if-predicted-up, otherwise-cash, including trading costs."""
    prices = actual_closes.to_numpy()
    if len(predicted_closes) != len(prices):
        raise ValueError("Prediction and price lengths must match.")
    predicted_up = np.r_[False, predicted_closes[1:] > prices[:-1]]
    returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
    position = predicted_up.astype(float)
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    strategy_returns = position * returns - turnover * transaction_cost
    equity = pd.Series((1 + strategy_returns).cumprod())
    sharpe = float(np.sqrt(annualization_factor) * strategy_returns.mean() / (strategy_returns.std() + 1e-12))
    buy_hold = float((prices[-1] / prices[0]) - 1)
    return {
        "strategy_total_return": float(equity.iloc[-1] - 1),
        "buy_and_hold_return": buy_hold,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(equity),
    }

