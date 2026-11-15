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
) -> tuple[dict[str, float], pd.DataFrame]:
    """Backtest long/flat predictions and calculate fair buy-and-hold metrics."""
    prices = actual_closes.to_numpy()
    if len(predicted_closes) != len(prices):
        raise ValueError("Prediction and price lengths must match.")
    # predicted_closes[i] forecasts close[i] from close[i - 1]. Therefore the
    # signal created at close t is explicitly stored at position t + 1.
    signal_for_next_day = predicted_closes[1:] > prices[:-1]
    position = np.r_[0.0, signal_for_next_day.astype(float)]
    returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    strategy_returns = position * returns - turnover * transaction_cost
    equity = pd.Series((1 + strategy_returns).cumprod())
    buy_hold_returns = returns
    buy_hold_equity = pd.Series((1 + buy_hold_returns).cumprod())
    sharpe = float(np.sqrt(annualization_factor) * strategy_returns.mean() / (strategy_returns.std() + 1e-12))
    buy_hold_sharpe = float(
        np.sqrt(annualization_factor) * buy_hold_returns.mean()
        / (buy_hold_returns.std() + 1e-12)
    )
    curves = pd.DataFrame({"strategy_equity": equity, "buy_and_hold_equity": buy_hold_equity})
    return {
        "strategy_total_return": float(equity.iloc[-1] - 1),
        "strategy_sharpe_ratio": sharpe,
        "strategy_max_drawdown": max_drawdown(equity),
        "trade_count": float(np.count_nonzero(turnover)),
        "market_exposure_pct": float(position.mean() * 100),
        "buy_and_hold_total_return": float(buy_hold_equity.iloc[-1] - 1),
        "buy_and_hold_sharpe_ratio": buy_hold_sharpe,
        "buy_and_hold_max_drawdown": max_drawdown(buy_hold_equity),
    }, curves

