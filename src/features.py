"""Technical-indicator feature engineering."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return", "log_return", "ma_5", "ma_20", "ma_50", "rsi", "macd",
    "macd_signal", "bb_middle", "bb_upper", "bb_lower", "volatility", "volume_change",
]


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add indicators and remove rows whose rolling windows are incomplete."""
    frame = data.copy()
    close = frame["Close"]
    frame["return"] = close.pct_change()
    frame["log_return"] = np.log(close).diff()
    for window in (5, 20, 50):
        frame[f"ma_{window}"] = close.rolling(window).mean()
    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    frame["rsi"] = 100 - (100 / (1 + relative_strength))
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["bb_middle"] = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    frame["bb_upper"] = frame["bb_middle"] + 2 * band_std
    frame["bb_lower"] = frame["bb_middle"] - 2 * band_std
    frame["volatility"] = frame["log_return"].rolling(20).std()
    frame["volume_change"] = frame["Volume"].pct_change()
    return frame.replace([np.inf, -np.inf], np.nan).dropna()

