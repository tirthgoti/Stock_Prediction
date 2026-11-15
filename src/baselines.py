"""Classical forecasting baselines."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


def zero_return_forecast(length: int) -> np.ndarray:
    """Predict a zero next-day log return."""
    return np.zeros(length, dtype=float)


def yesterday_price_forecast(previous_closes: np.ndarray) -> np.ndarray:
    """Keep the original yesterday's-price persistence baseline."""
    return np.asarray(previous_closes, dtype=float)


def arima_forecast(train: pd.Series, horizon: int, order: tuple[int, int, int] = (5, 1, 0)) -> np.ndarray:
    """Fit ARIMA on training log returns and forecast the test horizon."""
    # Use a plain numeric index: market dates have weekends/holidays, so they
    # do not provide a reliable fixed frequency for statsmodels forecasting.
    model = ARIMA(train.to_numpy(dtype=float), order=order).fit()
    return np.asarray(model.forecast(steps=horizon), dtype=float)


def xgboost_forecast(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str = "log_return"
) -> np.ndarray:
    """Predict each test day's log return from the previous day's features."""
    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        random_state=42,
    )
    # Shift the target by one day so this baseline has the same causal task.
    model.fit(train[features].iloc[:-1], train[target].iloc[1:])
    return model.predict(test[features].iloc[:-1])
