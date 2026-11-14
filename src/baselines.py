"""Classical forecasting baselines."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


def naive_forecast(closes: pd.Series) -> np.ndarray:
    """Predict tomorrow's close as today's close."""
    return closes.shift(1).to_numpy()


def arima_forecast(train: pd.Series, test: pd.Series, order: tuple[int, int, int] = (5, 1, 0)) -> np.ndarray:
    """Fit ARIMA on the training history and forecast the test horizon."""
    # Use a plain numeric index: market dates have weekends/holidays, so they
    # do not provide a reliable fixed frequency for statsmodels forecasting.
    model = ARIMA(train.to_numpy(dtype=float), order=order).fit()
    return np.asarray(model.forecast(steps=len(test)), dtype=float)


def xgboost_forecast(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str = "Close"
) -> np.ndarray:
    """Predict each test day's close from the previous day's features."""
    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        random_state=42,
    )
    # Shift the target by one day so this baseline has the same forecasting
    # task as the neural models and cannot use today's close as an input.
    model.fit(train[features].iloc[:-1], train[target].iloc[1:])
    return model.predict(test[features].iloc[:-1])
