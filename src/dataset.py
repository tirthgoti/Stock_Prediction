"""Chronological splits and sliding-window PyTorch datasets."""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Return a feature window, next close, and next-day direction label."""

    def __init__(self, features: np.ndarray, closes: np.ndarray, window_size: int) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.closes = torch.tensor(closes, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        """Return the number of valid next-day targets."""
        return len(self.features) - self.window_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one causal sequence and its two targets."""
        end = index + self.window_size
        x = self.features[index:end]
        target = self.closes[end]
        direction = (target > self.closes[end - 1]).float()
        return x, target, direction


def chronological_split(
    data: pd.DataFrame, ratios: tuple[float, float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data in time order; random splitting would leak future patterns."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.")
    first = int(len(data) * ratios[0])
    second = first + int(len(data) * ratios[1])
    return data.iloc[:first], data.iloc[first:second], data.iloc[second:]


def scale_splits(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using training rows only to prevent data leakage."""
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train[columns])
    validation_values = scaler.transform(validation[columns])
    test_values = scaler.transform(test[columns])
    return train_values, validation_values, test_values, scaler

