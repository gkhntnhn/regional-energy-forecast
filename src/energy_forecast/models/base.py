"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseForecaster(ABC):
    """Abstract base for all forecasting models.

    Every model must implement train, predict, save, and load.

    Args:
        config: Model-specific configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train the model.

        Args:
            train_df: Training data with features and target.
            val_df: Optional validation data for early stopping.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate 48-hour forecast.

        Args:
            X: Feature DataFrame for prediction period.

        Returns:
            DataFrame with hourly predictions.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model artifacts to disk.

        Args:
            path: Directory to save model files.
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model artifacts from disk.

        Args:
            path: Directory containing model files.
        """
        ...
