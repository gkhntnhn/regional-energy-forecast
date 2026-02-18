"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

# Default target column name - can be overridden via config
DEFAULT_TARGET_COL = "consumption"

# Standard prediction output column name — all models MUST use this
PREDICTION_COL = "consumption_mwh"


class BaseForecaster(ABC):
    """Abstract base for all forecasting models.

    Every model must implement train, predict, save, and load.

    Args:
        config: Model-specific configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._target_col: str = config.get("target_col", DEFAULT_TARGET_COL)

    @property
    def target_col(self) -> str:
        """Target column name for training and prediction."""
        return self._target_col

    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, float] | None:
        """Train the model.

        Args:
            train_df: Training data with features and target.
            val_df: Optional validation data for early stopping.
            **kwargs: Additional model-specific arguments.

        Returns:
            Optional dict of training metrics (e.g., loss, MAPE).
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
