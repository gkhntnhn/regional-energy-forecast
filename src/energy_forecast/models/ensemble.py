"""Weighted ensemble of multiple forecasters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecast.models.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """Weighted-average ensemble of CatBoost, Prophet, and TFT.

    Weights are optimized on validation set via scipy.optimize.minimize.
    Supports graceful degradation when a sub-model fails.

    Args:
        config: Ensemble configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.weights: dict[str, float] = {}

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Optimize ensemble weights on validation data."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate weighted-average ensemble prediction."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save ensemble weights."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load ensemble weights."""
        raise NotImplementedError
