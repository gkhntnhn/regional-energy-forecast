"""CatBoost gradient boosting forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecast.models.base import BaseForecaster


class CatBoostForecaster(BaseForecaster):
    """CatBoost-based hourly consumption forecaster.

    Args:
        config: CatBoost configuration from catboost.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train CatBoost model."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using trained CatBoost model."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save CatBoost model (.cbm format)."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load CatBoost model from .cbm file."""
        raise NotImplementedError
