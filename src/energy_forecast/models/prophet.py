"""Prophet trend and seasonality forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecast.models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Prophet-based hourly consumption forecaster.

    Args:
        config: Prophet configuration from prophet.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train Prophet model."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using trained Prophet model."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save Prophet model (.pkl format)."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load Prophet model from .pkl file."""
        raise NotImplementedError
