"""Temporal Fusion Transformer forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecast.models.base import BaseForecaster


class TFTForecaster(BaseForecaster):
    """TFT-based hourly consumption forecaster with uncertainty quantification.

    Args:
        config: TFT configuration from tft.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train TFT model."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with quantiles using trained TFT model."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save TFT model checkpoint (.ckpt format)."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load TFT model from checkpoint."""
        raise NotImplementedError
