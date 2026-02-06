"""Consumption lag and rolling feature engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from energy_forecast.features.base import BaseFeatureEngineer


class ConsumptionFeatureEngineer(BaseFeatureEngineer):
    """Generates lag, rolling, EWMA, momentum, and volatility features.

    CRITICAL: All lags use min_lag=48 to prevent data leakage.
    Rolling windows apply shift(1) BEFORE rolling().

    Args:
        config: Consumption feature configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate consumption features.

        Args:
            X: DataFrame with 'consumption' column.

        Returns:
            DataFrame with consumption features added.
        """
        raise NotImplementedError
