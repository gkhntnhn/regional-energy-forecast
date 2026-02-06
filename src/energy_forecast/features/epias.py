"""EPIAS market data feature engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from energy_forecast.features.base import BaseFeatureEngineer


class EpiasFeatureEngineer(BaseFeatureEngineer):
    """Generates lag, rolling, and expanding features from EPIAS data.

    CRITICAL: All lags use min_lag=48. Raw EPIAS values are DROPPED
    after feature creation — only derived features remain.

    Args:
        config: EPIAS feature configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate EPIAS-derived features.

        Args:
            X: DataFrame with raw EPIAS columns.

        Returns:
            DataFrame with derived features (raw values dropped).
        """
        raise NotImplementedError
