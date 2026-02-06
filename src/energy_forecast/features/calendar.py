"""Calendar and temporal feature engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from energy_forecast.features.base import BaseFeatureEngineer


class CalendarFeatureEngineer(BaseFeatureEngineer):
    """Generates datetime, cyclical, holiday, and business hour features.

    Args:
        config: Calendar feature configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate calendar features.

        Args:
            X: DataFrame with datetime index.

        Returns:
            DataFrame with calendar features added.
        """
        raise NotImplementedError
