"""Weather feature engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from energy_forecast.features.base import BaseFeatureEngineer


class WeatherFeatureEngineer(BaseFeatureEngineer):
    """Generates HDD/CDD, comfort index, rolling, and severity features.

    NOTE: Weather forecast data is NOT leakage — available from OpenMeteo
    at prediction time.

    Args:
        config: Weather feature configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate weather features.

        Args:
            X: DataFrame with weather columns.

        Returns:
            DataFrame with weather features added.
        """
        raise NotImplementedError
