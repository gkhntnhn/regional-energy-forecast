"""Solar irradiance feature engineering using pvlib."""

from __future__ import annotations

from typing import Any

import pandas as pd

from energy_forecast.features.base import BaseFeatureEngineer


class SolarFeatureEngineer(BaseFeatureEngineer):
    """Generates GHI/DNI/DHI, POA, clearness index, and cloud proxy features.

    NOTE: Solar features are NOT leakage — deterministic astronomical
    calculations that are exact for any date/time.

    Args:
        config: Solar feature configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate solar features.

        Args:
            X: DataFrame with datetime index.

        Returns:
            DataFrame with solar features added.
        """
        raise NotImplementedError
