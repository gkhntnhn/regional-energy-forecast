"""Open-Meteo weather API client."""

from __future__ import annotations

import pandas as pd


class OpenMeteoClient:
    """Fetches weather data from Open-Meteo API with caching.

    Args:
        config: OpenMeteo configuration dictionary.
    """

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical weather data.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Weighted-average weather DataFrame for Uludag region.
        """
        raise NotImplementedError

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch weather forecast for T and T+1.

        Args:
            forecast_days: Number of days to forecast.

        Returns:
            Weighted-average weather forecast DataFrame.
        """
        raise NotImplementedError
