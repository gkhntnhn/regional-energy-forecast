"""EPIAS Transparency Platform API client."""

from __future__ import annotations

import pandas as pd


class EpiasClient:
    """Fetches market data from EPIAS REST API with caching.

    Args:
        username: EPIAS account username.
        password: EPIAS account password.
    """

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch EPIAS data for date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with EPIAS market variables.
        """
        raise NotImplementedError
