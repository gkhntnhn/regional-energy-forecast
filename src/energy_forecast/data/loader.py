"""Consumption data loader from Excel files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads and validates consumption Excel data.

    Args:
        config: Data loader configuration dictionary.
    """

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def load_excel(self, path: Path) -> pd.DataFrame:
        """Load consumption data from Excel file.

        Args:
            path: Path to .xlsx file.

        Returns:
            Validated DataFrame with datetime index.
        """
        raise NotImplementedError
