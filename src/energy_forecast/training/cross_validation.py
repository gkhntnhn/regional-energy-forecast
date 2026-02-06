"""Time Series Cross-Validation with expanding window."""

from __future__ import annotations

from typing import Any

import pandas as pd


class TimeSeriesCrossValidator:
    """Expanding-window time series cross-validation.

    CRITICAL: Never shuffles data. Respects temporal ordering (has_time=true).

    Args:
        n_splits: Number of CV splits.
        val_period_days: Validation period length in days.
        test_period_days: Test period length in days.
    """

    def __init__(
        self,
        n_splits: int = 12,
        val_period_days: int = 30,
        test_period_days: int = 30,
    ) -> None:
        self.n_splits = n_splits
        self.val_period_days = val_period_days
        self.test_period_days = test_period_days

    def split(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate train/val/test splits.

        Args:
            df: Full dataset with datetime index.

        Returns:
            List of split dictionaries with train/val/test indices.
        """
        raise NotImplementedError
