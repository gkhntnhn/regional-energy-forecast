"""Calendar-month aligned expanding-window TSCV splitter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class SplitInfo:
    """Single CV split boundary info."""

    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class TimeSeriesSplitter:
    """Calendar-month aligned expanding-window TSCV splitter.

    All models (CatBoost, Prophet, TFT) use this same splitter.

    Each split has val and test periods aligned to full calendar months.
    Train always starts from data beginning and expands (backward-anchored).

    Example with 3-year data (2022-01 to 2024-12), n_splits=12:
      Split  0: train=[2022-01..2023-01] val=2023-02 test=2023-03
      Split  1: train=[2022-01..2023-02] val=2023-03 test=2023-04
      ...
      Split 11: train=[2022-01..2023-12] val=2024-01 test=2024-02
    """

    def __init__(
        self,
        n_splits: int = 12,
        val_months: int = 1,
        test_months: int = 1,
        gap_hours: int = 0,
    ) -> None:
        """Initialize splitter.

        Args:
            n_splits: Number of expanding-window splits.
            val_months: Validation period in calendar months.
            test_months: Test period in calendar months.
            gap_hours: Gap between train/val and val/test in hours.
        """
        self.n_splits = n_splits
        self.val_months = val_months
        self.test_months = test_months
        self.gap_hours = gap_hours

    @classmethod
    def from_config(cls, cv_config: object) -> TimeSeriesSplitter:
        """Construct from CrossValidationConfig.

        Args:
            cv_config: CrossValidationConfig instance.

        Returns:
            Configured TimeSeriesSplitter.
        """
        from energy_forecast.config import CrossValidationConfig

        if not isinstance(cv_config, CrossValidationConfig):
            msg = f"Expected CrossValidationConfig, got {type(cv_config).__name__}"
            raise TypeError(msg)

        if cv_config.shuffle:
            msg = (
                "shuffle=True is forbidden for time series cross-validation. "
                "Set shuffle=false in cross_validation config."
            )
            raise ValueError(msg)

        return cls(
            n_splits=cv_config.n_splits,
            val_months=cv_config.val_months,
            test_months=cv_config.test_months,
            gap_hours=cv_config.gap_hours,
        )

    def split(self, df: pd.DataFrame) -> list[SplitInfo]:
        """Generate split boundary infos from the data.

        Works backward from the last complete month in the data,
        producing ``n_splits`` expanding-window folds.

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            List of SplitInfo ordered oldest-first.

        Raises:
            ValueError: If index is not DatetimeIndex or data is insufficient.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame must have a DatetimeIndex."
            raise ValueError(msg)

        data_start = pd.Timestamp(df.index.min())
        data_end = pd.Timestamp(df.index.max())

        # Last complete month end: last day of the month containing data_end,
        # but only if data_end is actually that month's last hour.
        # Use the start of the next month minus 1 hour to find the last complete month.
        last_month_start = pd.Timestamp(year=data_end.year, month=data_end.month, day=1)
        last_month_last_hour = last_month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)

        if data_end >= last_month_last_hour:
            # The last month is complete
            anchor_end = last_month_last_hour
        else:
            # Roll back to end of previous month
            prev_month_start = last_month_start - pd.offsets.MonthBegin(1)
            anchor_end = prev_month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)

        splits: list[SplitInfo] = []

        for i in range(self.n_splits):
            # Each earlier split shifts 1 month back from the newest
            shift = self.n_splits - 1 - i

            # Test block end month: anchor shifted back by shift months
            test_block_end_month = pd.Timestamp(
                year=anchor_end.year, month=anchor_end.month, day=1
            ) - pd.DateOffset(months=shift)
            # test spans test_months months ending at test_block_end_month
            test_start_month = test_block_end_month - pd.DateOffset(months=self.test_months - 1)
            test_start = pd.Timestamp(
                year=test_start_month.year,
                month=test_start_month.month,
                day=1,
            )
            test_end = test_block_end_month + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)

            # Validation block: val_months months before test_start
            val_end_month = test_start - pd.DateOffset(months=1)
            val_start_month = val_end_month - pd.DateOffset(months=self.val_months - 1)
            val_start = pd.Timestamp(
                year=val_start_month.year,
                month=val_start_month.month,
                day=1,
            )
            val_end_raw = (
                pd.Timestamp(year=val_end_month.year, month=val_end_month.month, day=1)
                + pd.offsets.MonthEnd(0)
                + pd.Timedelta(hours=23)
            )
            val_end = val_end_raw - pd.Timedelta(hours=self.gap_hours)

            # Train block
            train_start = data_start
            train_end = val_start - pd.Timedelta(hours=1) - pd.Timedelta(hours=self.gap_hours)

            if train_end < train_start:
                msg = (
                    f"Insufficient data for {self.n_splits} splits. "
                    f"Split {i}: train_end ({train_end}) < train_start ({train_start})."
                )
                raise ValueError(msg)

            splits.append(
                SplitInfo(
                    split_idx=i,
                    train_start=pd.Timestamp(train_start),
                    train_end=pd.Timestamp(train_end),
                    val_start=pd.Timestamp(val_start),
                    val_end=pd.Timestamp(val_end),
                    test_start=pd.Timestamp(test_start),
                    test_end=pd.Timestamp(test_end),
                )
            )

        logger.info(
            "Generated {} TSCV splits (val={}mo, test={}mo, gap={}h)",
            len(splits),
            self.val_months,
            self.test_months,
            self.gap_hours,
        )
        for s in splits:
            logger.debug(
                "  Split {:>2d}: train=[{}..{}] val=[{}..{}] test=[{}..{}]",
                s.split_idx,
                s.train_start.date(),
                s.train_end.date(),
                s.val_start.date(),
                s.val_end.date(),
                s.test_start.date(),
                s.test_end.date(),
            )

        return splits

    def iter_splits(
        self,
        df: pd.DataFrame,
    ) -> Iterator[tuple[SplitInfo, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Yield (info, train_df, val_df, test_df) for each split.

        Args:
            df: DataFrame with DatetimeIndex.

        Yields:
            Tuple of (SplitInfo, train DataFrame, val DataFrame, test DataFrame).
        """
        for info in self.split(df):
            train_df: pd.DataFrame = df.loc[info.train_start : info.train_end]
            val_df: pd.DataFrame = df.loc[info.val_start : info.val_end]
            test_df: pd.DataFrame = df.loc[info.test_start : info.test_end]
            yield info, train_df, val_df, test_df
