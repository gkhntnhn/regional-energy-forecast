"""Tests for calendar-month aligned TSCV splitter."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import pytest

from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """3 years hourly data: 2022-01-01 00:00 to 2024-12-31 23:00."""
    idx = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({"consumption": rng.uniform(800, 1600, len(idx))}, index=idx)


@pytest.fixture
def splitter() -> TimeSeriesSplitter:
    """Default splitter with 12 splits."""
    return TimeSeriesSplitter(n_splits=12, val_months=1, test_months=1, gap_hours=0)


class TestBasicSplits:
    """Basic split generation tests."""

    def test_basic_split_count(self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame) -> None:
        """12 splits are generated."""
        splits = splitter.split(sample_df)
        assert len(splits) == 12

    def test_split_temporal_order(
        self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame
    ) -> None:
        """For each split: train_end < val_start <= val_end < test_start <= test_end."""
        splits = splitter.split(sample_df)
        for s in splits:
            assert s.train_end < s.val_start, f"Split {s.split_idx}: train_end >= val_start"
            assert s.val_start <= s.val_end, f"Split {s.split_idx}: val_start > val_end"
            assert s.val_end < s.test_start, f"Split {s.split_idx}: val_end >= test_start"
            assert s.test_start <= s.test_end, f"Split {s.split_idx}: test_start > test_end"

    def test_expanding_window(self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame) -> None:
        """Split 0 train period < split 11 train period (expanding)."""
        splits = splitter.split(sample_df)
        train_0 = splits[0].train_end - splits[0].train_start
        train_11 = splits[11].train_end - splits[11].train_start
        assert train_0 < train_11

    def test_no_overlap(self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame) -> None:
        """No timestamp appears in more than one of train/val/test per split."""
        splits = splitter.split(sample_df)
        for s in splits:
            train = set(pd.date_range(s.train_start, s.train_end, freq="h"))
            val = set(pd.date_range(s.val_start, s.val_end, freq="h"))
            test = set(pd.date_range(s.test_start, s.test_end, freq="h"))
            assert not train & val, f"Split {s.split_idx}: train/val overlap"
            assert not val & test, f"Split {s.split_idx}: val/test overlap"
            assert not train & test, f"Split {s.split_idx}: train/test overlap"


class TestCalendarAlignment:
    """Calendar-month alignment tests."""

    def test_val_is_calendar_month(
        self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame
    ) -> None:
        """val_start is 1st of month 00:00, val_end is last day 23:00."""
        splits = splitter.split(sample_df)
        for s in splits:
            assert s.val_start.day == 1
            assert s.val_start.hour == 0
            last_day = calendar.monthrange(s.val_end.year, s.val_end.month)[1]
            assert s.val_end.day == last_day
            assert s.val_end.hour == 23

    def test_test_is_calendar_month(
        self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame
    ) -> None:
        """test_start is 1st of month 00:00, test_end is last day 23:00."""
        splits = splitter.split(sample_df)
        for s in splits:
            assert s.test_start.day == 1
            assert s.test_start.hour == 0
            last_day = calendar.monthrange(s.test_end.year, s.test_end.month)[1]
            assert s.test_end.day == last_day
            assert s.test_end.hour == 23

    def test_february_28_days(self) -> None:
        """2023 is non-leap: Feb test/val has 28 days."""
        # Data: 2021-01 to 2023-12, use 10 splits so Feb 2023 is covered
        idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h")
        df = pd.DataFrame({"consumption": 1.0}, index=idx)
        sp = TimeSeriesSplitter(n_splits=10, val_months=1, test_months=1)
        splits = sp.split(df)
        # Find a split where test or val is Feb 2023
        feb_test = [s for s in splits if s.test_start.month == 2 and s.test_start.year == 2023]
        feb_val = [s for s in splits if s.val_start.month == 2 and s.val_start.year == 2023]
        found = False
        for s in feb_test:
            assert s.test_end.day == 28
            found = True
        for s in feb_val:
            assert s.val_end.day == 28
            found = True
        assert found, "No split has Feb 2023 as val or test"

    def test_february_29_days(self) -> None:
        """2024 is leap year: Feb has 29 days."""
        # Data: 2022-01 to 2024-12
        idx = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")
        df = pd.DataFrame({"consumption": 1.0}, index=idx)
        sp = TimeSeriesSplitter(n_splits=12, val_months=1, test_months=1)
        splits = sp.split(df)
        # Find split where val or test is Feb 2024
        feb_splits = [
            s
            for s in splits
            if (s.val_start.month == 2 and s.val_start.year == 2024)
            or (s.test_start.month == 2 and s.test_start.year == 2024)
        ]
        assert len(feb_splits) > 0
        for s in feb_splits:
            if s.val_start.month == 2 and s.val_start.year == 2024:
                assert s.val_end.day == 29
            if s.test_start.month == 2 and s.test_start.year == 2024:
                assert s.test_end.day == 29

    def test_months_vary_in_length(
        self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame
    ) -> None:
        """Different months have different day counts in val/test."""
        splits = splitter.split(sample_df)
        val_days = set()
        for s in splits:
            last_day = calendar.monthrange(s.val_end.year, s.val_end.month)[1]
            val_days.add(last_day)
        # Must have at least 2 different day counts (28/29/30/31)
        assert len(val_days) >= 2, f"All val months had same length: {val_days}"


class TestGapHours:
    """Tests for gap_hours parameter."""

    def test_gap_hours(self, sample_df: pd.DataFrame) -> None:
        """gap_hours=48 creates a gap between train/val."""
        sp = TimeSeriesSplitter(n_splits=3, val_months=1, test_months=1, gap_hours=48)
        splits = sp.split(sample_df)
        for s in splits:
            gap_train_val = (s.val_start - s.train_end).total_seconds() / 3600
            # With gap_hours=48 on train_end: train_end = val_start - 1h - 48h = val_start - 49h
            assert gap_train_val >= 49, (
                f"Split {s.split_idx}: gap_train_val={gap_train_val}h, expected >=49h"
            )


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_insufficient_data(self) -> None:
        """2 months data with n_splits=12 raises ValueError."""
        idx = pd.date_range("2024-01-01", "2024-02-29 23:00", freq="h")
        df = pd.DataFrame({"consumption": 1.0}, index=idx)
        sp = TimeSeriesSplitter(n_splits=12, val_months=1, test_months=1)
        with pytest.raises(ValueError, match="Insufficient data"):
            sp.split(df)

    def test_non_datetime_index(self) -> None:
        """Non-DatetimeIndex raises ValueError."""
        df = pd.DataFrame({"consumption": [1.0, 2.0, 3.0]})
        sp = TimeSeriesSplitter(n_splits=3, val_months=1, test_months=1)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            sp.split(df)


class TestFactory:
    """Test from_config classmethod."""

    def test_from_config(self) -> None:
        """Construct from CrossValidationConfig."""
        from energy_forecast.config import CrossValidationConfig

        cv_config = CrossValidationConfig(
            n_splits=5,
            val_months=2,
            test_months=1,
            gap_hours=24,
        )
        sp = TimeSeriesSplitter.from_config(cv_config)
        assert sp.n_splits == 5
        assert sp.val_months == 2
        assert sp.test_months == 1
        assert sp.gap_hours == 24

    def test_from_config_shuffle_true_raises(self) -> None:
        """shuffle=True in CrossValidationConfig raises ValueError."""
        from energy_forecast.config import CrossValidationConfig

        cv_config = CrossValidationConfig(n_splits=5, shuffle=True)
        with pytest.raises(ValueError, match="shuffle"):
            TimeSeriesSplitter.from_config(cv_config)


class TestIterSplits:
    """Test iter_splits method."""

    def test_iter_splits_dataframes(
        self, splitter: TimeSeriesSplitter, sample_df: pd.DataFrame
    ) -> None:
        """iter_splits yields correct DataFrame slices."""
        count = 0
        for info, train_df, val_df, test_df in splitter.iter_splits(sample_df):
            assert isinstance(info, SplitInfo)
            assert len(train_df) > 0
            assert len(val_df) > 0
            assert len(test_df) > 0

            # Check boundaries
            assert train_df.index.min() >= info.train_start
            assert train_df.index.max() <= info.train_end
            assert val_df.index.min() >= info.val_start
            assert val_df.index.max() <= info.val_end
            assert test_df.index.min() >= info.test_start
            assert test_df.index.max() <= info.test_end
            count += 1
        assert count == 12


class TestCustomSplits:
    """Test non-default parameters."""

    def test_n_splits_3(self, sample_df: pd.DataFrame) -> None:
        """n_splits=3 produces 3 splits."""
        sp = TimeSeriesSplitter(n_splits=3, val_months=1, test_months=1)
        splits = sp.split(sample_df)
        assert len(splits) == 3

    def test_no_shuffle_attribute(self) -> None:
        """Splitter has no shuffle mechanism."""
        sp = TimeSeriesSplitter()
        assert not hasattr(sp, "shuffle")
