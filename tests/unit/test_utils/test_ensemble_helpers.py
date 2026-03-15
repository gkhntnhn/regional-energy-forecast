"""Tests for ensemble_helpers utility functions."""

from __future__ import annotations

import pandas as pd

from energy_forecast.utils.ensemble_helpers import build_context_features


class TestBuildContextFeatures:
    """Tests for build_context_features."""

    def test_adds_hour_column(self) -> None:
        """build_context_features adds hour from DatetimeIndex."""
        idx = pd.date_range("2026-01-01", periods=24, freq="h")
        target = pd.DataFrame({"pred": range(24)}, index=idx)
        source = pd.DataFrame(index=idx)

        build_context_features(target, source, ["hour"])
        assert "hour" in target.columns
        assert target["hour"].iloc[0] == 0
        assert target["hour"].iloc[23] == 23

    def test_adds_day_of_week(self) -> None:
        """build_context_features adds day_of_week."""
        idx = pd.date_range("2026-01-05", periods=1, freq="h")  # Monday
        target = pd.DataFrame({"pred": [1]}, index=idx)
        source = pd.DataFrame(index=idx)

        build_context_features(target, source, ["day_of_week"])
        assert target["day_of_week"].iloc[0] == 0  # Monday = 0

    def test_adds_is_weekend(self) -> None:
        """build_context_features adds is_weekend flag."""
        # Saturday
        idx = pd.date_range("2026-01-03", periods=1, freq="h")
        target = pd.DataFrame({"pred": [1]}, index=idx)
        source = pd.DataFrame(index=idx)

        build_context_features(target, source, ["is_weekend"])
        assert target["is_weekend"].iloc[0] == 1

    def test_adds_month(self) -> None:
        """build_context_features adds month column."""
        idx = pd.date_range("2026-03-15", periods=1, freq="h")
        target = pd.DataFrame({"pred": [1]}, index=idx)
        source = pd.DataFrame(index=idx)

        build_context_features(target, source, ["month"])
        assert target["month"].iloc[0] == 3

    def test_is_holiday_from_source(self) -> None:
        """build_context_features copies is_holiday from source_df."""
        idx = pd.date_range("2026-01-01", periods=2, freq="h")
        target = pd.DataFrame({"pred": [1, 2]}, index=idx)
        source = pd.DataFrame({"is_holiday": [1, 0]}, index=idx)

        build_context_features(target, source, ["is_holiday"])
        assert target["is_holiday"].iloc[0] == 1
        assert target["is_holiday"].iloc[1] == 0

    def test_is_holiday_defaults_to_zero(self) -> None:
        """When source has no is_holiday column, defaults to 0."""
        idx = pd.date_range("2026-01-01", periods=2, freq="h")
        target = pd.DataFrame({"pred": [1, 2]}, index=idx)
        source = pd.DataFrame(index=idx)

        build_context_features(target, source, ["is_holiday"])
        assert (target["is_holiday"] == 0).all()

    def test_is_holiday_with_max_len(self) -> None:
        """max_len truncates is_holiday values."""
        idx = pd.date_range("2026-01-01", periods=3, freq="h")
        target = pd.DataFrame({"pred": [1, 2, 3]}, index=idx)
        source = pd.DataFrame({"is_holiday": [1, 0, 1]}, index=idx)

        # Only take first 2 values
        target_short = target.iloc[:2].copy()
        build_context_features(target_short, source, ["is_holiday"], max_len=2)
        assert len(target_short["is_holiday"]) == 2

    def test_cast_categorical_to_str(self) -> None:
        """cast_categorical_to_str converts hour/day_of_week/month to str."""
        idx = pd.date_range("2026-01-01 10:00", periods=1, freq="h")
        target = pd.DataFrame({"pred": [1]}, index=idx)
        source = pd.DataFrame(index=idx)

        build_context_features(
            target,
            source,
            ["hour", "day_of_week", "month"],
            cast_categorical_to_str=True,
        )
        assert target["hour"].dtype == object  # str
        assert target["hour"].iloc[0] == "10"
        assert target["day_of_week"].dtype == object
        assert target["month"].dtype == object
        assert target["month"].iloc[0] == "1"

    def test_cast_categorical_only_affects_existing(self) -> None:
        """cast_categorical_to_str only converts columns that exist."""
        idx = pd.date_range("2026-01-01", periods=1, freq="h")
        target = pd.DataFrame({"pred": [1]}, index=idx)
        source = pd.DataFrame(index=idx)

        # Only add hour, not month or day_of_week
        build_context_features(
            target,
            source,
            ["hour"],
            cast_categorical_to_str=True,
        )
        assert target["hour"].dtype == object
        assert "month" not in target.columns

    def test_multiple_features_at_once(self) -> None:
        """All context features can be added in a single call."""
        idx = pd.date_range("2026-01-01", periods=24, freq="h")
        target = pd.DataFrame({"pred": range(24)}, index=idx)
        source = pd.DataFrame({"is_holiday": [1] + [0] * 23}, index=idx)

        build_context_features(
            target, source, ["hour", "day_of_week", "is_weekend", "month", "is_holiday"]
        )
        assert set(["hour", "day_of_week", "is_weekend", "month", "is_holiday"]).issubset(
            target.columns
        )
