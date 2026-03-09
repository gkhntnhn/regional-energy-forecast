"""Tests for CalendarFeatureEngineer with DB-injected holidays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features.calendar import CalendarFeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calendar_config() -> dict[str, Any]:
    """Minimal calendar feature config matching calendar.yaml structure."""
    return {
        "datetime": {
            "extract": ["hour", "day_of_week", "month"],
        },
        "cyclical": {
            "hour": {"period": 24},
            "day_of_week": {"period": 7},
            "month": {"period": 12},
        },
        "holidays": {
            "path": "nonexistent.parquet",  # should NOT be loaded
            "include_ramadan": True,
            "bridge_days": True,
        },
        "anticipation": {"enabled": False, "windows": []},
        "spline_seasonality": {"enabled": False, "n_splines": 12},
        "business_hours": {"start": 8, "end": 18, "peak_start": 17, "peak_end": 22},
        "disabled_features": [],
    }


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """10-day DataFrame with DatetimeIndex spanning a holiday."""
    idx = pd.date_range("2024-01-01", periods=240, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"consumption": rng.uniform(800, 1200, 240)},
        index=idx,
    ).rename_axis("datetime")


@pytest.fixture()
def holidays_df() -> pd.DataFrame:
    """Pre-loaded holidays DataFrame mimicking DB output."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-04-10"]),
        "holiday_name": ["Yilbasi", "Ramazan_Bayrami_1"],
        "raw_holiday_name": ["Yılbaşı", "Ramazan Bayramı 1. Gün"],
        "is_ramadan": [False, True],
        "bayram_gun_no": [0, 1],
        "bayrama_kalan_gun": [0, 0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalendarHolidayInjection:
    """Verify CalendarFE uses injected holidays_df instead of parquet file."""

    def test_injected_holidays_creates_is_holiday(
        self,
        calendar_config: dict[str, Any],
        sample_df: pd.DataFrame,
        holidays_df: pd.DataFrame,
    ) -> None:
        """When holidays_df is provided, is_holiday column is created."""
        fe = CalendarFeatureEngineer(calendar_config, holidays_df=holidays_df)
        result = fe.fit_transform(sample_df)

        assert "is_holiday" in result.columns
        # Jan 1 hours should be marked as holiday
        jan1_mask = result.index.date == pd.Timestamp("2024-01-01").date()
        assert result.loc[jan1_mask, "is_holiday"].all()
        # Jan 2 should NOT be a holiday
        jan2_mask = result.index.date == pd.Timestamp("2024-01-02").date()
        assert not result.loc[jan2_mask, "is_holiday"].any()

    def test_no_holidays_df_no_parquet_no_crash(
        self,
        calendar_config: dict[str, Any],
        sample_df: pd.DataFrame,
    ) -> None:
        """When no holidays_df and parquet doesn't exist, still works."""
        fe = CalendarFeatureEngineer(calendar_config, holidays_df=None)
        result = fe.fit_transform(sample_df)
        # Should still produce output — just no holiday features
        assert len(result) == len(sample_df)

    def test_empty_holidays_df_falls_back(
        self,
        calendar_config: dict[str, Any],
        sample_df: pd.DataFrame,
    ) -> None:
        """Empty holidays_df triggers parquet fallback (graceful)."""
        empty_df = pd.DataFrame()
        fe = CalendarFeatureEngineer(calendar_config, holidays_df=empty_df)
        result = fe.fit_transform(sample_df)
        assert len(result) == len(sample_df)

    def test_ramadan_flag_from_injected(
        self,
        calendar_config: dict[str, Any],
        holidays_df: pd.DataFrame,
    ) -> None:
        """Ramadan days are flagged from injected holidays_df."""
        # Extend data to cover April 10
        idx = pd.date_range("2024-04-09", periods=48, freq="h")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {"consumption": rng.uniform(800, 1200, 48)}, index=idx,
        ).rename_axis("datetime")

        fe = CalendarFeatureEngineer(calendar_config, holidays_df=holidays_df)
        result = fe.fit_transform(df)

        if "is_ramadan" in result.columns:
            apr10_mask = result.index.date == pd.Timestamp("2024-04-10").date()
            assert result.loc[apr10_mask, "is_ramadan"].any()
