"""Unit tests for CalendarFeatureEngineer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config.settings import CalendarConfig
from energy_forecast.features.calendar import CalendarFeatureEngineer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calendar_df() -> pd.DataFrame:
    """7 days x 24 hours = 168 rows starting 2024-01-01 (Monday)."""
    idx = pd.date_range("2024-01-01", periods=168, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"consumption": 800.0 + rng.random(168) * 400},
        index=idx,
    ).rename_axis("datetime")


@pytest.fixture()
def default_config() -> dict[str, Any]:
    """Default CalendarConfig as a plain dict."""
    return CalendarConfig().model_dump()


@pytest.fixture()
def engineer(default_config: dict[str, Any]) -> CalendarFeatureEngineer:
    """CalendarFeatureEngineer with default config."""
    return CalendarFeatureEngineer(config=default_config)


@pytest.fixture()
def transformed_df(
    engineer: CalendarFeatureEngineer,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calendar features applied to the 7-day sample."""
    return engineer.transform(calendar_df)


@pytest.fixture()
def holidays_parquet(tmp_path: Path) -> Path:
    """Small holidays parquet with varied holiday types."""
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2024-01-01",  # New Year (resmi) — Monday
                "2024-04-10",  # Ramazan Bayrami (dini) — Wednesday
                "2024-04-11",  # Ramazan Bayrami 2. Gun (dini) — Thursday
                "2024-04-12",  # Ramazan Bayrami 3. Gun (dini) — Friday
                "2024-04-23",  # 23 Nisan (resmi) — Tuesday
            ]),
            "holiday_name": [
                "New Year's Day",
                "Ramazan Bayrami 1. Gun",
                "Ramazan Bayrami 2. Gun",
                "Ramazan Bayrami 3. Gun",
                "National Sovereignty and Children's Day",
            ],
            "is_ramadan": [0, 0, 0, 0, 0],
            "bayram_gun_no": [0, 1, 2, 3, 0],
        }
    )
    p = tmp_path / "turkish_holidays.parquet"
    holidays.to_parquet(p, index=False)
    return p


@pytest.fixture()
def holiday_config(holidays_parquet: Path) -> dict[str, Any]:
    """CalendarConfig dict pointing to the tmp holidays file."""
    cfg = CalendarConfig().model_dump()
    cfg["holidays"]["path"] = str(holidays_parquet)
    return cfg


@pytest.fixture()
def holiday_engineer(holiday_config: dict[str, Any]) -> CalendarFeatureEngineer:
    """Engineer with holidays parquet available."""
    return CalendarFeatureEngineer(config=holiday_config)


# ---------------------------------------------------------------------------
# 1. Datetime extraction
# ---------------------------------------------------------------------------


class TestDatetimeExtraction:
    """Datetime component columns are correctly extracted."""

    def test_datetime_features_extracted(self, transformed_df: pd.DataFrame) -> None:
        """hour, day_of_week, etc. are added as columns."""
        expected = [
            "hour",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "year",
        ]
        for col in expected:
            assert col in transformed_df.columns, f"Missing column: {col}"

    def test_hour_values_range(self, transformed_df: pd.DataFrame) -> None:
        """Extracted hour values span 0-23."""
        assert transformed_df["hour"].min() == 0
        assert transformed_df["hour"].max() == 23

    def test_config_driven_extraction(self, calendar_df: pd.DataFrame) -> None:
        """Only configured features are extracted."""
        cfg = CalendarConfig().model_dump()
        cfg["datetime"]["extract"] = ["hour", "month"]
        # Remove cyclical keys that won't have source columns
        cfg["cyclical"] = {
            "hour": {"period": 24},
            "month": {"period": 12},
        }
        eng = CalendarFeatureEngineer(config=cfg)
        result = eng.transform(calendar_df)

        assert "hour" in result.columns
        assert "month" in result.columns
        # Features not in the extract list should be absent
        assert "day_of_week" not in result.columns
        assert "quarter" not in result.columns


# ---------------------------------------------------------------------------
# 2. Cyclical encoding
# ---------------------------------------------------------------------------


class TestCyclicalEncoding:
    """Sin/cos cyclical features have correct ranges and values."""

    def test_cyclical_sin_cos_range(self, transformed_df: pd.DataFrame) -> None:
        """All sin/cos columns have values in [-1, 1]."""
        sin_cos_cols = [
            c for c in transformed_df.columns if c.endswith("_sin") or c.endswith("_cos")
        ]
        assert len(sin_cos_cols) > 0, "No cyclical columns found"
        for col in sin_cos_cols:
            assert transformed_df[col].min() >= -1.0 - 1e-9, f"{col} below -1"
            assert transformed_df[col].max() <= 1.0 + 1e-9, f"{col} above 1"

    def test_cyclical_hour_midnight(self, transformed_df: pd.DataFrame) -> None:
        """hour=0 produces hour_sin approximately 0."""
        midnight_rows = transformed_df[transformed_df["hour"] == 0]
        assert len(midnight_rows) > 0
        # sin(2*pi*0/24) = 0
        assert all(np.abs(midnight_rows["hour_sin"]) < 1e-6)

    def test_cyclical_hour_noon(self, transformed_df: pd.DataFrame) -> None:
        """hour=12 produces hour_cos approximately -1."""
        noon_rows = transformed_df[transformed_df["hour"] == 12]
        assert len(noon_rows) > 0
        # cos(2*pi*12/24) = cos(pi) = -1
        assert all(np.abs(noon_rows["hour_cos"] - (-1.0)) < 1e-6)


# ---------------------------------------------------------------------------
# 3. Holiday features
# ---------------------------------------------------------------------------


class TestHolidayFeatures:
    """Holiday, Ramadan, and proximity flags are set correctly."""

    def test_holiday_flag_known_date(
        self,
        holiday_engineer: CalendarFeatureEngineer,
        calendar_df: pd.DataFrame,
    ) -> None:
        """2024-01-01 (known holiday in parquet) has is_holiday=1."""
        result = holiday_engineer.transform(calendar_df)
        jan1_rows = result.loc["2024-01-01"]
        assert (jan1_rows["is_holiday"] == 1).all()

    def test_holiday_flag_normal_day(
        self,
        holiday_engineer: CalendarFeatureEngineer,
        calendar_df: pd.DataFrame,
    ) -> None:
        """2024-01-02 (not a holiday) has is_holiday=0."""
        result = holiday_engineer.transform(calendar_df)
        jan2_rows = result.loc["2024-01-02"]
        assert (jan2_rows["is_holiday"] == 0).all()

    def test_holiday_file_missing_graceful(self, calendar_df: pd.DataFrame) -> None:
        """Missing holidays file sets all is_holiday=0 without error."""
        cfg = CalendarConfig().model_dump()
        cfg["holidays"]["path"] = "/nonexistent/path/holidays.parquet"
        eng = CalendarFeatureEngineer(config=cfg)
        result = eng.transform(calendar_df)

        assert "is_holiday" in result.columns
        assert (result["is_holiday"] == 0).all()

    def test_days_until_holiday(
        self,
        holiday_engineer: CalendarFeatureEngineer,
        calendar_df: pd.DataFrame,
    ) -> None:
        """On the holiday itself, days_until_holiday is 0."""
        result = holiday_engineer.transform(calendar_df)
        jan1_rows = result.loc["2024-01-01"]
        assert (jan1_rows["days_until_holiday"] == 0).all()


# ---------------------------------------------------------------------------
# 4. Weekend detection
# ---------------------------------------------------------------------------


class TestWeekendDetection:
    """Weekend flag is correctly set based on day of week."""

    def test_is_weekend_saturday(self, transformed_df: pd.DataFrame) -> None:
        """Saturday (2024-01-06) has is_weekend=1."""
        sat_rows = transformed_df.loc["2024-01-06"]
        assert (sat_rows["is_weekend"] == 1).all()

    def test_is_weekend_monday(self, transformed_df: pd.DataFrame) -> None:
        """Monday (2024-01-01) has is_weekend=0."""
        mon_rows = transformed_df.loc["2024-01-01"]
        assert (mon_rows["is_weekend"] == 0).all()


# ---------------------------------------------------------------------------
# 5. Business and peak hours
# ---------------------------------------------------------------------------


class TestBusinessHours:
    """Business-hour and peak-hour flags use configured boundaries."""

    def test_business_hours_within(self, transformed_df: pd.DataFrame) -> None:
        """hour=10 (within 8-18) has is_business_hours=1."""
        rows = transformed_df[transformed_df["hour"] == 10]
        assert (rows["is_business_hours"] == 1).all()

    def test_business_hours_outside(self, transformed_df: pd.DataFrame) -> None:
        """hour=22 (outside 8-18) has is_business_hours=0."""
        rows = transformed_df[transformed_df["hour"] == 22]
        assert (rows["is_business_hours"] == 0).all()

    def test_peak_hours(self, transformed_df: pd.DataFrame) -> None:
        """hour=18 (within peak 17-22) has is_peak=1."""
        rows = transformed_df[transformed_df["hour"] == 18]
        assert (rows["is_peak"] == 1).all()

    def test_non_peak_hours(self, transformed_df: pd.DataFrame) -> None:
        """hour=10 (outside peak 17-22) has is_peak=0."""
        rows = transformed_df[transformed_df["hour"] == 10]
        assert (rows["is_peak"] == 0).all()


# ---------------------------------------------------------------------------
# 6. Season flags
# ---------------------------------------------------------------------------


class TestSeasonFlags:
    """Heating and cooling season flags reflect month-based rules."""

    def test_season_winter(self, transformed_df: pd.DataFrame) -> None:
        """January is heating season (is_heating_season=1)."""
        assert (transformed_df["is_heating_season"] == 1).all()

    def test_season_summer(self) -> None:
        """July is cooling season (is_cooling_season=1)."""
        idx = pd.date_range("2024-07-01", periods=24, freq="h")
        df = pd.DataFrame({"consumption": 1000.0}, index=idx).rename_axis("datetime")
        cfg = CalendarConfig().model_dump()
        eng = CalendarFeatureEngineer(config=cfg)
        result = eng.transform(df)

        assert (result["is_cooling_season"] == 1).all()
        assert (result["is_heating_season"] == 0).all()

    def test_season_column_values(self, transformed_df: pd.DataFrame) -> None:
        """January maps to season=0 (winter)."""
        assert (transformed_df["season"] == 0).all()


# ---------------------------------------------------------------------------
# 7. Holiday interaction features (P0-2)
# ---------------------------------------------------------------------------


class TestHolidayInteractionFeatures:
    """Holiday × hour, holiday type, and bridge day features."""

    def test_is_holiday_x_hour_on_holiday(
        self,
        holiday_engineer: CalendarFeatureEngineer,
        calendar_df: pd.DataFrame,
    ) -> None:
        """is_holiday_x_hour = hour on holiday, 0 on normal days."""
        result = holiday_engineer.transform(calendar_df)
        # Jan 1 is holiday — is_holiday_x_hour should equal the hour
        jan1_rows = result.loc["2024-01-01"]
        expected = jan1_rows.index.hour
        np.testing.assert_array_equal(jan1_rows["is_holiday_x_hour"].values, expected)

    def test_is_holiday_x_hour_on_normal_day(
        self,
        holiday_engineer: CalendarFeatureEngineer,
        calendar_df: pd.DataFrame,
    ) -> None:
        """is_holiday_x_hour = 0 for non-holiday days."""
        result = holiday_engineer.transform(calendar_df)
        jan2_rows = result.loc["2024-01-02"]
        assert (jan2_rows["is_holiday_x_hour"] == 0).all()

    def test_tatil_tipi_dini(self, tmp_path: Path) -> None:
        """Ramazan Bayrami is classified as dini (tatil_tipi=2)."""
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-10"]),
                "holiday_name": ["Ramazan Bayrami 1. Gun"],
                "is_ramadan": [0],
                "bayram_gun_no": [1],
            }
        )
        p = tmp_path / "h.parquet"
        holidays.to_parquet(p, index=False)

        cfg = CalendarConfig().model_dump()
        cfg["holidays"]["path"] = str(p)
        eng = CalendarFeatureEngineer(config=cfg)

        idx = pd.date_range("2024-04-10", periods=24, freq="h")
        df = pd.DataFrame({"consumption": 1000.0}, index=idx).rename_axis("datetime")
        result = eng.transform(df)

        assert (result["tatil_tipi"] == 2).all()

    def test_tatil_tipi_resmi(self, tmp_path: Path) -> None:
        """New Year's Day is classified as resmi (tatil_tipi=1)."""
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "holiday_name": ["New Year's Day"],
                "is_ramadan": [0],
                "bayram_gun_no": [0],
            }
        )
        p = tmp_path / "h.parquet"
        holidays.to_parquet(p, index=False)

        cfg = CalendarConfig().model_dump()
        cfg["holidays"]["path"] = str(p)
        eng = CalendarFeatureEngineer(config=cfg)

        idx = pd.date_range("2024-01-01", periods=24, freq="h")
        df = pd.DataFrame({"consumption": 1000.0}, index=idx).rename_axis("datetime")
        result = eng.transform(df)

        assert (result["tatil_tipi"] == 1).all()

    def test_tatil_tipi_none_on_normal_day(
        self,
        holiday_engineer: CalendarFeatureEngineer,
        calendar_df: pd.DataFrame,
    ) -> None:
        """Normal days have tatil_tipi=0."""
        result = holiday_engineer.transform(calendar_df)
        jan2_rows = result.loc["2024-01-02"]
        assert (jan2_rows["tatil_tipi"] == 0).all()

    def test_bridge_day_friday_after_thursday_holiday(self, tmp_path: Path) -> None:
        """Friday is a bridge day when Thursday is a holiday (Thu+Fri+Sat+Sun)."""
        # Thursday 2024-04-04 is holiday, Friday should be bridge
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-04"]),
                "holiday_name": ["Test Holiday"],
                "is_ramadan": [0],
                "bayram_gun_no": [0],
            }
        )
        p = tmp_path / "h.parquet"
        holidays.to_parquet(p, index=False)

        cfg = CalendarConfig().model_dump()
        cfg["holidays"]["path"] = str(p)
        eng = CalendarFeatureEngineer(config=cfg)

        idx = pd.date_range("2024-04-01", periods=168, freq="h")
        df = pd.DataFrame({"consumption": 1000.0}, index=idx).rename_axis("datetime")
        result = eng.transform(df)

        # Friday 2024-04-05: prev=Thu(holiday), next=Sat(weekend)
        friday_rows = result.loc["2024-04-05"]
        assert (friday_rows["is_bridge_day"] == 1).all()

    def test_bridge_day_tuesday_after_monday_holiday(self, tmp_path: Path) -> None:
        """Monday is a bridge day when Tuesday is a holiday (Sat+Sun+Mon+Tue)."""
        # Tuesday 2024-04-23 is 23 Nisan holiday, Monday should be bridge
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-23"]),
                "holiday_name": ["National Sovereignty and Children's Day"],
                "is_ramadan": [0],
                "bayram_gun_no": [0],
            }
        )
        p = tmp_path / "h.parquet"
        holidays.to_parquet(p, index=False)

        cfg = CalendarConfig().model_dump()
        cfg["holidays"]["path"] = str(p)
        eng = CalendarFeatureEngineer(config=cfg)

        idx = pd.date_range("2024-04-19", periods=168, freq="h")
        df = pd.DataFrame({"consumption": 1000.0}, index=idx).rename_axis("datetime")
        result = eng.transform(df)

        # Monday 2024-04-22: prev=Sun(weekend), next=Tue(holiday) → classic bridge
        monday_rows = result.loc["2024-04-22"]
        assert (monday_rows["is_bridge_day"] == 1).all()

    def test_no_bridge_on_normal_friday(
        self,
        holiday_engineer: CalendarFeatureEngineer,
    ) -> None:
        """Regular Fridays without nearby holidays are NOT bridge days."""
        # Use a period with no holidays nearby
        idx = pd.date_range("2024-02-05", periods=168, freq="h")
        df = pd.DataFrame({"consumption": 1000.0}, index=idx).rename_axis("datetime")
        result = holiday_engineer.transform(df)

        # Friday 2024-02-09: no nearby holiday → NOT bridge
        friday_rows = result.loc["2024-02-09"]
        assert (friday_rows["is_bridge_day"] == 0).all()

    def test_missing_holidays_sets_defaults(self, calendar_df: pd.DataFrame) -> None:
        """Missing holidays file sets tatil_tipi=0 and is_holiday_x_hour=0."""
        cfg = CalendarConfig().model_dump()
        cfg["holidays"]["path"] = "/nonexistent/path/holidays.parquet"
        eng = CalendarFeatureEngineer(config=cfg)
        result = eng.transform(calendar_df)

        assert "tatil_tipi" in result.columns
        assert "is_holiday_x_hour" in result.columns
        assert (result["tatil_tipi"] == 0).all()
        assert (result["is_holiday_x_hour"] == 0).all()
