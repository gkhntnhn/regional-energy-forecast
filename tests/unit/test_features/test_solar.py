"""Unit tests for SolarFeatureEngineer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from energy_forecast.config.settings import SolarConfig
from energy_forecast.features.solar import SolarFeatureEngineer


@pytest.fixture()
def solar_config() -> dict[str, Any]:
    """Default solar feature config dict."""
    return SolarConfig().model_dump()


@pytest.fixture()
def engineer(solar_config: dict[str, Any]) -> SolarFeatureEngineer:
    """SolarFeatureEngineer with default config."""
    return SolarFeatureEngineer(solar_config)


@pytest.fixture()
def solar_df() -> pd.DataFrame:
    """72-row (3 days) DataFrame with DatetimeIndex in summer."""
    idx = pd.date_range("2024-06-15", periods=72, freq="h")
    return pd.DataFrame(
        {"placeholder": range(72)},
        index=idx,
    ).rename_axis("datetime")


class TestSolarFeatureEngineer:
    """Tests for SolarFeatureEngineer."""

    def test_elevation_midday_positive(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Noon solar elevation should be positive in summer."""
        result = engineer.fit_transform(solar_df)
        assert "sol_elevation" in result.columns
        # Hour 12 on first day = index 12
        noon_elevation = result["sol_elevation"].iloc[12]
        assert noon_elevation > 0

    def test_elevation_midnight_zero(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Midnight solar elevation should be zero or negative."""
        result = engineer.fit_transform(solar_df)
        # Hour 0 on first day = index 0
        midnight_elevation = result["sol_elevation"].iloc[0]
        assert midnight_elevation <= 0

    def test_ghi_daytime_positive(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Clear-sky GHI at midday should be positive."""
        result = engineer.fit_transform(solar_df)
        assert "sol_ghi" in result.columns
        noon_ghi = result["sol_ghi"].iloc[12]
        assert noon_ghi > 0

    def test_ghi_nighttime_zero(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Clear-sky GHI at midnight should be approximately zero."""
        result = engineer.fit_transform(solar_df)
        midnight_ghi = result["sol_ghi"].iloc[0]
        assert midnight_ghi == pytest.approx(0.0, abs=1e-3)

    def test_clearness_range(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Clearness index should be between 0 and 1."""
        result = engineer.fit_transform(solar_df)
        assert "sol_clearness_index" in result.columns
        kt = result["sol_clearness_index"]
        assert (kt >= 0.0).all()
        assert (kt <= 1.0).all()

    def test_is_daylight_flag(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Daylight flag is 1 during day and 0 at night."""
        result = engineer.fit_transform(solar_df)
        assert "sol_is_daylight" in result.columns
        # Noon (hour 12) in summer should be daylight
        assert result["sol_is_daylight"].iloc[12] == 1
        # Midnight (hour 0) should not be daylight
        assert result["sol_is_daylight"].iloc[0] == 0

    def test_poa_calculated(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """POA global irradiance exists and is positive at noon."""
        result = engineer.fit_transform(solar_df)
        assert "sol_poa_global" in result.columns
        noon_poa = result["sol_poa_global"].iloc[12]
        assert noon_poa > 0

    def test_lead_features_created(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """Lead features sol_ghi_lead_1, _2, _3 are created when enabled."""
        result = engineer.fit_transform(solar_df)
        for h in [1, 2, 3]:
            col = f"sol_ghi_lead_{h}"
            assert col in result.columns, f"Missing column: {col}"

    def test_lead_disabled(self, solar_df: pd.DataFrame) -> None:
        """No lead columns when lead.enabled=False."""
        config = SolarConfig().model_dump()
        config["lead"]["enabled"] = False
        eng = SolarFeatureEngineer(config)
        result = eng.fit_transform(solar_df)
        lead_cols = [c for c in result.columns if "lead" in c]
        assert len(lead_cols) == 0

    def test_all_solar_columns(
        self,
        engineer: SolarFeatureEngineer,
        solar_df: pd.DataFrame,
    ) -> None:
        """All expected sol_ columns are present in output."""
        result = engineer.fit_transform(solar_df)
        expected_cols = [
            "sol_elevation",
            "sol_azimuth",
            "sol_ghi",
            "sol_dni",
            "sol_dhi",
            "sol_poa_global",
            "sol_clearness_index",
            "sol_cloud_proxy",
            "sol_is_daylight",
            "sol_daylight_hours",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"
