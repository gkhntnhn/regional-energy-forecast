"""Unit tests for WeatherFeatureEngineer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config.settings import WeatherFeaturesConfig
from energy_forecast.features.weather import WeatherFeatureEngineer, map_wmo_group


@pytest.fixture()
def weather_config() -> dict[str, Any]:
    """Default weather feature config dict."""
    return WeatherFeaturesConfig().model_dump()


@pytest.fixture()
def engineer(weather_config: dict[str, Any]) -> WeatherFeatureEngineer:
    """WeatherFeatureEngineer with default config."""
    return WeatherFeatureEngineer(weather_config)


@pytest.fixture()
def weather_df() -> pd.DataFrame:
    """168-row DataFrame with weather columns and DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=168, freq="h")
    data: dict[str, Any] = {
        "temperature_2m": rng.uniform(-5, 35, 168),
        "weather_code": rng.choice([0, 1, 2, 3, 45, 61, 71, 95], 168),
        "wind_speed_10m": rng.uniform(0, 60, 168),
        "precipitation": rng.exponential(1.0, 168),
    }
    return pd.DataFrame(data, index=idx).rename_axis("datetime")


class TestWeatherFeatureEngineer:
    """Tests for WeatherFeatureEngineer."""

    def test_hdd_calculation(self, engineer: WeatherFeatureEngineer) -> None:
        """HDD = max(18 - T, 0). With T=10, HDD should be 8."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [10.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_hdd" in result.columns
        assert result["wth_hdd"].iloc[0] == pytest.approx(8.0)

    def test_cdd_calculation(self, engineer: WeatherFeatureEngineer) -> None:
        """CDD = max(T - 24, 0). With T=30, CDD should be 6."""
        idx = pd.date_range("2024-07-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [30.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_cdd" in result.columns
        assert result["wth_cdd"].iloc[0] == pytest.approx(6.0)

    def test_extreme_cold_flag(self, engineer: WeatherFeatureEngineer) -> None:
        """Temperature below extreme_cold (0) sets wth_extreme_cold=1."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [-5.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_extreme_cold" in result.columns
        assert result["wth_extreme_cold"].iloc[0] == 1

    def test_extreme_hot_flag(self, engineer: WeatherFeatureEngineer) -> None:
        """Temperature above extreme_hot (35) sets wth_extreme_hot=1."""
        idx = pd.date_range("2024-07-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [40.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_extreme_hot" in result.columns
        assert result["wth_extreme_hot"].iloc[0] == 1

    def test_rolling_created(
        self,
        engineer: WeatherFeatureEngineer,
        weather_df: pd.DataFrame,
    ) -> None:
        """Rolling window features are created for temperature."""
        result = engineer.fit_transform(weather_df)
        expected_col = "temperature_2m_window_6_mean"
        assert expected_col in result.columns

    def test_severity_mapping(self, engineer: WeatherFeatureEngineer) -> None:
        """WMO code 95 (thunderstorm) maps to severity=3."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "weather_code": [95.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_severity" in result.columns
        assert result["wth_severity"].iloc[0] == 3

    def test_severity_code_clear(self, engineer: WeatherFeatureEngineer) -> None:
        """WMO code 0 (clear) maps to severity=0."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_severity" in result.columns
        assert result["wth_severity"].iloc[0] == 0

    def test_temp_change(self, engineer: WeatherFeatureEngineer) -> None:
        """Temperature change over 3h is computed correctly."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        temps = np.arange(168, dtype=float)
        df = pd.DataFrame(
            {
                "temperature_2m": temps,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_temp_change_3h" in result.columns
        # At index 3, temp_change_3h = temps[3] - temps[0] = 3
        assert result["wth_temp_change_3h"].iloc[3] == pytest.approx(3.0)
        # First 3 rows should be NaN (no 3h lookback)
        assert pd.isna(result["wth_temp_change_3h"].iloc[0])

    def test_heavy_precip_flag(self, engineer: WeatherFeatureEngineer) -> None:
        """Precipitation > precip_threshold (10) sets wth_heavy_precip=1."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [15.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_heavy_precip" in result.columns
        assert result["wth_heavy_precip"].iloc[0] == 1

    def test_extreme_wind_flag(self, engineer: WeatherFeatureEngineer) -> None:
        """Wind speed > high_wind (25) sets wth_extreme_wind=1."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [55.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "wth_extreme_wind" in result.columns
        assert result["wth_extreme_wind"].iloc[0] == 1


# ---------------------------------------------------------------------------
# P0 feature: cdd_x_is_peak
# ---------------------------------------------------------------------------


class TestCddPeakInteraction:
    """CDD × is_peak cross-feature interaction (calendar × weather)."""

    def test_cdd_x_is_peak_hot_peak(self, engineer: WeatherFeatureEngineer) -> None:
        """Hot temperature during peak hours produces positive cdd_x_is_peak."""
        idx = pd.date_range("2024-07-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [30.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
                "is_peak": [0] * 168,  # will set specific hours
            },
            index=idx,
        )
        # Set peak hours (17-21) to 1
        df.loc[df.index.hour.isin([17, 18, 19, 20, 21]), "is_peak"] = 1

        result = engineer.fit_transform(df)

        assert "cdd_x_is_peak" in result.columns
        # CDD = 30 - 24 = 6.0
        peak_rows = result[result["is_peak"] == 1]
        np.testing.assert_allclose(peak_rows["cdd_x_is_peak"].values, 6.0)

    def test_cdd_x_is_peak_hot_off_peak(self, engineer: WeatherFeatureEngineer) -> None:
        """Hot temperature outside peak hours gives cdd_x_is_peak=0."""
        idx = pd.date_range("2024-07-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [30.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
                "is_peak": [0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)

        assert "cdd_x_is_peak" in result.columns
        assert (result["cdd_x_is_peak"] == 0.0).all()

    def test_cdd_x_is_peak_cold_peak(self, engineer: WeatherFeatureEngineer) -> None:
        """Cold temperature during peak hours gives cdd_x_is_peak=0 (CDD=0)."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [5.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
                "is_peak": [1] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)

        assert "cdd_x_is_peak" in result.columns
        assert (result["cdd_x_is_peak"] == 0.0).all()

    def test_cdd_x_is_peak_missing_is_peak(
        self, engineer: WeatherFeatureEngineer
    ) -> None:
        """Without is_peak column, cdd_x_is_peak is not created (graceful skip)."""
        idx = pd.date_range("2024-07-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [30.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "cdd_x_is_peak" not in result.columns


# ---------------------------------------------------------------------------
# Tests: WMO weather_group categorical feature
# ---------------------------------------------------------------------------


class TestMapWmoGroup:
    """Tests for map_wmo_group() function."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0, "clear"),
            (1, "cloudy"),
            (2, "cloudy"),
            (3, "cloudy"),
            (45, "fog"),
            (48, "fog"),
            (51, "drizzle"),
            (55, "drizzle"),
            (61, "rain"),
            (63, "rain"),
            (65, "rain"),
            (71, "snow"),
            (75, "snow"),
            (77, "snow"),
            (80, "showers"),
            (82, "showers"),
            (95, "thunderstorm"),
            (96, "thunderstorm"),
            (99, "thunderstorm"),
        ],
    )
    def test_known_codes(self, code: int, expected: str) -> None:
        """Known WMO codes map to the correct group."""
        assert map_wmo_group(float(code)) == expected

    def test_unknown_code(self) -> None:
        """Unknown WMO code maps to 'unknown'."""
        assert map_wmo_group(42.0) == "unknown"
        assert map_wmo_group(100.0) == "unknown"

    def test_nan_code(self) -> None:
        """NaN code maps to 'unknown'."""
        assert map_wmo_group(float("nan")) == "unknown"


class TestWeatherGroup:
    """Tests for weather_group feature in WeatherFeatureEngineer."""

    def test_weather_group_created(self, engineer: WeatherFeatureEngineer) -> None:
        """weather_group column is created from weather_code."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "weather_code": [61.0] * 168,  # rain
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "weather_group" in result.columns
        assert result["weather_group"].iloc[0] == "rain"

    def test_weather_group_is_string(self, engineer: WeatherFeatureEngineer) -> None:
        """weather_group column has string (object) dtype."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "weather_code": [0.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert result["weather_group"].dtype == object

    def test_weather_group_mixed_codes(self, engineer: WeatherFeatureEngineer) -> None:
        """Different weather_code values produce correct group labels."""
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 4,
                "weather_code": [0.0, 61.0, 71.0, 95.0],
                "wind_speed_10m": [5.0] * 4,
                "precipitation": [0.0] * 4,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert result["weather_group"].tolist() == ["clear", "rain", "snow", "thunderstorm"]

    def test_weather_group_skipped_without_weather_code(
        self, engineer: WeatherFeatureEngineer
    ) -> None:
        """Without weather_code column, weather_group is not created."""
        idx = pd.date_range("2024-01-01", periods=168, freq="h")
        df = pd.DataFrame(
            {
                "temperature_2m": [15.0] * 168,
                "wind_speed_10m": [5.0] * 168,
                "precipitation": [0.0] * 168,
            },
            index=idx,
        )
        result = engineer.fit_transform(df)
        assert "weather_group" not in result.columns
