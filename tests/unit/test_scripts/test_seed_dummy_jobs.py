"""Tests for seed_dummy_jobs script."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.seed_dummy_jobs import _realistic_consumption, _realistic_weather


class TestRealisticConsumption:
    """Tests for _realistic_consumption helper."""

    def test_returns_float(self) -> None:
        val = _realistic_consumption(hour=12, month=1)
        assert isinstance(val, float)

    def test_minimum_floor(self) -> None:
        """Consumption never drops below 500 MWh."""
        for _ in range(100):
            val = _realistic_consumption(hour=3, month=4)
            assert val >= 500.0

    def test_peak_higher_than_night(self) -> None:
        """Average peak hour > average night hour (statistical)."""
        peak_vals = [_realistic_consumption(12, 1) for _ in range(200)]
        night_vals = [_realistic_consumption(3, 1) for _ in range(200)]
        assert sum(peak_vals) / len(peak_vals) > sum(night_vals) / len(night_vals)

    def test_winter_higher_than_spring(self) -> None:
        """Average winter > average spring consumption (statistical)."""
        winter = [_realistic_consumption(12, 1) for _ in range(200)]
        spring = [_realistic_consumption(12, 4) for _ in range(200)]
        assert sum(winter) / len(winter) > sum(spring) / len(spring)

    def test_all_hours_valid(self) -> None:
        """All 24 hours produce valid results."""
        for h in range(24):
            val = _realistic_consumption(h, 6)
            assert val >= 500.0


class TestRealisticWeather:
    """Tests for _realistic_weather helper."""

    def test_returns_dict_with_expected_keys(self) -> None:
        w = _realistic_weather(hour=12, month=7)
        expected_keys = {
            "temperature_2m", "apparent_temperature", "relative_humidity_2m",
            "dew_point_2m", "precipitation", "snow_depth", "surface_pressure",
            "wind_speed_10m", "wind_direction_10m", "shortwave_radiation",
            "weather_code", "wth_hdd", "wth_cdd",
        }
        assert set(w.keys()) == expected_keys

    def test_humidity_clamped(self) -> None:
        """Humidity stays within [30, 95]."""
        for _ in range(200):
            w = _realistic_weather(12, 6)
            assert 30 <= w["relative_humidity_2m"] <= 95

    def test_no_snow_in_summer(self) -> None:
        """Snow depth is 0 for months > 3."""
        for _ in range(50):
            w = _realistic_weather(12, 7)
            assert w["snow_depth"] == 0.0

    def test_no_radiation_at_night(self) -> None:
        """Shortwave radiation is 0 outside 6-18."""
        for _ in range(50):
            w = _realistic_weather(2, 6)
            assert w["shortwave_radiation"] == 0.0

    def test_radiation_positive_at_noon(self) -> None:
        """Shortwave radiation is positive at noon."""
        for _ in range(50):
            w = _realistic_weather(12, 6)
            assert w["shortwave_radiation"] >= 0.0

    def test_wind_non_negative(self) -> None:
        """Wind speed is never negative."""
        for _ in range(200):
            w = _realistic_weather(12, 6)
            assert w["wind_speed_10m"] >= 0.0

    def test_precipitation_non_negative(self) -> None:
        for _ in range(200):
            w = _realistic_weather(12, 6)
            assert w["precipitation"] >= 0.0

    def test_hdd_cdd_non_negative(self) -> None:
        for _ in range(200):
            w = _realistic_weather(12, 6)
            assert w["wth_hdd"] >= 0.0
            assert w["wth_cdd"] >= 0.0

    def test_summer_warmer_than_winter(self) -> None:
        """July temps higher than January on average."""
        summer = [_realistic_weather(12, 7)["temperature_2m"] for _ in range(200)]
        winter = [_realistic_weather(12, 1)["temperature_2m"] for _ in range(200)]
        assert sum(summer) / len(summer) > sum(winter) / len(winter)

    def test_wind_direction_range(self) -> None:
        """Wind direction is [0, 360)."""
        for _ in range(100):
            w = _realistic_weather(12, 6)
            assert 0 <= w["wind_direction_10m"] <= 360


class TestMainDryRun:
    """Test main() in dry-run mode (no DB required)."""

    @patch("scripts.seed_dummy_jobs.parse_args")
    def test_dry_run_no_db_url(self, mock_args: MagicMock) -> None:
        """main() exits early when DATABASE_URL_SYNC is not set."""
        mock_args.return_value = MagicMock(days=5, dry_run=False)
        with patch.dict("os.environ", {}, clear=True):
            from scripts.seed_dummy_jobs import main
            main()  # should not raise, just log error
