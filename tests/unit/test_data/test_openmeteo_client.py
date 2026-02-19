"""Unit tests for OpenMeteoClient (official SDK)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import openmeteo_requests
import pandas as pd
import pytest

from energy_forecast.config.settings import (
    CityConfig,
    GeocodingConfig,
    OpenMeteoApiConfig,
    OpenMeteoConfig,
    RegionConfig,
    WeatherCacheConfig,
)
from energy_forecast.data.exceptions import OpenMeteoApiError
from energy_forecast.data.openmeteo_client import OpenMeteoClient

# ---------------------------------------------------------------------------
# Mock helpers for FlatBuffers SDK responses
# ---------------------------------------------------------------------------


class MockVariable:
    """Mock openmeteo_sdk VariableWithValues."""

    def __init__(self, values: np.ndarray[Any, Any]) -> None:
        self._values = values

    def ValuesAsNumpy(self) -> np.ndarray[Any, Any]:  # noqa: N802
        return self._values


class MockHourly:
    """Mock openmeteo_sdk VariablesWithTime (Hourly block)."""

    def __init__(
        self,
        variables: list[np.ndarray[Any, Any]],
        time_start: int,
        time_end: int,
        interval: int = 3600,
    ) -> None:
        self._variables = [MockVariable(v) for v in variables]
        self._time_start = time_start
        self._time_end = time_end
        self._interval = interval

    def Variables(self, index: int) -> MockVariable:  # noqa: N802
        return self._variables[index]

    def Time(self) -> int:  # noqa: N802
        return self._time_start

    def TimeEnd(self) -> int:  # noqa: N802
        return self._time_end

    def Interval(self) -> int:  # noqa: N802
        return self._interval


class MockWeatherResponse:
    """Mock openmeteo_sdk WeatherApiResponse."""

    def __init__(
        self,
        hourly: MockHourly | None,
        utc_offset: int = 10800,
    ) -> None:
        self._hourly = hourly
        self._utc_offset = utc_offset

    def Hourly(self) -> MockHourly | None:  # noqa: N802
        return self._hourly

    def UtcOffsetSeconds(self) -> int:  # noqa: N802
        return self._utc_offset


def _make_mock_response(
    hours: int = 24,
    temp_base: float = 10.0,
    precip_base: float = 0.0,
) -> MockWeatherResponse:
    """Build a mock SDK response for 2 variables (temperature_2m, precipitation).

    Timestamps use 2024-01-01 00:00 UTC as base (epoch 1704067200).
    UTC offset = 10800 (Europe/Istanbul = UTC+3).
    """
    # Epoch for 2024-01-01T00:00:00 UTC minus 3h offset
    # because SDK Time() is in UTC, and we add utc_offset in the client
    time_start = 1704067200 - 10800  # adjust so that after +utc_offset → midnight
    time_end = time_start + hours * 3600

    temp_values = np.array([temp_base + i * 0.5 for i in range(hours)], dtype=np.float32)
    precip_values = np.full(hours, precip_base, dtype=np.float32)

    hourly = MockHourly(
        variables=[temp_values, precip_values],
        time_start=time_start,
        time_end=time_end,
    )
    return MockWeatherResponse(hourly=hourly)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_region() -> RegionConfig:
    """Two-city region for simpler testing."""
    return RegionConfig(
        name="Test",
        cities=[
            CityConfig(name="CityA", weight=0.6, latitude=40.0, longitude=29.0),
            CityConfig(name="CityB", weight=0.4, latitude=39.0, longitude=28.0),
        ],
    )


@pytest.fixture()
def test_config(tmp_path: Path) -> OpenMeteoConfig:
    """OpenMeteo config with temp cache path and 2 variables."""
    return OpenMeteoConfig(
        api=OpenMeteoApiConfig(
            base_url_historical="https://archive-api.open-meteo.com/v1/archive",
            base_url_forecast="https://api.open-meteo.com/v1/forecast",
            base_url_historical_forecast=(
                "https://historical-forecast-api.open-meteo.com/v1/forecast"
            ),
            timeout=10,
            retry_attempts=1,
            backoff_factor=0.1,
        ),
        variables=["temperature_2m", "precipitation"],
        cache=WeatherCacheConfig(path=str(tmp_path / "test_cache.db")),
    )


@pytest.fixture()
def client(test_config: OpenMeteoConfig, test_region: RegionConfig) -> OpenMeteoClient:
    """Create test OpenMeteoClient."""
    return OpenMeteoClient(config=test_config, region=test_region)


# ---------------------------------------------------------------------------
# Tests: fetch_historical
# ---------------------------------------------------------------------------


class TestFetchHistorical:
    """Tests for OpenMeteoClient.fetch_historical()."""

    def test_returns_dataframe(self, client: OpenMeteoClient) -> None:
        """Historical fetch returns DataFrame with correct columns."""
        resp_a = _make_mock_response(temp_base=10.0)
        resp_b = _make_mock_response(temp_base=20.0)

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_a], [resp_b]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert "temperature_2m" in df.columns
        assert "precipitation" in df.columns
        assert len(df) == 24

    def test_datetime_index(self, client: OpenMeteoClient) -> None:
        """Output has DatetimeIndex named 'datetime'."""
        resp = _make_mock_response()
        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp], [resp]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "datetime"


# ---------------------------------------------------------------------------
# Tests: fetch_forecast
# ---------------------------------------------------------------------------


class TestFetchForecast:
    """Tests for OpenMeteoClient.fetch_forecast()."""

    def test_returns_dataframe(self, client: OpenMeteoClient) -> None:
        """Forecast fetch returns DataFrame."""
        resp = _make_mock_response(hours=48)
        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp], [resp]],
        ):
            df = client.fetch_forecast(forecast_days=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Tests: fetch_historical_forecast
# ---------------------------------------------------------------------------


class TestFetchHistoricalForecast:
    """Tests for OpenMeteoClient.fetch_historical_forecast()."""

    def test_returns_dataframe(self, client: OpenMeteoClient) -> None:
        """Historical forecast fetch returns DataFrame."""
        resp = _make_mock_response(hours=48)
        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp], [resp]],
        ):
            df = client.fetch_historical_forecast("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48

    def test_uses_historical_forecast_url(self, client: OpenMeteoClient) -> None:
        """Historical forecast uses the correct endpoint URL."""
        resp = _make_mock_response()
        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp], [resp]],
        ) as mock_api:
            client.fetch_historical_forecast("2024-01-01", "2024-01-01")

        first_call_url = mock_api.call_args_list[0][0][0]
        assert "historical-forecast-api" in first_call_url


# ---------------------------------------------------------------------------
# Tests: weighted average
# ---------------------------------------------------------------------------


class TestWeightedAverage:
    """Tests for weighted average calculation."""

    def test_weighted_average_correct(self, client: OpenMeteoClient) -> None:
        """Weighted average: CityA(0.6)*10 + CityB(0.4)*20 = 14.0 at hour 0."""
        resp_a = _make_mock_response(temp_base=10.0)
        resp_b = _make_mock_response(temp_base=20.0)

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_a], [resp_b]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        expected = 0.6 * 10.0 + 0.4 * 20.0
        assert abs(df["temperature_2m"].iloc[0] - expected) < 0.01

    def test_weighted_average_nan_renormalized(self, client: OpenMeteoClient) -> None:
        """When one city has NaN, weights are re-normalized using remaining cities."""
        resp_a = _make_mock_response(temp_base=10.0)  # CityA: temp=10.0
        resp_b = _make_mock_response(temp_base=20.0)  # CityB: temp=20.0

        # Inject NaN into CityA temperature for first 3 hours
        hourly_a = resp_a.Hourly()
        assert hourly_a is not None
        temp_a = hourly_a.Variables(0).ValuesAsNumpy()
        temp_a[:3] = np.nan

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_a], [resp_b]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        # For hours 0-2: CityA is NaN, so result = CityB's value (re-normalized to weight 1.0)
        assert abs(df["temperature_2m"].iloc[0] - 20.0) < 0.01
        assert abs(df["temperature_2m"].iloc[1] - 20.5) < 0.01
        assert abs(df["temperature_2m"].iloc[2] - 21.0) < 0.01

        # For hour 3: both cities valid, normal weighted average
        expected_h3 = 0.6 * (10.0 + 3 * 0.5) + 0.4 * (20.0 + 3 * 0.5)
        assert abs(df["temperature_2m"].iloc[3] - expected_h3) < 0.01

    def test_weighted_average_all_nan_stays_nan(self, client: OpenMeteoClient) -> None:
        """When all cities are NaN for a timestep, result is NaN."""
        resp_a = _make_mock_response(temp_base=10.0)
        resp_b = _make_mock_response(temp_base=20.0)

        # Inject NaN into both cities for hour 0
        hourly_a = resp_a.Hourly()
        hourly_b = resp_b.Hourly()
        assert hourly_a is not None and hourly_b is not None
        hourly_a.Variables(0).ValuesAsNumpy()[0] = np.nan
        hourly_b.Variables(0).ValuesAsNumpy()[0] = np.nan

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_a], [resp_b]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        # Hour 0: all NaN → result NaN
        assert pd.isna(df["temperature_2m"].iloc[0])
        # Hour 1: both valid → normal weighted average
        assert not pd.isna(df["temperature_2m"].iloc[1])

    def test_config_variables_used(self, client: OpenMeteoClient) -> None:
        """Only configured variables appear in output."""
        resp = _make_mock_response()
        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp], [resp]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert set(df.columns) == {"temperature_2m", "precipitation"}


# ---------------------------------------------------------------------------
# Tests: weather_code dominant city strategy
# ---------------------------------------------------------------------------


class TestWeatherCodeDominantCity:
    """weather_code must use dominant city, not weighted average."""

    def test_weather_code_uses_dominant_city(self, tmp_path: Path) -> None:
        """weather_code takes the highest-weight city's value, not average."""
        config = OpenMeteoConfig(
            variables=["temperature_2m", "weather_code"],
            cache=WeatherCacheConfig(path=str(tmp_path / "wc_cache.db")),
        )
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="Bursa", weight=0.6, latitude=40.0, longitude=29.0),
                CityConfig(name="Balikesir", weight=0.4, latitude=39.0, longitude=28.0),
            ],
        )
        client = OpenMeteoClient(config=config, region=region)

        hours = 24
        time_start = 1704067200 - 10800
        time_end = time_start + hours * 3600

        # Bursa: rain (61), Balikesir: clear (0)
        resp_bursa = MockWeatherResponse(
            hourly=MockHourly(
                variables=[
                    np.full(hours, 10.0, dtype=np.float32),
                    np.full(hours, 61.0, dtype=np.float32),  # rain
                ],
                time_start=time_start,
                time_end=time_end,
            )
        )
        resp_balikesir = MockWeatherResponse(
            hourly=MockHourly(
                variables=[
                    np.full(hours, 20.0, dtype=np.float32),
                    np.full(hours, 0.0, dtype=np.float32),  # clear
                ],
                time_start=time_start,
                time_end=time_end,
            )
        )

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_bursa], [resp_balikesir]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        # weather_code should be Bursa's 61 (dominant), NOT 0.6*61 + 0.4*0 = 36.6
        assert df["weather_code"].iloc[0] == 61.0
        # temperature should still be weighted average
        expected_temp = 0.6 * 10.0 + 0.4 * 20.0
        assert abs(df["temperature_2m"].iloc[0] - expected_temp) < 0.01

    def test_weather_code_nan_fallback(self, tmp_path: Path) -> None:
        """When dominant city has NaN, falls back to next-highest-weight city."""
        config = OpenMeteoConfig(
            variables=["temperature_2m", "weather_code"],
            cache=WeatherCacheConfig(path=str(tmp_path / "wc_cache.db")),
        )
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="Bursa", weight=0.6, latitude=40.0, longitude=29.0),
                CityConfig(name="Balikesir", weight=0.4, latitude=39.0, longitude=28.0),
            ],
        )
        client = OpenMeteoClient(config=config, region=region)

        hours = 24
        time_start = 1704067200 - 10800
        time_end = time_start + hours * 3600

        # Bursa: NaN weather_code for first 3 hours
        wc_bursa = np.full(hours, 61.0, dtype=np.float32)
        wc_bursa[:3] = np.nan

        resp_bursa = MockWeatherResponse(
            hourly=MockHourly(
                variables=[
                    np.full(hours, 10.0, dtype=np.float32),
                    wc_bursa,
                ],
                time_start=time_start,
                time_end=time_end,
            )
        )
        resp_balikesir = MockWeatherResponse(
            hourly=MockHourly(
                variables=[
                    np.full(hours, 20.0, dtype=np.float32),
                    np.full(hours, 3.0, dtype=np.float32),  # overcast
                ],
                time_start=time_start,
                time_end=time_end,
            )
        )

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_bursa], [resp_balikesir]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        # Hours 0-2: Bursa NaN → fallback to Balikesir (3)
        assert df["weather_code"].iloc[0] == 3.0
        assert df["weather_code"].iloc[1] == 3.0
        assert df["weather_code"].iloc[2] == 3.0
        # Hour 3+: Bursa valid → use Bursa (61)
        assert df["weather_code"].iloc[3] == 61.0

    def test_weather_code_all_nan(self, tmp_path: Path) -> None:
        """When all cities have NaN weather_code, result is NaN."""
        config = OpenMeteoConfig(
            variables=["temperature_2m", "weather_code"],
            cache=WeatherCacheConfig(path=str(tmp_path / "wc_cache.db")),
        )
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="A", weight=0.6, latitude=40.0, longitude=29.0),
                CityConfig(name="B", weight=0.4, latitude=39.0, longitude=28.0),
            ],
        )
        client = OpenMeteoClient(config=config, region=region)

        hours = 24
        time_start = 1704067200 - 10800
        time_end = time_start + hours * 3600

        wc_a = np.full(hours, 61.0, dtype=np.float32)
        wc_a[0] = np.nan
        wc_b = np.full(hours, 3.0, dtype=np.float32)
        wc_b[0] = np.nan

        resp_a = MockWeatherResponse(
            hourly=MockHourly(
                variables=[np.full(hours, 10.0, dtype=np.float32), wc_a],
                time_start=time_start,
                time_end=time_end,
            )
        )
        resp_b = MockWeatherResponse(
            hourly=MockHourly(
                variables=[np.full(hours, 20.0, dtype=np.float32), wc_b],
                time_start=time_start,
                time_end=time_end,
            )
        )

        with patch.object(
            client._client,
            "weather_api",
            side_effect=[[resp_a], [resp_b]],
        ):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert pd.isna(df["weather_code"].iloc[0])
        assert df["weather_code"].iloc[1] == 61.0  # A is dominant


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    def test_api_error_raises(self, client: OpenMeteoClient) -> None:
        """SDK error raises OpenMeteoApiError."""
        with (
            patch.object(
                client._client,
                "weather_api",
                side_effect=openmeteo_requests.OpenMeteoRequestsError("Server Error"),
            ),
            pytest.raises(OpenMeteoApiError, match="OpenMeteo API error"),
        ):
            client._fetch_single_location(
                url="https://example.com",
                latitude=40.0,
                longitude=29.0,
                start_date="2024-01-01",
                end_date="2024-01-01",
            )

    def test_no_hourly_raises(self, client: OpenMeteoClient) -> None:
        """Missing hourly data raises OpenMeteoApiError."""
        resp = MockWeatherResponse(hourly=None)
        with (
            patch.object(
                client._client,
                "weather_api",
                return_value=[resp],
            ),
            pytest.raises(OpenMeteoApiError, match="missing hourly"),
        ):
            client._fetch_single_location(
                url="https://example.com",
                latitude=40.0,
                longitude=29.0,
                start_date="2024-01-01",
                end_date="2024-01-01",
            )

    def test_generic_exception_raises(self, client: OpenMeteoClient) -> None:
        """Generic exception is wrapped in OpenMeteoApiError."""
        with (
            patch.object(
                client._client,
                "weather_api",
                side_effect=ConnectionError("network down"),
            ),
            pytest.raises(OpenMeteoApiError, match="request failed"),
        ):
            client._fetch_single_location(
                url="https://example.com",
                latitude=40.0,
                longitude=29.0,
                start_date="2024-01-01",
                end_date="2024-01-01",
            )


# ---------------------------------------------------------------------------
# Tests: parse SDK response
# ---------------------------------------------------------------------------


class TestParseSdkResponse:
    """Tests for SDK response parsing."""

    def test_parse_valid_response(self, client: OpenMeteoClient) -> None:
        """Valid SDK response is parsed correctly."""
        resp = _make_mock_response(hours=12, temp_base=5.0)
        df = client._parse_sdk_response(resp)
        assert len(df) == 12
        assert abs(df["temperature_2m"].iloc[0] - 5.0) < 0.01

    def test_output_schema_columns(
        self,
        tmp_path: Path,
        sample_openmeteo_response: dict[str, Any],
    ) -> None:
        """Full 11-variable response has all expected columns."""
        full_config = OpenMeteoConfig(
            cache=WeatherCacheConfig(path=str(tmp_path / "full_cache.db")),
        )
        full_region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="A", weight=1.0, latitude=40.0, longitude=29.0),
            ],
        )
        full_client = OpenMeteoClient(config=full_config, region=full_region)

        # Build a mock response with 11 variables
        rng = np.random.default_rng(42)
        variables = [rng.random(24).astype(np.float32) * 30 for _ in range(11)]
        time_start = 1704067200 - 10800
        hourly = MockHourly(
            variables=variables,
            time_start=time_start,
            time_end=time_start + 24 * 3600,
        )
        resp = MockWeatherResponse(hourly=hourly)

        df = full_client._parse_sdk_response(resp)
        assert len(df.columns) == 11


# ---------------------------------------------------------------------------
# Tests: geocoding
# ---------------------------------------------------------------------------


class TestGeocoding:
    """Tests for Geocoding API."""

    def test_geocoding_resolve(self, tmp_path: Path) -> None:
        """Geocoding resolves city name to coordinates."""
        config = OpenMeteoConfig(
            cache=WeatherCacheConfig(path=str(tmp_path / "geo_cache.db")),
            geocoding=GeocodingConfig(enabled=True),
        )
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="A", weight=1.0, latitude=40.0, longitude=29.0),
            ],
        )
        client = OpenMeteoClient(config=config, region=region)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "latitude": 40.19559,
                    "longitude": 29.06013,
                    "elevation": 155.0,
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(client._client, "_session", create=True) as mock_session:
            mock_session.get.return_value = mock_response
            coords = client.resolve_coordinates("Bursa")

        assert abs(coords["latitude"] - 40.19559) < 0.001
        assert abs(coords["longitude"] - 29.06013) < 0.001
        assert abs(coords["elevation"] - 155.0) < 0.1

    def test_geocoding_disabled(self, tmp_path: Path) -> None:
        """Geocoding raises error when disabled."""
        config = OpenMeteoConfig(
            cache=WeatherCacheConfig(path=str(tmp_path / "geo_cache.db")),
            geocoding=GeocodingConfig(enabled=False),
        )
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="A", weight=1.0, latitude=40.0, longitude=29.0),
            ],
        )
        client = OpenMeteoClient(config=config, region=region)

        with pytest.raises(OpenMeteoApiError, match="disabled"):
            client.resolve_coordinates("Bursa")

    def test_geocoding_no_results(self, tmp_path: Path) -> None:
        """Geocoding raises error when no results found."""
        config = OpenMeteoConfig(
            cache=WeatherCacheConfig(path=str(tmp_path / "geo_cache.db")),
            geocoding=GeocodingConfig(enabled=True),
        )
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="A", weight=1.0, latitude=40.0, longitude=29.0),
            ],
        )
        client = OpenMeteoClient(config=config, region=region)

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None

        with (
            patch.object(client._client, "_session", create=True) as mock_session,
            pytest.raises(OpenMeteoApiError, match="No geocoding results"),
        ):
            mock_session.get.return_value = mock_response
            client.resolve_coordinates("NonexistentCity")


# ---------------------------------------------------------------------------
# Tests: import of OpenMeteoRequestsError
# ---------------------------------------------------------------------------


class TestImports:
    """Verify SDK imports work."""

    def test_openmeteo_requests_error_importable(self) -> None:
        """OpenMeteoRequestsError is importable from the SDK."""
        assert hasattr(openmeteo_requests, "OpenMeteoRequestsError")

    def test_client_has_weather_api(self) -> None:
        """Client exposes weather_api method."""
        assert hasattr(openmeteo_requests.Client, "weather_api")
