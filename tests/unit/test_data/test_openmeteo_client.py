"""Unit tests for OpenMeteoClient."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast.config.settings import (
    CityConfig,
    OpenMeteoApiConfig,
    OpenMeteoConfig,
    RegionConfig,
    WeatherCacheConfig,
)
from energy_forecast.data.exceptions import OpenMeteoApiError
from energy_forecast.data.openmeteo_client import OpenMeteoClient


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
    """OpenMeteo config with temp cache path."""
    return OpenMeteoConfig(
        api=OpenMeteoApiConfig(
            base_url_historical="https://archive-api.open-meteo.com/v1/archive",
            base_url_forecast="https://api.open-meteo.com/v1/forecast",
            timeout=10,
            retry_attempts=1,
        ),
        variables=["temperature_2m", "precipitation"],
        cache=WeatherCacheConfig(path=str(tmp_path / "test_cache.db")),
    )


@pytest.fixture()
def client(test_config: OpenMeteoConfig, test_region: RegionConfig) -> OpenMeteoClient:
    """Create test OpenMeteoClient."""
    return OpenMeteoClient(config=test_config, region=test_region)


def _make_weather_response(
    hours: int = 24,
    temp_base: float = 10.0,
    precip_base: float = 0.0,
) -> dict[str, Any]:
    """Build a mock OpenMeteo JSON response."""
    times = pd.date_range("2024-01-01", periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [temp_base + i * 0.5 for i in range(hours)],
            "precipitation": [precip_base] * hours,
        },
    }


def _mock_get(response_data: dict[str, Any]) -> MagicMock:
    """Create mock httpx GET response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = response_data
    resp.raise_for_status.return_value = None
    return resp


class TestFetchHistorical:
    """Tests for OpenMeteoClient.fetch_historical()."""

    def test_returns_dataframe(self, client: OpenMeteoClient) -> None:
        """Historical fetch returns DataFrame with correct columns."""
        resp_a = _make_weather_response(temp_base=10.0)
        resp_b = _make_weather_response(temp_base=20.0)

        responses = [_mock_get(resp_a), _mock_get(resp_b)]
        with patch.object(client._client, "get", side_effect=responses):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert "temperature_2m" in df.columns
        assert "precipitation" in df.columns
        assert len(df) == 24

    def test_datetime_index(self, client: OpenMeteoClient) -> None:
        """Output has DatetimeIndex named 'datetime'."""
        resp = _make_weather_response()
        responses = [_mock_get(resp), _mock_get(resp)]
        with patch.object(client._client, "get", side_effect=responses):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "datetime"


class TestFetchForecast:
    """Tests for OpenMeteoClient.fetch_forecast()."""

    def test_returns_dataframe(self, client: OpenMeteoClient) -> None:
        """Forecast fetch returns DataFrame."""
        resp = _make_weather_response(hours=48)
        responses = [_mock_get(resp), _mock_get(resp)]
        with patch.object(client._client, "get", side_effect=responses):
            df = client.fetch_forecast(forecast_days=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestWeightedAverage:
    """Tests for weighted average calculation."""

    def test_weighted_average_correct(self, client: OpenMeteoClient) -> None:
        """Weighted average: CityA(0.6)*10 + CityB(0.4)*20 = 14.0 at hour 0."""
        resp_a = _make_weather_response(temp_base=10.0)
        resp_b = _make_weather_response(temp_base=20.0)

        responses = [_mock_get(resp_a), _mock_get(resp_b)]
        with patch.object(client._client, "get", side_effect=responses):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        # At hour 0: 0.6 * 10.0 + 0.4 * 20.0 = 14.0
        expected = 0.6 * 10.0 + 0.4 * 20.0
        assert abs(df["temperature_2m"].iloc[0] - expected) < 0.01

    def test_config_variables_used(self, client: OpenMeteoClient) -> None:
        """Only configured variables appear in output."""
        resp = _make_weather_response()
        responses = [_mock_get(resp), _mock_get(resp)]
        with patch.object(client._client, "get", side_effect=responses):
            df = client.fetch_historical("2024-01-01", "2024-01-01")

        assert set(df.columns) == {"temperature_2m", "precipitation"}


class TestSingleLocation:
    """Tests for single location fetch."""

    def test_single_location_parse(self, client: OpenMeteoClient) -> None:
        """Single location returns parsed DataFrame."""
        resp = _make_weather_response(hours=24, temp_base=15.0)
        with patch.object(client._client, "get", return_value=_mock_get(resp)):
            df = client._fetch_single_location(
                base_url="https://example.com",
                latitude=40.0,
                longitude=29.0,
                start_date="2024-01-01",
                end_date="2024-01-01",
            )
        assert len(df) == 24
        assert df["temperature_2m"].iloc[0] == 15.0


class TestErrorHandling:
    """Tests for error handling."""

    def test_api_error_raises(self, client: OpenMeteoClient) -> None:
        """HTTP error raises OpenMeteoApiError."""
        import httpx

        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=error_resp,
        )
        with (
            patch.object(client._client, "get", return_value=error_resp),
            pytest.raises(OpenMeteoApiError),
        ):
            client._fetch_single_location(
                base_url="https://example.com",
                latitude=40.0,
                longitude=29.0,
                start_date="2024-01-01",
                end_date="2024-01-01",
            )

    def test_invalid_response_raises(self, client: OpenMeteoClient) -> None:
        """Missing 'hourly' key raises OpenMeteoApiError."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"error": False}
        resp.raise_for_status.return_value = None

        with (
            patch.object(client._client, "get", return_value=resp),
            pytest.raises(OpenMeteoApiError, match="missing"),
        ):
            client._fetch_single_location(
                base_url="https://example.com",
                latitude=40.0,
                longitude=29.0,
                start_date="2024-01-01",
                end_date="2024-01-01",
            )


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_valid_response(self, client: OpenMeteoClient) -> None:
        """Valid JSON response is parsed correctly."""
        data = _make_weather_response(hours=12, temp_base=5.0)
        df = client._parse_response(data)
        assert len(df) == 12
        assert df["temperature_2m"].iloc[0] == 5.0

    def test_output_schema_columns(
        self,
        client: OpenMeteoClient,
        sample_openmeteo_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Full 11-variable response has all expected columns."""
        # Use a client with full variable list for this test
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

        df = full_client._parse_response(sample_openmeteo_response)
        assert len(df.columns) == 11
