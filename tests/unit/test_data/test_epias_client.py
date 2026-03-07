"""Unit tests for EpiasClient."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast.config.settings import EpiasApiConfig
from energy_forecast.data.epias_client import EpiasClient, _to_epias_timestamp
from energy_forecast.data.exceptions import EpiasApiError, EpiasAuthError


@pytest.fixture()
def client(tmp_path: Path) -> EpiasClient:
    """Create an EpiasClient with test config."""
    config = EpiasApiConfig(
        cache_dir=str(tmp_path / "epias_cache"),
        rate_limit_seconds=0.0,  # no delay in tests
    )
    return EpiasClient(
        username="test_user",
        password="test_pass",
        config=config,
    )


def _mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
) -> MagicMock:
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {}
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestAuthentication:
    """Tests for EpiasClient.authenticate()."""

    def test_authenticate_returns_token(self, client: EpiasClient) -> None:
        """Successful auth returns TGT token."""
        with patch.object(
            client._client,
            "post",
            return_value=_mock_response(200, text="TGT-12345"),
        ):
            token = client.authenticate()
        assert token == "TGT-12345"

    def test_authenticate_caches_token(self, client: EpiasClient) -> None:
        """Second call returns cached token without API call."""
        mock_post = MagicMock(return_value=_mock_response(200, text="TGT-12345"))
        with patch.object(client._client, "post", mock_post):
            client.authenticate()
            client.authenticate()
        assert mock_post.call_count == 1

    def test_authenticate_failure_raises(self, client: EpiasClient) -> None:
        """401 response raises EpiasAuthError."""
        with (
            patch.object(
                client._client,
                "post",
                return_value=_mock_response(401),
            ),
            pytest.raises(EpiasAuthError),
        ):
            client.authenticate()


class TestFetch:
    """Tests for EpiasClient.fetch()."""

    def _setup_mocked_client(
        self,
        client: EpiasClient,
        items: list[dict[str, Any]],
    ) -> Any:
        """Setup client with mocked auth and API responses."""
        auth_resp = _mock_response(200, text="TGT-12345")
        api_resp = _mock_response(200, json_data={"body": {"content": items}})

        mock_post = MagicMock(side_effect=[auth_resp] + [api_resp] * 10)
        patcher = patch.object(client._client, "post", mock_post)
        patcher.start()
        return patcher

    def test_fetch_returns_dataframe(self, client: EpiasClient) -> None:
        """Fetch returns a DataFrame with DatetimeIndex."""
        items = [
            {"date": f"2024-01-01T{h:02d}:00:00+03:00", "toplam": 100.0 + h} for h in range(24)
        ]
        patcher = self._setup_mocked_client(client, items)
        try:
            df = client.fetch("2024-01-01", "2024-01-01")
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) > 0
        finally:
            patcher.stop()

    def test_cache_hit_skips_api(self, client: EpiasClient) -> None:
        """When cache exists, API is not called for that year."""
        # Pre-populate cache
        idx = pd.date_range("2024-01-01", periods=24, freq="h", name="datetime")
        cached_df = pd.DataFrame({"Real_Time_Consumption": range(24)}, index=idx, dtype=float)
        client.save_cache(2024, cached_df)

        mock_post = MagicMock()
        with patch.object(client._client, "post", mock_post):
            df = client.load_cache(2024)
        assert df is not None
        assert len(df) == 24
        mock_post.assert_not_called()

    def test_cache_miss_fetches_api(self, client: EpiasClient) -> None:
        """When no cache, API is called and result is cached."""
        assert client.load_cache(2024) is None

        items = [{"date": f"2024-01-01T{h:02d}:00:00+03:00", "toplam": 100.0} for h in range(24)]
        patcher = self._setup_mocked_client(client, items)
        try:
            client.fetch_year(2024)
            cached = client.load_cache(2024)
            assert cached is not None
        finally:
            patcher.stop()

    def test_output_datetime_index(self, client: EpiasClient) -> None:
        """Output has correct DatetimeIndex name."""
        idx = pd.date_range("2024-01-01", periods=24, freq="h", name="datetime")
        df = pd.DataFrame({"Real_Time_Consumption": range(24)}, index=idx, dtype=float)
        client.save_cache(2024, df)

        loaded = client.load_cache(2024)
        assert loaded is not None
        assert loaded.index.name == "datetime"


class TestCacheReadWrite:
    """Tests for cache save/load round-trip."""

    def test_cache_round_trip(self, client: EpiasClient) -> None:
        """Save then load returns equivalent DataFrame."""
        idx = pd.date_range("2024-01-01", periods=48, freq="h", name="datetime")
        df = pd.DataFrame(
            {
                "Real_Time_Consumption": range(48),
                "DAM_Purchase": range(48),
            },
            index=idx,
            dtype=float,
        )

        client.save_cache(2024, df)
        loaded = client.load_cache(2024)

        assert loaded is not None
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)


class TestEpiasTimestamp:
    """Tests for _to_epias_timestamp helper."""

    def test_start_of_day(self) -> None:
        """Start of day timestamp format."""
        ts = _to_epias_timestamp("2024-01-15")
        assert "2024-01-15T00:00:00" in ts

    def test_end_of_day(self) -> None:
        """End of day timestamp format."""
        ts = _to_epias_timestamp("2024-01-15", end_of_day=True)
        assert "2024-01-15T23:00:00" in ts


class TestRetryBehavior:
    """Tests for retry and error handling."""

    def test_server_error_raises_after_retries(
        self,
        client: EpiasClient,
    ) -> None:
        """500 error after all retries raises EpiasApiError."""
        auth_resp = _mock_response(200, text="TGT-12345")
        error_resp = _mock_response(500)

        mock_post = MagicMock(side_effect=[auth_resp] + [error_resp] * 5)
        with (
            patch.object(client._client, "post", mock_post),
            pytest.raises(EpiasApiError),
        ):
            client._fetch_variable(
                endpoint="/test",
                response_key="value",
                column_name="test",
                start_date="2024-01-01",
                end_date="2024-01-01",
            )
