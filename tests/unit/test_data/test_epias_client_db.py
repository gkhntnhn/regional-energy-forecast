"""Tests for EpiasClient database integration (Faz 2).

Tests the DB-first load_cache and dual-write save_cache behavior.
Uses mock for SyncDataAccess to avoid PostgreSQL dependency.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_forecast.data.epias_client import EpiasClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def epias_config(tmp_path: Path) -> MagicMock:
    """Minimal EpiasApiConfig mock with tmp_path as cache_dir."""
    cfg = MagicMock()
    cfg.base_url = "https://seffaflik.epias.com.tr/api"
    cfg.auth_url = "https://cas.epias.com.tr/cas/v1/tickets"
    cfg.timeout_seconds = 30
    cfg.retry_attempts = 1
    cfg.backoff_factor = 0.1
    cfg.rate_limit_seconds = 0
    cfg.token_ttl_seconds = 3600
    cfg.cache_dir = str(tmp_path)
    cfg.file_pattern = "epias_market_{year}.parquet"
    cfg.generation_file_pattern = "epias_generation_{year}.parquet"
    return cfg


@pytest.fixture()
def sample_market_df() -> pd.DataFrame:
    """Sample market DataFrame with standard column names."""
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Real_Time_Consumption": rng.uniform(500, 1500, 24),
        "DAM_Purchase": rng.uniform(400, 1200, 24),
        "Load_Forecast": rng.uniform(600, 1600, 24),
    }, index=idx).rename_axis("datetime")


@pytest.fixture()
def sample_generation_df() -> pd.DataFrame:
    """Sample generation DataFrame with gen_ prefixed columns."""
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "gen_natural_gas": rng.uniform(1000, 5000, 24),
        "gen_wind": rng.uniform(100, 1000, 24),
        "gen_total": rng.uniform(5000, 15000, 24),
    }, index=idx).rename_axis("datetime")


def _make_client(config: MagicMock) -> EpiasClient:
    """Create EpiasClient without DB session."""
    return EpiasClient(config=config, username="test", password="test")


def _make_client_with_mock_db(
    config: MagicMock,
) -> tuple[EpiasClient, MagicMock]:
    """Create EpiasClient with mocked _db attribute."""
    client = _make_client(config)
    mock_dao = MagicMock()
    client._db = mock_dao
    return client, mock_dao


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


class TestDfToMarketRows:
    """Test _df_to_market_rows static method."""

    def test_column_mapping(self, sample_market_df: pd.DataFrame) -> None:
        """DataFrame columns are mapped to DB column names."""
        rows = EpiasClient._df_to_market_rows(sample_market_df)
        assert len(rows) == 24
        first = rows[0]
        assert "rtc" in first
        assert "dam_purchase" in first
        assert "load_forecast" in first
        assert "datetime" in first
        assert "Real_Time_Consumption" not in first

    def test_nan_handling(self) -> None:
        """NaN values are converted to None."""
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        df = pd.DataFrame({
            "Real_Time_Consumption": [100.0, float("nan")],
        }, index=idx).rename_axis("datetime")

        rows = EpiasClient._df_to_market_rows(df)
        assert rows[0]["rtc"] == 100.0
        assert rows[1]["rtc"] is None


class TestDfToGenerationRows:
    """Test _df_to_generation_rows static method."""

    def test_gen_prefix_filter(self, sample_generation_df: pd.DataFrame) -> None:
        """Only gen_ prefixed columns are included."""
        rows = EpiasClient._df_to_generation_rows(sample_generation_df)
        assert len(rows) == 24
        first = rows[0]
        assert "gen_natural_gas" in first
        assert "gen_wind" in first
        assert "gen_total" in first
        assert "datetime" in first


# ---------------------------------------------------------------------------
# DB-first load_cache
# ---------------------------------------------------------------------------


class TestLoadCacheDbFirst:
    """Test load_cache DB-first behavior."""

    def test_db_first_returns_data(self, epias_config: MagicMock) -> None:
        """When DB has data, parquet is not read."""
        client, mock_dao = _make_client_with_mock_db(epias_config)

        idx = pd.date_range("2024-01-01", periods=24, freq="h")
        db_df = pd.DataFrame(
            {"rtc": np.random.default_rng(42).uniform(500, 1500, 24)},
            index=idx,
        ).rename_axis("datetime")
        mock_dao.get_epias_market_year.return_value = db_df

        result = client.load_cache(2024)
        assert result is not None
        assert len(result) == 24
        mock_dao.get_epias_market_year.assert_called_once_with(2024)

    def test_db_empty_falls_back_to_parquet(
        self, epias_config: MagicMock,
        sample_market_df: pd.DataFrame,
    ) -> None:
        """When DB is empty, falls back to parquet file."""
        client, mock_dao = _make_client_with_mock_db(epias_config)
        mock_dao.get_epias_market_year.return_value = pd.DataFrame()

        # Write parquet fallback (cache_dir comes from epias_config)
        path = Path(epias_config.cache_dir) / "epias_market_2024.parquet"
        save_df = sample_market_df.reset_index()
        save_df.to_parquet(path, engine="pyarrow")

        result = client.load_cache(2024)
        assert result is not None
        assert len(result) == 24

    def test_no_db_session_uses_parquet(
        self, epias_config: MagicMock,
        sample_market_df: pd.DataFrame,
    ) -> None:
        """Without db_session, only parquet is used."""
        path = Path(epias_config.cache_dir) / "epias_market_2024.parquet"
        save_df = sample_market_df.reset_index()
        save_df.to_parquet(path, engine="pyarrow")

        client = _make_client(epias_config)
        result = client.load_cache(2024)
        assert result is not None
        assert len(result) == 24

    def test_no_db_no_parquet_returns_none(
        self, epias_config: MagicMock,
    ) -> None:
        """Without DB or parquet, returns None."""
        client = _make_client(epias_config)
        assert client.load_cache(2024) is None


# ---------------------------------------------------------------------------
# Dual-write save_cache
# ---------------------------------------------------------------------------


class TestSaveCacheDualWrite:
    """Test save_cache dual-write (DB + parquet)."""

    def test_dual_write_market(
        self, epias_config: MagicMock,
        sample_market_df: pd.DataFrame,
    ) -> None:
        """save_cache writes to both DB and parquet for market data."""
        client, mock_dao = _make_client_with_mock_db(epias_config)
        mock_dao.upsert_epias_market.return_value = 24

        client.save_cache(2024, sample_market_df)

        mock_dao.upsert_epias_market.assert_called_once()
        parquet_path = Path(epias_config.cache_dir) / "epias_market_2024.parquet"
        assert parquet_path.exists()

    def test_dual_write_generation(
        self, epias_config: MagicMock,
        sample_generation_df: pd.DataFrame,
    ) -> None:
        """save_cache writes generation data to DB with correct method."""
        client, mock_dao = _make_client_with_mock_db(epias_config)
        mock_dao.upsert_epias_generation.return_value = 24

        gen_pattern = epias_config.generation_file_pattern
        client.save_cache(2024, sample_generation_df, file_pattern=gen_pattern)

        mock_dao.upsert_epias_generation.assert_called_once()

    def test_db_failure_still_writes_parquet(
        self, epias_config: MagicMock,
        sample_market_df: pd.DataFrame,
    ) -> None:
        """DB write failure is non-fatal; parquet is still written."""
        client, mock_dao = _make_client_with_mock_db(epias_config)
        mock_dao.upsert_epias_market.side_effect = Exception("DB error")

        # Should not raise
        client.save_cache(2024, sample_market_df)

        parquet_path = Path(epias_config.cache_dir) / "epias_market_2024.parquet"
        assert parquet_path.exists()


# ---------------------------------------------------------------------------
# _is_generation_pattern
# ---------------------------------------------------------------------------


class TestIsGenerationPattern:
    """Test file pattern detection."""

    def test_none_is_not_generation(self, epias_config: MagicMock) -> None:
        """None file_pattern is market (default)."""
        client = _make_client(epias_config)
        assert not client._is_generation_pattern(None)

    def test_generation_pattern(self, epias_config: MagicMock) -> None:
        """Matching generation pattern returns True."""
        client = _make_client(epias_config)
        assert client._is_generation_pattern(
            epias_config.generation_file_pattern,
        )

    def test_market_pattern_is_not_generation(
        self, epias_config: MagicMock,
    ) -> None:
        """Market pattern returns False."""
        client = _make_client(epias_config)
        assert not client._is_generation_pattern("epias_market_{year}.parquet")
