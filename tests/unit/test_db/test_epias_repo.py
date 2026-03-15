"""Tests for EpiasRepository.

pg_insert (ON CONFLICT DO UPDATE) is PostgreSQL-only. Upsert tests use AsyncMock.
Read tests also mock session.execute to avoid SQLite/PG _mapping differences.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.repositories.epias_repo import EpiasRepository
from energy_forecast.utils import TZ_ISTANBUL


class TestUpsertMarket:
    """Tests for upsert_market method."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self) -> None:
        """upsert_market with empty list returns 0."""
        repo = EpiasRepository(AsyncMock(spec=AsyncSession))
        assert await repo.upsert_market([]) == 0

    @pytest.mark.asyncio
    async def test_calls_execute(self) -> None:
        """upsert_market calls session.execute for non-empty rows."""
        mock_session = AsyncMock(spec=AsyncSession)

        with patch("energy_forecast.db.repositories.epias_repo.pg_insert") as mock_pg:
            mock_stmt = MagicMock()
            mock_pg.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_stmt.excluded = {col: col for col in [
                "fdpp", "rtc", "dam_purchase", "bilateral", "load_forecast", "fetched_at",
            ]}

            repo = EpiasRepository(mock_session)
            count = await repo.upsert_market([
                {"datetime": datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL), "rtc": 100.0},
            ])

        assert count == 1
        mock_session.execute.assert_awaited_once()


class TestUpsertGeneration:
    """Tests for upsert_generation method."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self) -> None:
        """upsert_generation with empty list returns 0."""
        repo = EpiasRepository(AsyncMock(spec=AsyncSession))
        assert await repo.upsert_generation([]) == 0

    @pytest.mark.asyncio
    async def test_calls_execute(self) -> None:
        """upsert_generation calls session.execute for non-empty rows."""
        mock_session = AsyncMock(spec=AsyncSession)

        with patch("energy_forecast.db.repositories.epias_repo.pg_insert") as mock_pg:
            mock_stmt = MagicMock()
            mock_pg.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_stmt.excluded = {col: col for col in [
                "gen_total", "gen_natural_gas", "fetched_at",
            ]}

            repo = EpiasRepository(mock_session)
            count = await repo.upsert_generation([
                {"datetime": datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL), "gen_total": 50000.0},
            ])

        assert count == 1
        mock_session.execute.assert_awaited_once()


class TestGetMarketRange:
    """Tests for get_market_range method."""

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """get_market_range returns empty DataFrame when no data."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = EpiasRepository(mock_session)
        df = await repo.get_market_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert df.empty

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        """get_market_range returns DataFrame with datetime index."""
        mock_session = AsyncMock(spec=AsyncSession)
        row = MagicMock()
        row._mapping = {
            "datetime": datetime(2026, 1, 1, 0, 0, tzinfo=TZ_ISTANBUL),
            "rtc": 1500.0,
            "fdpp": 1400.0,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = EpiasRepository(mock_session)
        df = await repo.get_market_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert len(df) == 1
        assert df.index.name == "datetime"


class TestGetGenerationRange:
    """Tests for get_generation_range method."""

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """get_generation_range returns empty DataFrame when no data."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = EpiasRepository(mock_session)
        df = await repo.get_generation_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert df.empty

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        """get_generation_range returns DataFrame with datetime index."""
        mock_session = AsyncMock(spec=AsyncSession)
        row = MagicMock()
        row._mapping = {
            "datetime": datetime(2026, 1, 1, 0, 0, tzinfo=TZ_ISTANBUL),
            "gen_total": 50000.0,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = EpiasRepository(mock_session)
        df = await repo.get_generation_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert len(df) == 1
        assert df.index.name == "datetime"


class TestGetLatestMarketDatetime:
    """Tests for get_latest_market_datetime method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self) -> None:
        """get_latest_market_datetime returns None when table is empty."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = EpiasRepository(mock_session)
        result = await repo.get_latest_market_datetime()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_datetime(self) -> None:
        """get_latest_market_datetime returns latest timestamp."""
        mock_session = AsyncMock(spec=AsyncSession)
        expected = datetime(2026, 1, 15, 23, 0, tzinfo=TZ_ISTANBUL)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected
        mock_session.execute.return_value = mock_result

        repo = EpiasRepository(mock_session)
        result = await repo.get_latest_market_datetime()
        assert result == expected
