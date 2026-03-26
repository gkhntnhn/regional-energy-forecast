"""Tests for WeatherCacheRepository.

pg_insert is PostgreSQL-only. Upsert and read tests use AsyncMock.
delete_stale is tested with real SQLite.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import WeatherCacheModel
from energy_forecast.db.repositories.weather_cache_repo import WeatherCacheRepository
from energy_forecast.utils import TZ_ISTANBUL


class TestUpsert:
    """Tests for upsert method."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self) -> None:
        """upsert with empty list returns 0."""
        repo = WeatherCacheRepository(AsyncMock(spec=AsyncSession))
        assert await repo.upsert([]) == 0

    @pytest.mark.asyncio
    async def test_calls_execute(self) -> None:
        """upsert calls session.execute for non-empty rows."""
        mock_session = AsyncMock(spec=AsyncSession)

        with patch("energy_forecast.db.repositories.weather_cache_repo.pg_insert") as mock_pg:
            mock_stmt = MagicMock()
            mock_pg.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_stmt.excluded = {
                col: col
                for col in [
                    "temperature_2m",
                    "fetched_at",
                ]
            }

            repo = WeatherCacheRepository(mock_session)
            count = await repo.upsert(
                [
                    {
                        "datetime": datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
                        "city": "Bursa",
                        "source": "historical",
                        "temperature_2m": 5.0,
                    },
                ]
            )

        assert count == 1
        mock_session.execute.assert_awaited_once()


class TestGetRange:
    """Tests for get_range method."""

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """get_range returns empty DataFrame when no data."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = WeatherCacheRepository(mock_session)
        df = await repo.get_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert df.empty

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        """get_range returns DataFrame with data."""
        mock_session = AsyncMock(spec=AsyncSession)
        row = MagicMock()
        row._mapping = {
            "datetime": datetime(2026, 1, 1, 0, 0, tzinfo=TZ_ISTANBUL),
            "city": "Bursa",
            "source": "historical",
            "temperature_2m": 5.0,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = WeatherCacheRepository(mock_session)
        df = await repo.get_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert len(df) == 1

    @pytest.mark.asyncio
    async def test_source_filter(self) -> None:
        """get_range with source filter calls execute with extra WHERE."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = WeatherCacheRepository(mock_session)
        await repo.get_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
            source="forecast",
        )
        mock_session.execute.assert_awaited_once()


class TestGetByCity:
    """Tests for get_by_city method."""

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """get_by_city returns empty DataFrame when no data."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = WeatherCacheRepository(mock_session)
        df = await repo.get_by_city(
            "Bursa",
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert df.empty

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        """get_by_city returns DataFrame with data."""
        mock_session = AsyncMock(spec=AsyncSession)
        row = MagicMock()
        row._mapping = {
            "datetime": datetime(2026, 1, 1, 0, 0, tzinfo=TZ_ISTANBUL),
            "city": "Bursa",
            "source": "historical",
            "temperature_2m": 5.0,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = WeatherCacheRepository(mock_session)
        df = await repo.get_by_city(
            "Bursa",
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 1, 2, tzinfo=TZ_ISTANBUL),
        )
        assert len(df) == 1


class TestDeleteStale:
    """Tests for delete_stale method (real SQLite)."""

    @pytest.mark.asyncio
    async def test_delete_stale_no_rows(self, db_session: AsyncSession) -> None:
        """delete_stale returns 0 when no rows match."""
        repo = WeatherCacheRepository(db_session)
        count = await repo.delete_stale(datetime(2020, 1, 1, tzinfo=TZ_ISTANBUL))
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_stale_removes_old_rows(self, db_session: AsyncSession) -> None:
        """delete_stale removes rows with fetched_at before cutoff."""
        # Add a row with old fetched_at
        old = WeatherCacheModel(
            dt=datetime(2025, 6, 1, 0, 0, tzinfo=TZ_ISTANBUL),
            city="Bursa",
            source="historical",
            temperature_2m=20.0,
            fetched_at=datetime(2025, 1, 1, tzinfo=TZ_ISTANBUL),
        )
        db_session.add(old)
        await db_session.flush()

        repo = WeatherCacheRepository(db_session)
        count = await repo.delete_stale(datetime(2025, 6, 1, tzinfo=TZ_ISTANBUL))
        assert count == 1
