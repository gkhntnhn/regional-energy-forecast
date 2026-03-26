"""Tests for ProfileRepository.

pg_insert is PostgreSQL-only. All tests use AsyncMock for session.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.repositories.profile_repo import ProfileRepository
from energy_forecast.utils import TZ_ISTANBUL


class TestBulkUpsert:
    """Tests for bulk_upsert method."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self) -> None:
        """bulk_upsert with empty list returns 0."""
        repo = ProfileRepository(AsyncMock(spec=AsyncSession))
        assert await repo.bulk_upsert([]) == 0

    @pytest.mark.asyncio
    async def test_calls_execute(self) -> None:
        """bulk_upsert calls session.execute for non-empty rows."""
        mock_session = AsyncMock(spec=AsyncSession)

        with patch("energy_forecast.db.repositories.profile_repo.pg_insert") as mock_pg:
            mock_stmt = MagicMock()
            mock_pg.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_stmt.excluded = {
                col: col
                for col in [
                    "profile_residential_lv",
                    "fetched_at",
                ]
            }

            repo = ProfileRepository(mock_session)
            count = await repo.bulk_upsert(
                [
                    {
                        "datetime": datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
                        "profile_residential_lv": 1.0,
                    },
                ]
            )

        assert count == 1
        mock_session.execute.assert_awaited_once()


class TestGetByYear:
    """Tests for get_by_year method."""

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """get_by_year returns empty DataFrame when no data."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = ProfileRepository(mock_session)
        df = await repo.get_by_year(2026)
        assert df.empty

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        """get_by_year returns DataFrame with datetime index."""
        mock_session = AsyncMock(spec=AsyncSession)
        row = MagicMock()
        row._mapping = {
            "datetime": datetime(2026, 1, 1, 0, 0, tzinfo=TZ_ISTANBUL),
            "profile_residential_lv": 1.05,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = ProfileRepository(mock_session)
        df = await repo.get_by_year(2026)
        assert len(df) == 1
        assert df.index.name == "datetime"


class TestGetRange:
    """Tests for get_range method."""

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """get_range returns empty DataFrame when no data."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = ProfileRepository(mock_session)
        df = await repo.get_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 12, 31, tzinfo=TZ_ISTANBUL),
        )
        assert df.empty

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        """get_range returns DataFrame with datetime index."""
        mock_session = AsyncMock(spec=AsyncSession)
        row = MagicMock()
        row._mapping = {
            "datetime": datetime(2026, 6, 15, 12, 0, tzinfo=TZ_ISTANBUL),
            "profile_residential_lv": 0.98,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = ProfileRepository(mock_session)
        df = await repo.get_range(
            datetime(2026, 1, 1, tzinfo=TZ_ISTANBUL),
            datetime(2026, 12, 31, tzinfo=TZ_ISTANBUL),
        )
        assert len(df) == 1
        assert df.index.name == "datetime"
