"""Tests for HolidayRepository.

Since bulk_upsert uses pg_insert (PostgreSQL-only ON CONFLICT DO UPDATE),
upsert tests use AsyncMock. Read tests also use AsyncMock because the
``dict(r._mapping)`` pattern in the repo is designed for PostgreSQL rows.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.repositories.holiday_repo import HolidayRepository


class TestBulkUpsert:
    """Tests for bulk_upsert method."""

    @pytest.mark.asyncio
    async def test_bulk_upsert_empty_list(self, db_session: AsyncSession) -> None:
        """bulk_upsert with empty list returns 0 without hitting DB."""
        repo = HolidayRepository(db_session)
        count = await repo.bulk_upsert([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_bulk_upsert_calls_execute(self) -> None:
        """bulk_upsert with rows calls session.execute."""
        mock_session = AsyncMock(spec=AsyncSession)

        with patch("energy_forecast.db.repositories.holiday_repo.pg_insert") as mock_pg:
            mock_stmt = MagicMock()
            mock_pg.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_stmt.excluded = {
                col: col
                for col in [
                    "holiday_name",
                    "raw_holiday_name",
                    "is_ramadan",
                    "bayram_gun_no",
                    "bayrama_kalan_gun",
                ]
            }

            repo = HolidayRepository(mock_session)
            rows = [
                {
                    "date": date(2026, 1, 1),
                    "holiday_name": "Yilbasi",
                    "raw_holiday_name": "Yilbasi",
                    "is_ramadan": 0,
                    "bayram_gun_no": 0,
                    "bayrama_kalan_gun": -1,
                },
            ]
            count = await repo.bulk_upsert(rows)

        assert count == 1
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bulk_upsert_multiple_rows(self) -> None:
        """bulk_upsert returns correct count for multiple rows."""
        mock_session = AsyncMock(spec=AsyncSession)

        with patch("energy_forecast.db.repositories.holiday_repo.pg_insert") as mock_pg:
            mock_stmt = MagicMock()
            mock_pg.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_stmt.excluded = {
                col: col
                for col in [
                    "holiday_name",
                    "raw_holiday_name",
                    "is_ramadan",
                    "bayram_gun_no",
                    "bayrama_kalan_gun",
                ]
            }

            repo = HolidayRepository(mock_session)
            rows = [
                {
                    "date": date(2026, 1, 1),
                    "holiday_name": "Yilbasi",
                    "raw_holiday_name": "Yilbasi",
                    "is_ramadan": 0,
                    "bayram_gun_no": 0,
                    "bayrama_kalan_gun": -1,
                },
                {
                    "date": date(2026, 4, 23),
                    "holiday_name": "23 Nisan",
                    "raw_holiday_name": "23 Nisan",
                    "is_ramadan": 0,
                    "bayram_gun_no": 1,
                    "bayrama_kalan_gun": 0,
                },
            ]
            count = await repo.bulk_upsert(rows)

        assert count == 2


class TestGetAll:
    """Tests for get_all method."""

    @pytest.mark.asyncio
    async def test_get_all_empty(self) -> None:
        """get_all returns empty DataFrame when no holidays."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = HolidayRepository(mock_session)
        df = await repo.get_all()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_all_returns_dataframe(self) -> None:
        """get_all returns DataFrame with date index when data exists."""
        mock_session = AsyncMock(spec=AsyncSession)

        # Simulate mappings result
        row1 = MagicMock()
        row1._mapping = {
            "date": date(2026, 1, 1),
            "holiday_name": "Yilbasi",
            "raw_holiday_name": "Yilbasi",
            "is_ramadan": 0,
            "bayram_gun_no": 0,
            "bayrama_kalan_gun": -1,
        }
        row2 = MagicMock()
        row2._mapping = {
            "date": date(2026, 4, 23),
            "holiday_name": "23 Nisan",
            "raw_holiday_name": "23 Nisan",
            "is_ramadan": 0,
            "bayram_gun_no": 1,
            "bayrama_kalan_gun": 0,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row1, row2]
        mock_session.execute.return_value = mock_result

        repo = HolidayRepository(mock_session)
        df = await repo.get_all()
        assert len(df) == 2
        assert df.index.name == "date"
        assert "holiday_name" in df.columns

    @pytest.mark.asyncio
    async def test_get_all_calls_execute(self) -> None:
        """get_all executes a SELECT query."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = HolidayRepository(mock_session)
        await repo.get_all()
        mock_session.execute.assert_awaited_once()


class TestGetRange:
    """Tests for get_range method."""

    @pytest.mark.asyncio
    async def test_get_range_empty(self) -> None:
        """get_range returns empty DataFrame when no holidays in range."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = HolidayRepository(mock_session)
        df = await repo.get_range(date(2026, 6, 1), date(2026, 7, 1))
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_range_returns_dataframe(self) -> None:
        """get_range returns DataFrame with date index."""
        mock_session = AsyncMock(spec=AsyncSession)

        row = MagicMock()
        row._mapping = {
            "date": date(2026, 4, 23),
            "holiday_name": "23 Nisan",
            "raw_holiday_name": "23 Nisan",
            "is_ramadan": 0,
            "bayram_gun_no": 1,
            "bayrama_kalan_gun": 0,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        repo = HolidayRepository(mock_session)
        df = await repo.get_range(date(2026, 4, 1), date(2026, 5, 1))
        assert len(df) == 1
        assert df.index.name == "date"
        assert df.iloc[0]["holiday_name"] == "23 Nisan"

    @pytest.mark.asyncio
    async def test_get_range_calls_execute(self) -> None:
        """get_range executes a SELECT query with date filters."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = HolidayRepository(mock_session)
        await repo.get_range(date(2026, 1, 1), date(2026, 12, 31))
        mock_session.execute.assert_awaited_once()
