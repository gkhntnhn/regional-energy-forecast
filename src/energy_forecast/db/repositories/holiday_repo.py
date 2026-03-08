"""Holiday repository — upsert and query Turkish holiday calendar."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import TurkishHolidayModel

_HOLIDAY_UPDATE_COLS = [
    "holiday_name",
    "raw_holiday_name",
    "is_ramadan",
    "bayram_gun_no",
    "bayrama_kalan_gun",
]


class HolidayRepository:
    """Async data access layer for turkish_holidays table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert holiday rows. ON CONFLICT on date DO UPDATE.

        Args:
            rows: List of dicts with keys: date, holiday_name, raw_holiday_name,
                  is_ramadan, bayram_gun_no, bayrama_kalan_gun.

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0
        stmt = pg_insert(TurkishHolidayModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],
            set_={col: stmt.excluded[col] for col in _HOLIDAY_UPDATE_COLS},
        )
        await self._session.execute(stmt)
        return len(rows)

    async def get_range(self, start: date, end: date) -> pd.DataFrame:
        """Get holidays between start and end dates (inclusive).

        Args:
            start: Start date.
            end: End date.

        Returns:
            DataFrame with date index and holiday columns.
        """
        stmt = (
            select(TurkishHolidayModel)
            .where(TurkishHolidayModel.date >= start)
            .where(TurkishHolidayModel.date <= end)
            .order_by(TurkishHolidayModel.date)
        )
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("date")
        return df

    async def get_all(self) -> pd.DataFrame:
        """Return all holidays ordered by date.

        Returns:
            DataFrame with date index and holiday columns.
        """
        stmt = select(TurkishHolidayModel).order_by(TurkishHolidayModel.date)
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("date")
        return df
