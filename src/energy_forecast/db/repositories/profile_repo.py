"""Profile coefficient repository — upsert and query EPİAŞ profile data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import ProfileCoefficientModel
from energy_forecast.db.upsert_columns import PROFILE_UPDATE_COLS as _PROFILE_UPDATE_COLS


class ProfileRepository:
    """Async data access layer for profile_coefficients table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert profile coefficient rows. ON CONFLICT on datetime DO UPDATE.

        Args:
            rows: List of dicts with datetime + 14 profile columns.

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0
        stmt = pg_insert(ProfileCoefficientModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime"],
            set_={col: stmt.excluded[col] for col in _PROFILE_UPDATE_COLS},
        )
        await self._session.execute(stmt)
        return len(rows)

    async def get_by_year(self, year: int) -> pd.DataFrame:
        """Get profile coefficients for a specific year.

        Uses EXTRACT(YEAR FROM datetime) since year column was removed.

        Args:
            year: Calendar year (e.g. 2024).

        Returns:
            DataFrame with datetime index and profile columns.
        """
        from sqlalchemy import extract

        stmt = (
            select(ProfileCoefficientModel)
            .where(extract("year", ProfileCoefficientModel.dt) == year)
            .order_by(ProfileCoefficientModel.dt)
        )
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("datetime")
        return df

    async def get_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Get profile coefficients in a datetime range.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            DataFrame with datetime index and profile columns.
        """
        stmt = (
            select(ProfileCoefficientModel)
            .where(ProfileCoefficientModel.dt >= start)
            .where(ProfileCoefficientModel.dt <= end)
            .order_by(ProfileCoefficientModel.dt)
        )
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("datetime")
        return df
