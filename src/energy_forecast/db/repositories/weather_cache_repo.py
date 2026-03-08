"""Weather cache repository — upsert and query structured weather observations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import WeatherCacheModel

_WEATHER_UPDATE_COLS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "weather_code",
    "fetched_at",
]


class WeatherCacheRepository:
    """Async data access layer for weather_cache table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert weather rows. ON CONFLICT on (datetime, city, source) DO UPDATE.

        Args:
            rows: List of dicts with keys: datetime, city, source + weather columns.

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0
        stmt = pg_insert(WeatherCacheModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime", "city", "source"],
            set_={col: stmt.excluded[col] for col in _WEATHER_UPDATE_COLS},
        )
        await self._session.execute(stmt)
        return len(rows)

    async def get_range(
        self,
        start: datetime,
        end: datetime,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Get weather data in datetime range, optionally filtered by source.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            source: Optional source filter ('historical' or 'forecast').

        Returns:
            DataFrame with columns: datetime, city, source + weather variables.
        """
        stmt = (
            select(WeatherCacheModel)
            .where(WeatherCacheModel.dt >= start)
            .where(WeatherCacheModel.dt <= end)
        )
        if source is not None:
            stmt = stmt.where(WeatherCacheModel.source == source)
        stmt = stmt.order_by(WeatherCacheModel.dt, WeatherCacheModel.city)
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    async def get_by_city(
        self, city: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get weather data for a specific city in datetime range.

        Args:
            city: City name (e.g. 'Bursa', 'Balıkesir').
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            DataFrame ordered by datetime.
        """
        stmt = (
            select(WeatherCacheModel)
            .where(WeatherCacheModel.city == city)
            .where(WeatherCacheModel.dt >= start)
            .where(WeatherCacheModel.dt <= end)
            .order_by(WeatherCacheModel.dt)
        )
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    async def delete_stale(self, older_than: datetime) -> int:
        """Delete rows where fetched_at < older_than (TTL cleanup).

        Args:
            older_than: Cutoff datetime — rows older than this are deleted.

        Returns:
            Number of rows deleted.
        """
        stmt = delete(WeatherCacheModel).where(
            WeatherCacheModel.fetched_at < older_than
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[attr-defined, no-any-return]
