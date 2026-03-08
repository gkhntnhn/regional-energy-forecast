"""EPIAS repository — upsert and query for market and generation data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import EpiasGenerationModel, EpiasMarketModel

_MARKET_UPDATE_COLS = ["fdpp", "rtc", "dam_purchase", "bilateral", "load_forecast", "fetched_at"]
_GENERATION_UPDATE_COLS = [
    "gen_asphaltite_coal",
    "gen_biomass",
    "gen_black_coal",
    "gen_dammed_hydro",
    "gen_fueloil",
    "gen_geothermal",
    "gen_import_coal",
    "gen_import_export",
    "gen_lignite",
    "gen_lng",
    "gen_naphta",
    "gen_natural_gas",
    "gen_river",
    "gen_sun",
    "gen_total",
    "gen_wasteheat",
    "gen_wind",
    "fetched_at",
]


class EpiasRepository:
    """Async data access layer for EPIAS market and generation tables."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_market(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert EPIAS market rows. ON CONFLICT on datetime DO UPDATE.

        Args:
            rows: List of dicts with keys: datetime, fdpp, rtc, dam_purchase,
                  bilateral, load_forecast. fetched_at is optional (defaults to now).

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0
        stmt = pg_insert(EpiasMarketModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime"],
            set_={col: stmt.excluded[col] for col in _MARKET_UPDATE_COLS},
        )
        await self._session.execute(stmt)
        return len(rows)

    async def upsert_generation(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert EPIAS generation rows. ON CONFLICT on datetime DO UPDATE.

        Args:
            rows: List of dicts with datetime + 17 gen_ fuel type columns.

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0
        stmt = pg_insert(EpiasGenerationModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime"],
            set_={col: stmt.excluded[col] for col in _GENERATION_UPDATE_COLS},
        )
        await self._session.execute(stmt)
        return len(rows)

    async def get_market_range(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get market data between start and end (inclusive) as DataFrame.

        Args:
            start: Start datetime (UTC or timezone-aware).
            end: End datetime (UTC or timezone-aware).

        Returns:
            DataFrame with datetime index and market columns.
        """
        stmt = (
            select(EpiasMarketModel)
            .where(EpiasMarketModel.dt >= start)
            .where(EpiasMarketModel.dt <= end)
            .order_by(EpiasMarketModel.dt)
        )
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("datetime")
        return df

    async def get_generation_range(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get generation data between start and end (inclusive) as DataFrame.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            DataFrame with datetime index and gen_ columns.
        """
        stmt = (
            select(EpiasGenerationModel)
            .where(EpiasGenerationModel.dt >= start)
            .where(EpiasGenerationModel.dt <= end)
            .order_by(EpiasGenerationModel.dt)
        )
        result = await self._session.execute(stmt)
        rows = [dict(r._mapping) for r in result.mappings().all()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("datetime")
        return df

    async def get_latest_market_datetime(self) -> datetime | None:
        """Return the most recent market data timestamp.

        Returns:
            Latest datetime or None if table is empty.
        """
        stmt = select(EpiasMarketModel.dt).order_by(
            EpiasMarketModel.dt.desc()
        ).limit(1)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return row
