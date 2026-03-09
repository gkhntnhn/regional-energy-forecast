"""Sync data access layer for CLI scripts and sync clients.

Uses sync SQLAlchemy Session with PostgreSQL dialect (pg_insert).
Mirrors async repository patterns but for sync contexts.

Usage:
    from energy_forecast.db.sync_repos import SyncDataAccess
    from energy_forecast.db.engine import create_sync_engine, create_sync_session_factory

    engine = create_sync_engine(db_url)
    factory = create_sync_session_factory(engine)
    with factory() as session:
        dao = SyncDataAccess(session)
        df = dao.get_epias_market_range(start, end)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import extract, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from energy_forecast.db.models import (
    EpiasGenerationModel,
    EpiasMarketModel,
    ProfileCoefficientModel,
    TurkishHolidayModel,
    WeatherCacheModel,
)

# Column lists for ON CONFLICT DO UPDATE (mirrors async repos)
_MARKET_UPDATE_COLS = [
    "fdpp", "rtc", "dam_purchase", "bilateral", "load_forecast", "fetched_at",
]
_GENERATION_UPDATE_COLS = [
    "gen_asphaltite_coal", "gen_biomass", "gen_black_coal", "gen_dammed_hydro",
    "gen_fueloil", "gen_geothermal", "gen_import_coal", "gen_import_export",
    "gen_lignite", "gen_lng", "gen_naphta", "gen_natural_gas", "gen_river",
    "gen_sun", "gen_total", "gen_wasteheat", "gen_wind", "fetched_at",
]
_WEATHER_UPDATE_COLS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dew_point_2m", "precipitation", "snow_depth", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "shortwave_radiation",
    "weather_code", "fetched_at",
]
_HOLIDAY_UPDATE_COLS = [
    "holiday_name", "raw_holiday_name", "is_ramadan",
    "bayram_gun_no", "bayrama_kalan_gun",
]
_PROFILE_UPDATE_COLS = [
    "profile_residential_lv", "profile_residential_mv",
    "profile_industrial_lv", "profile_industrial_mv",
    "profile_commercial_lv", "profile_commercial_mv",
    "profile_agricultural_irrigation_lv", "profile_agricultural_irrigation_mv",
    "profile_lighting", "profile_government",
    "profile_residential", "profile_industrial",
    "profile_commercial", "profile_agricultural_irrigation",
    "fetched_at",
]


def _rows_to_df(
    result: Any,
    index_col: str = "datetime",
) -> pd.DataFrame:
    """Convert SQLAlchemy ORM result to DataFrame with index.

    Handles ``select(Model)`` results — extracts column values from
    ORM instances via ``__dict__``, excluding SQLAlchemy internals.
    Renames ``dt`` attribute to ``datetime`` for consistency with
    downstream consumers.
    """
    instances = result.scalars().all()
    if not instances:
        return pd.DataFrame()
    rows = [
        {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        for obj in instances
    ]
    df = pd.DataFrame(rows)
    # ORM models use 'dt' attribute for 'datetime' column — rename back
    if "dt" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"dt": "datetime"})
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df


class SyncDataAccess:
    """Sync data access layer for CLI scripts and sync clients.

    Provides read/write methods for all external data tables using
    sync SQLAlchemy Session. Designed for use in EpiasClient,
    OpenMeteoClient, prepare_dataset.py, and PredictionService.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -----------------------------------------------------------------
    # EPIAS Market
    # -----------------------------------------------------------------

    def get_epias_market_range(
        self, start: datetime, end: datetime,
    ) -> pd.DataFrame:
        """Get market data between start and end (inclusive)."""
        stmt = (
            select(EpiasMarketModel)
            .where(EpiasMarketModel.dt >= start)
            .where(EpiasMarketModel.dt <= end)
            .order_by(EpiasMarketModel.dt)
        )
        result = self._session.execute(stmt)
        return _rows_to_df(result)

    def get_epias_market_year(self, year: int) -> pd.DataFrame:
        """Get market data for a specific year."""
        stmt = (
            select(EpiasMarketModel)
            .where(extract("year", EpiasMarketModel.dt) == year)
            .order_by(EpiasMarketModel.dt)
        )
        result = self._session.execute(stmt)
        return _rows_to_df(result)

    def upsert_epias_market(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert EPIAS market rows. Returns count."""
        if not rows:
            return 0
        stmt = pg_insert(EpiasMarketModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime"],
            set_={col: stmt.excluded[col] for col in _MARKET_UPDATE_COLS},
        )
        self._session.execute(stmt)
        self._session.flush()
        return len(rows)

    # -----------------------------------------------------------------
    # EPIAS Generation
    # -----------------------------------------------------------------

    def get_epias_generation_range(
        self, start: datetime, end: datetime,
    ) -> pd.DataFrame:
        """Get generation data between start and end (inclusive)."""
        stmt = (
            select(EpiasGenerationModel)
            .where(EpiasGenerationModel.dt >= start)
            .where(EpiasGenerationModel.dt <= end)
            .order_by(EpiasGenerationModel.dt)
        )
        result = self._session.execute(stmt)
        return _rows_to_df(result)

    def get_epias_generation_year(self, year: int) -> pd.DataFrame:
        """Get generation data for a specific year."""
        stmt = (
            select(EpiasGenerationModel)
            .where(extract("year", EpiasGenerationModel.dt) == year)
            .order_by(EpiasGenerationModel.dt)
        )
        result = self._session.execute(stmt)
        return _rows_to_df(result)

    def upsert_epias_generation(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert EPIAS generation rows. Returns count."""
        if not rows:
            return 0
        stmt = pg_insert(EpiasGenerationModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime"],
            set_={col: stmt.excluded[col] for col in _GENERATION_UPDATE_COLS},
        )
        self._session.execute(stmt)
        self._session.flush()
        return len(rows)

    # -----------------------------------------------------------------
    # Weather Cache
    # -----------------------------------------------------------------

    def get_weather_range(
        self,
        start: datetime,
        end: datetime,
        city: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Get weather data in range, optionally filtered by city/source."""
        stmt = (
            select(WeatherCacheModel)
            .where(WeatherCacheModel.dt >= start)
            .where(WeatherCacheModel.dt <= end)
        )
        if city is not None:
            stmt = stmt.where(WeatherCacheModel.city == city)
        if source is not None:
            stmt = stmt.where(WeatherCacheModel.source == source)
        stmt = stmt.order_by(WeatherCacheModel.dt, WeatherCacheModel.city)
        result = self._session.execute(stmt)
        return _rows_to_df(result)

    def upsert_weather(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert weather rows. Returns count."""
        if not rows:
            return 0
        stmt = pg_insert(WeatherCacheModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["datetime", "city", "source"],
            set_={col: stmt.excluded[col] for col in _WEATHER_UPDATE_COLS},
        )
        self._session.execute(stmt)
        self._session.flush()
        return len(rows)

    # -----------------------------------------------------------------
    # Holidays
    # -----------------------------------------------------------------

    def get_holidays(self) -> pd.DataFrame:
        """Return all holidays ordered by date."""
        stmt = select(TurkishHolidayModel).order_by(TurkishHolidayModel.date)
        result = self._session.execute(stmt)
        return _rows_to_df(result, index_col="date")

    def get_holidays_range(self, start: date, end: date) -> pd.DataFrame:
        """Get holidays between start and end dates (inclusive)."""
        stmt = (
            select(TurkishHolidayModel)
            .where(TurkishHolidayModel.date >= start)
            .where(TurkishHolidayModel.date <= end)
            .order_by(TurkishHolidayModel.date)
        )
        result = self._session.execute(stmt)
        return _rows_to_df(result, index_col="date")

    # -----------------------------------------------------------------
    # Profile Coefficients
    # -----------------------------------------------------------------

    def get_profile_range(
        self, start: datetime, end: datetime,
    ) -> pd.DataFrame:
        """Get profile coefficients in a datetime range."""
        stmt = (
            select(ProfileCoefficientModel)
            .where(ProfileCoefficientModel.dt >= start)
            .where(ProfileCoefficientModel.dt <= end)
            .order_by(ProfileCoefficientModel.dt)
        )
        result = self._session.execute(stmt)
        return _rows_to_df(result)
