"""Seed PostgreSQL with external data from parquet files.

Usage::

    python scripts/seed_db.py           # sample data (data/seed/)
    python scripts/seed_db.py --full    # all parquet data (full import)

Imports four data sources:
  - EPIAS market (epias_market_{year}.parquet)
  - EPIAS generation (epias_generation_{year}.parquet)
  - Turkish holidays (turkish_holidays.parquet)
  - Profile coefficients (profile_coef_{year}.parquet)

Weather cache is intentionally excluded: the existing weather_cache.sqlite is an
opaque requests-cache binary. Weather data will be populated in Phase 2 when the
OpenMeteo client is refactored to write structured observations directly to DB.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

# Load .env before importing project config (DATABASE_URL_SYNC etc.)
load_dotenv()

from energy_forecast.config import EnvConfig  # noqa: E402
from energy_forecast.db.models import (  # noqa: E402
    EpiasGenerationModel,
    EpiasMarketModel,
    ProfileCoefficientModel,
    TurkishHolidayModel,
)

# ---------------------------------------------------------------------------
# Column name mappings: parquet column → DB column
# ---------------------------------------------------------------------------

# epias_market_{year}.parquet columns (from _VARIABLE_MAP keys in epias_client.py)
_MARKET_COL_MAP: dict[str, str] = {
    "FDPP": "fdpp",
    "Real_Time_Consumption": "rtc",
    "DAM_Purchase": "dam_purchase",
    "Bilateral_Agreement_Purchase": "bilateral",
    "Load_Forecast": "load_forecast",
}

# epias_generation columns are already in gen_* format — no remapping needed
_GENERATION_COLS = [
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
]

# profile_coef columns are already in profile_* format — no remapping needed
_PROFILE_COLS = [
    "profile_residential_lv",
    "profile_residential_mv",
    "profile_industrial_lv",
    "profile_industrial_mv",
    "profile_commercial_lv",
    "profile_commercial_mv",
    "profile_agricultural_irrigation_lv",
    "profile_agricultural_irrigation_mv",
    "profile_lighting",
    "profile_government",
    "profile_residential",
    "profile_industrial",
    "profile_commercial",
    "profile_agricultural_irrigation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sync_session(database_url_sync: str) -> sessionmaker:  # type: ignore[type-arg]
    engine = create_engine(database_url_sync, pool_size=1, max_overflow=0, pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _to_utc_aware(dt_series: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    """Ensure datetime series is UTC-aware (required by asyncpg TIMESTAMPTZ)."""
    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize("UTC")
    return dt_series.dt.tz_convert("UTC")


def _upsert_batch(
    session: Any,
    model: Any,
    rows: list[dict[str, Any]],
    conflict_cols: list[str],
    update_cols: list[str],
    label: str,
) -> int:
    """Execute a bulk upsert and return row count."""
    if not rows:
        logger.warning("{}: no rows to insert (empty file?)", label)
        return 0
    stmt = pg_insert(model).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=conflict_cols,
        set_={col: stmt.excluded[col] for col in update_cols},
    )
    session.execute(stmt)
    session.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Seed functions
# ---------------------------------------------------------------------------


def seed_epias_market(session: Any, paths: list[Path]) -> int:
    """Import epias_market_{year}.parquet files into epias_market table."""
    total = 0
    for path in paths:
        try:
            df = pd.read_parquet(path)
            if "datetime" not in df.columns:
                logger.warning("{}: 'datetime' column missing, skipping", path.name)
                continue

            df["datetime"] = _to_utc_aware(pd.to_datetime(df["datetime"]))

            # Rename parquet columns to DB column names
            df = df.rename(columns=_MARKET_COL_MAP)

            # Keep only DB columns that exist in parquet (FDPP may be missing in old files)
            db_cols = list(_MARKET_COL_MAP.values())
            present_cols = [c for c in db_cols if c in df.columns]
            df = df[["datetime", *present_cols]]

            # Add fetched_at
            df["fetched_at"] = _now_utc()

            rows = df.to_dict(orient="records")
            update_cols = [*present_cols, "fetched_at"]
            n = _upsert_batch(
                session,
                EpiasMarketModel,
                rows,
                ["datetime"],
                update_cols,
                path.name,
            )
            total += n
            logger.info("{}: {} rows upserted", path.name, n)
        except Exception as exc:
            logger.warning("{}: skipped — {}", path.name, exc)
    return total


def seed_epias_generation(session: Any, paths: list[Path]) -> int:
    """Import epias_generation_{year}.parquet files into epias_generation table."""
    total = 0
    for path in paths:
        try:
            df = pd.read_parquet(path)
            if "datetime" not in df.columns:
                logger.warning("{}: 'datetime' column missing, skipping", path.name)
                continue

            df["datetime"] = _to_utc_aware(pd.to_datetime(df["datetime"]))

            present_cols = [c for c in _GENERATION_COLS if c in df.columns]
            df = df[["datetime", *present_cols]]
            df["fetched_at"] = _now_utc()

            rows = df.to_dict(orient="records")
            n = _upsert_batch(
                session,
                EpiasGenerationModel,
                rows,
                ["datetime"],
                [*present_cols, "fetched_at"],
                path.name,
            )
            total += n
            logger.info("{}: {} rows upserted", path.name, n)
        except Exception as exc:
            logger.warning("{}: skipped — {}", path.name, exc)
    return total


def seed_holidays(session: Any, path: Path) -> int:
    """Import turkish_holidays.parquet into turkish_holidays table."""
    try:
        df = pd.read_parquet(path)

        # Ensure 'date' is a Python date (not datetime)
        if "date" not in df.columns:
            logger.warning("{}: 'date' column missing, skipping", path.name)
            return 0

        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Align columns to DB schema
        col_map = {
            "holiday_name": None,
            "raw_holiday_name": None,
            "is_ramadan": 0,
            "bayram_gun_no": 0,
            "bayrama_kalan_gun": -1,
        }
        for col, default in col_map.items():
            if col not in df.columns:
                df[col] = default

        # Replace pandas NA with None for nullable string columns
        for str_col in ["holiday_name", "raw_holiday_name"]:
            if str_col in df.columns:
                df[str_col] = df[str_col].where(df[str_col].notna(), other=None)

        df = df[["date", *col_map.keys()]]
        rows = df.to_dict(orient="records")

        n = _upsert_batch(
            session,
            TurkishHolidayModel,
            rows,
            ["date"],
            list(col_map.keys()),
            path.name,
        )
        logger.info("{}: {} rows upserted", path.name, n)
        return n
    except Exception as exc:
        logger.warning("{}: skipped — {}", path.name, exc)
        return 0


def seed_profile_coefficients(session: Any, paths: list[Path]) -> int:
    """Import profile_coef_{year}.parquet files into profile_coefficients table."""
    total = 0
    for path in paths:
        try:
            df = pd.read_parquet(path)
            if "datetime" not in df.columns:
                logger.warning("{}: 'datetime' column missing, skipping", path.name)
                continue

            df["datetime"] = _to_utc_aware(pd.to_datetime(df["datetime"]))

            present_cols = [c for c in _PROFILE_COLS if c in df.columns]
            df = df[["datetime", *present_cols]]
            df["fetched_at"] = _now_utc()

            rows = df.to_dict(orient="records")
            n = _upsert_batch(
                session,
                ProfileCoefficientModel,
                rows,
                ["datetime"],
                [*present_cols, "fetched_at"],
                path.name,
            )
            total += n
            logger.info("{}: {} rows upserted", path.name, n)
        except Exception as exc:
            logger.warning("{}: skipped — {}", path.name, exc)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _discover_files(full: bool) -> dict[str, list[Path]]:
    """Discover parquet files to import based on mode."""
    if full:
        market_paths = sorted(Path("data/external/epias").glob("epias_market_*.parquet"))
        gen_paths = sorted(Path("data/external/epias").glob("epias_generation_*.parquet"))
        holiday_path = Path("data/static/turkish_holidays.parquet")
        profile_paths = sorted(Path("data/external/profile").glob("profile_coef_*.parquet"))
    else:
        # Sample mode: data/seed/ directory (for quick smoke tests)
        seed_dir = Path("data/seed")
        market_paths = sorted(seed_dir.glob("epias_market_*.parquet"))
        gen_paths = sorted(seed_dir.glob("epias_generation_*.parquet"))
        holiday_path = seed_dir / "turkish_holidays.parquet"
        profile_paths = sorted(seed_dir.glob("profile_coef_*.parquet"))

    return {
        "market": market_paths,
        "generation": gen_paths,
        "holidays": [holiday_path] if holiday_path.exists() else [],
        "profile": profile_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed PostgreSQL with external parquet data.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Import all parquet files from data/external/ + data/static/",
    )
    args = parser.parse_args()

    env = EnvConfig()
    if not env.database_url_sync:
        logger.error("DATABASE_URL_SYNC not set in environment — cannot seed DB.")
        sys.exit(1)

    logger.info("Connecting to DB (sync/psycopg2)...")
    session_factory = _make_sync_session(env.database_url_sync)

    files = _discover_files(args.full)
    mode = "full" if args.full else "sample"
    logger.info("Seed mode: {} — discovered files:", mode)
    for source, paths in files.items():
        logger.info("  {}: {} file(s)", source, len(paths))

    total_rows = 0
    with session_factory() as session:
        # Check DB connectivity
        session.execute(text("SELECT 1"))

        # EPIAS market
        if files["market"]:
            n = seed_epias_market(session, files["market"])
            logger.info("epias_market: total {} rows", n)
            total_rows += n
        else:
            logger.warning("epias_market: no parquet files found")

        # EPIAS generation
        if files["generation"]:
            n = seed_epias_generation(session, files["generation"])
            logger.info("epias_generation: total {} rows", n)
            total_rows += n
        else:
            logger.warning("epias_generation: no parquet files found")

        # Turkish holidays
        if files["holidays"]:
            n = seed_holidays(session, files["holidays"][0])
            logger.info("turkish_holidays: {} rows", n)
            total_rows += n
        else:
            logger.warning("turkish_holidays: parquet file not found")

        # Profile coefficients
        if files["profile"]:
            n = seed_profile_coefficients(session, files["profile"])
            logger.info("profile_coefficients: total {} rows", n)
            total_rows += n
        else:
            logger.warning("profile_coefficients: no parquet files found")

    logger.info(
        "Seed complete. Total rows upserted: {}. "
        "Note: weather_cache NOT seeded (Faz 2).",
        total_rows,
    )


if __name__ == "__main__":
    main()
