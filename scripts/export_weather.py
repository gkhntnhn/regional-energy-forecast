"""Export weather_cache table to yearly parquet files.

Creates parquet files in data/external/weather/ directory, one per year.
Only exports source='historical' rows (forecast is ephemeral).

Usage:
    python scripts/export_weather.py                   # Export all years
    python scripts/export_weather.py --year 2024       # Export single year
    python scripts/export_weather.py --dry-run          # Show plan only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export weather_cache DB table to yearly parquet files.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Export only this year (default: all years in DB).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )

    db_url = os.getenv("DATABASE_URL_SYNC")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set.")
        return 1

    from sqlalchemy import extract, func, select

    from energy_forecast.config import load_config
    from energy_forecast.db.engine import create_sync_engine, create_sync_session_factory
    from energy_forecast.db.models import WeatherCacheModel

    configs_dir = PROJECT_ROOT / "configs"
    settings = load_config(configs_dir)

    parquet_dir = Path(settings.openmeteo.cache.parquet_dir)
    file_pattern = settings.openmeteo.cache.file_pattern

    engine = create_sync_engine(db_url)
    factory = create_sync_session_factory(engine)
    session = factory()

    start_time = time.monotonic()

    try:
        # Determine which years to export
        if args.year:
            years = [args.year]
        else:
            stmt = (
                select(
                    extract("year", WeatherCacheModel.dt).label("yr"),
                )
                .where(WeatherCacheModel.source == "historical")
                .group_by("yr")
                .order_by("yr")
            )
            result = session.execute(stmt)
            years = [int(row.yr) for row in result]

        if not years:
            logger.warning("No historical weather data found in DB.")
            return 0

        logger.info("Years to export: {}", years)

        if args.dry_run:
            for yr in years:
                count_stmt = (
                    select(func.count())
                    .select_from(WeatherCacheModel)
                    .where(WeatherCacheModel.source == "historical")
                    .where(extract("year", WeatherCacheModel.dt) == yr)
                )
                count = session.execute(count_stmt).scalar() or 0
                path = parquet_dir / file_pattern.format(year=yr)
                logger.info("  {} -> {} ({} rows)", yr, path, count)
            logger.info("Dry run -- no files written.")
            return 0

        # Export each year
        parquet_dir.mkdir(parents=True, exist_ok=True)
        total_rows = 0

        for yr in years:
            stmt = (
                select(WeatherCacheModel)
                .where(WeatherCacheModel.source == "historical")
                .where(extract("year", WeatherCacheModel.dt) == yr)
                .order_by(WeatherCacheModel.dt, WeatherCacheModel.city)
            )
            result = session.execute(stmt)
            instances = result.scalars().all()

            if not instances:
                logger.warning("  {} -> 0 rows, skipping", yr)
                continue

            # Convert ORM instances to dicts
            rows: list[dict[str, object]] = []
            for inst in instances:
                d = {
                    k: v
                    for k, v in inst.__dict__.items()
                    if not k.startswith("_")
                }
                # Rename ORM 'dt' back to 'datetime'
                if "dt" in d:
                    d["datetime"] = d.pop("dt")
                # Drop fetched_at — internal metadata
                d.pop("fetched_at", None)
                rows.append(d)

            df = pd.DataFrame(rows)
            # Normalize datetime to tz-naive for parquet compatibility
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)

            path = parquet_dir / file_pattern.format(year=yr)
            df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
            total_rows += len(df)
            logger.info("  {} -> {} ({} rows)", yr, path, len(df))

    except Exception as e:
        logger.error("Export failed: {}", e)
        return 1
    finally:
        session.close()

    elapsed = time.monotonic() - start_time
    logger.info("Export complete: {} rows across {} years in {:.1f}s",
                total_rows, len(years), elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
