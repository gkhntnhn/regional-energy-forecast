"""Seed weather_cache table from OpenMeteo historical API.

Fetches per-city hourly weather data and writes to weather_cache table.
Uses monthly chunks to stay within OpenMeteo API limits.

Usage:
    python scripts/seed_weather.py                            # Last 1 year
    python scripts/seed_weather.py --start 2020-01-01         # From date to today
    python scripts/seed_weather.py --start 2024-01-01 --end 2024-12-31
    python scripts/seed_weather.py --dry-run                  # Show plan, don't write
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed weather_cache table from OpenMeteo historical API.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 1 year ago).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: yesterday).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without writing to DB.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between API requests (default: 2.0).",
    )
    return parser.parse_args()


def generate_monthly_chunks(
    start_date: str, end_date: str,
) -> list[tuple[str, str]]:
    """Generate monthly (start, end) date pairs."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []

    current = start
    while current <= end:
        month_end = (current + pd.offsets.MonthEnd(1))
        chunk_end = min(month_end, end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        current = chunk_end + timedelta(days=1)
    return chunks


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )

    # Default date range
    if args.start is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    else:
        start_date = args.start

    if args.end is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = args.end

    logger.info("Weather seed: {} to {}", start_date, end_date)

    # Generate monthly chunks
    chunks = generate_monthly_chunks(start_date, end_date)
    logger.info("Plan: {} monthly chunks", len(chunks))

    if args.dry_run:
        for i, (cs, ce) in enumerate(chunks, 1):
            logger.info("  Chunk {}: {} to {}", i, cs, ce)
        logger.info("Dry run — no DB writes. {} chunks x 4 cities = {} API calls",
                     len(chunks), len(chunks) * 4)
        return 0

    # Connect to DB
    db_url = os.getenv("DATABASE_URL_SYNC")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set. Cannot seed without database.")
        return 1

    from energy_forecast.config import load_config
    from energy_forecast.data.openmeteo_client import OpenMeteoClient
    from energy_forecast.db.engine import create_sync_engine, create_sync_session_factory
    from energy_forecast.db.sync_repos import SyncDataAccess

    configs_dir = PROJECT_ROOT / "configs"
    settings = load_config(configs_dir)

    engine = create_sync_engine(db_url)
    factory = create_sync_session_factory(engine)
    session = factory()
    dao = SyncDataAccess(session)

    total_rows = 0
    start_time = time.monotonic()

    try:
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            logger.info("[{}/{}] Fetching {} to {}", i, len(chunks), chunk_start, chunk_end)

            # Fetch per-city data from API (no DB session — raw API fetch)
            with OpenMeteoClient(
                config=settings.openmeteo,
                region=settings.region,
                timezone=settings.project.timezone,
            ) as client:
                city_dfs = client._fetch_all_cities(
                    url=settings.openmeteo.api.base_url_historical,
                    start_date=chunk_start,
                    end_date=chunk_end,
                )

            # Write per-city data to DB
            chunk_rows = 0
            for city, df in city_dfs:
                rows: list[dict[str, object]] = []
                for idx, row in df.iterrows():
                    d: dict[str, object] = {
                        "datetime": idx,
                        "city": city.name,
                        "source": "historical",
                    }
                    for col in df.columns:
                        val = row[col]
                        d[col] = float(val) if pd.notna(val) else None
                    rows.append(d)
                count = dao.upsert_weather(rows)
                chunk_rows += count

            session.commit()
            total_rows += chunk_rows
            logger.info("  Written {} rows ({} total)", chunk_rows, total_rows)

            # Rate limit
            if i < len(chunks):
                time.sleep(args.delay)

    except Exception as e:
        logger.error("Seed failed: {}", e)
        session.rollback()
        return 1
    finally:
        session.close()

    elapsed = time.monotonic() - start_time
    logger.info("Weather seed complete: {} rows in {:.1f}s", total_rows, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
