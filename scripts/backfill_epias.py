"""Backfill EPIAS year-based cache.

Fetches historical EPIAS market data and saves as yearly parquet files.
Skips years that are already cached.

Usage::

    python scripts/backfill_epias.py [--start-year 2020] [--end-year 2025]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from energy_forecast.config.settings import EnvConfig, EpiasApiConfig
from energy_forecast.data.epias_client import EpiasClient
from energy_forecast.data.exceptions import EpiasApiError


def main(start_year: int = 2020, end_year: int | None = None) -> None:
    """Backfill missing EPIAS cache years.

    Args:
        start_year: First year to backfill.
        end_year: Last year to backfill (default: current year).
    """
    if end_year is None:
        end_year = datetime.now().year

    env = EnvConfig()
    if not env.epias_username or not env.epias_password:
        logger.error("EPIAS_USERNAME and EPIAS_PASSWORD must be set in .env")
        sys.exit(1)

    cache_dir = Path("data/external/epias")
    logger.info(
        "Backfilling EPIAS cache: {} to {} → {}",
        start_year,
        end_year,
        cache_dir,
    )

    success = 0
    skipped = 0
    failed = 0

    epias_config = EpiasApiConfig(cache_dir=str(cache_dir))
    with EpiasClient(
        username=env.epias_username,
        password=env.epias_password,
        config=epias_config,
    ) as client:
        for year in range(start_year, end_year + 1):
            cached = client.load_cache(year)
            if cached is not None:
                logger.info("Year {} already cached ({} rows)", year, len(cached))
                skipped += 1
                continue

            try:
                df = client.fetch_year(year)
                logger.info("Fetched year {}: {} rows", year, len(df))
                success += 1
            except EpiasApiError as exc:
                logger.error("Failed to fetch year {}: {}", year, exc)
                failed += 1

    logger.info(
        "Backfill complete: {} success, {} skipped, {} failed",
        success,
        skipped,
        failed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill EPIAS cache")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
