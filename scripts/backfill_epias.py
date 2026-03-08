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

from energy_forecast.config import EnvConfig, EpiasApiConfig
from energy_forecast.utils import TZ_ISTANBUL
from energy_forecast.data.epias_client import EpiasClient
from energy_forecast.data.exceptions import EpiasApiError


def main(
    start_year: int = 2020,
    end_year: int | None = None,
    *,
    generation: bool = False,
) -> None:
    """Backfill missing EPIAS cache years.

    Args:
        start_year: First year to backfill.
        end_year: Last year to backfill (default: current year).
        generation: Also backfill generation data cache.
    """
    if end_year is None:
        end_year = datetime.now(tz=TZ_ISTANBUL).year

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
        # Market data backfill
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
            "Market backfill complete: {} success, {} skipped, {} failed",
            success,
            skipped,
            failed,
        )

        # Generation data backfill
        if generation:
            gen_pattern = epias_config.generation_file_pattern
            gen_success = 0
            gen_skipped = 0
            gen_failed = 0

            logger.info("Backfilling generation cache...")
            for year in range(start_year, end_year + 1):
                cached = client.load_cache(year, file_pattern=gen_pattern)
                if cached is not None:
                    logger.info(
                        "Generation year {} already cached ({} rows)", year, len(cached)
                    )
                    gen_skipped += 1
                    continue

                try:
                    df = client.fetch_generation_year(year)
                    logger.info("Fetched generation year {}: {} rows", year, len(df))
                    gen_success += 1
                except EpiasApiError as exc:
                    logger.error("Failed to fetch generation year {}: {}", year, exc)
                    gen_failed += 1

            logger.info(
                "Generation backfill complete: {} success, {} skipped, {} failed",
                gen_success,
                gen_skipped,
                gen_failed,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill EPIAS cache")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Also backfill generation data cache.",
    )
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year, generation=args.generation)
