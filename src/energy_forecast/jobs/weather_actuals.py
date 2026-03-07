"""Fetch actual weather data from OpenMeteo Archive API (T-2 day).

Runs daily at 04:00 via lifespan scheduler or manually via CLI.
Idempotent: skips dates that already have actuals in the database.

Usage:
    # CLI
    uv run python -m energy_forecast.jobs.weather_actuals

    # Makefile
    make fetch-weather-actuals
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from loguru import logger

from energy_forecast.utils import TZ_ISTANBUL


async def fetch_and_store_actuals(
    session_factory: object,
    settings: object,
) -> int:
    """Fetch actual weather for T-2 and store in DB.

    OpenMeteo Archive API publishes actuals with 24-48h delay,
    so we fetch data from 2 days ago to ensure availability.

    Args:
        session_factory: Async session factory for DB access.
        settings: Application settings with OpenMeteo config.

    Returns:
        Number of rows inserted (0 if already exists or no DB).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from energy_forecast.config.settings import Settings
    from energy_forecast.data.openmeteo_client import OpenMeteoClient
    from energy_forecast.db.repositories.weather_repo import WeatherSnapshotRepository

    sf = session_factory
    if not isinstance(sf, async_sessionmaker):
        logger.warning("No valid session factory — skipping weather actuals")
        return 0

    cfg = settings
    if not isinstance(cfg, Settings):
        logger.warning("No valid settings — skipping weather actuals")
        return 0

    target_date = datetime.now(tz=TZ_ISTANBUL).date() - timedelta(days=2)
    target_dt = datetime(
        target_date.year, target_date.month, target_date.day,
        tzinfo=TZ_ISTANBUL,
    )

    # Idempotent check
    async with sf() as session:
        repo = WeatherSnapshotRepository(session)
        if await repo.has_actuals_for_date(target_dt):
            logger.info(
                "Weather actuals for {} already exist, skipping",
                target_date,
            )
            return 0

    # Fetch from OpenMeteo Archive API
    logger.info("Fetching weather actuals for {}", target_date)
    end_date = target_date + timedelta(days=1)
    with OpenMeteoClient(
        config=cfg.openmeteo,
        region=cfg.region,
        timezone=cfg.project.timezone,
    ) as client:
        actuals_df = client.fetch_historical(
            start_date=str(target_date),
            end_date=str(end_date),
        )

    if actuals_df.empty:
        logger.warning("No weather actuals returned for {}", target_date)
        return 0

    # Store in DB
    fetched_at = datetime.now(tz=TZ_ISTANBUL)
    async with sf() as session:
        repo = WeatherSnapshotRepository(session)
        count = await repo.bulk_create_actuals(
            weather_df=actuals_df,
            fetched_at=fetched_at,
        )
        await session.commit()

    logger.info(
        "Stored {} weather actual rows for {}", count, target_date
    )
    return count


async def run_scheduler(
    session_factory: object,
    settings: object,
    run_hour: int = 4,
) -> None:
    """Background scheduler — runs fetch_and_store_actuals daily.

    Checks every hour, executes when current hour matches run_hour.
    Designed to run as an asyncio task in the FastAPI lifespan.

    Args:
        session_factory: Async session factory.
        settings: Application settings.
        run_hour: Hour of day (0-23) to run the fetch. Default 4 (04:00).
    """
    last_run_date: datetime | None = None

    while True:
        await asyncio.sleep(3600)  # Check every hour
        now = datetime.now(tz=TZ_ISTANBUL)

        if now.hour != run_hour:
            continue

        # Only run once per day
        if last_run_date and last_run_date.date() == now.date():
            continue

        try:
            count = await fetch_and_store_actuals(session_factory, settings)
            last_run_date = now
            if count > 0:
                logger.info(
                    "Scheduler: stored {} weather actuals", count
                )
        except Exception as e:
            logger.error("Scheduler: weather actuals fetch failed: {}", e)


def main() -> None:
    """CLI entry point for manual execution."""
    from energy_forecast.config.settings import load_config
    from energy_forecast.db import create_db_engine, create_session_factory

    settings = load_config()
    if not settings.env.database_url:
        logger.error("DATABASE_URL not set — cannot store weather actuals")
        return

    engine = create_db_engine(settings.env.database_url, settings.database)
    sf = create_session_factory(engine)

    async def _run() -> None:
        try:
            count = await fetch_and_store_actuals(sf, settings)
            logger.info("Done — {} rows inserted", count)
        finally:
            await engine.dispose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
