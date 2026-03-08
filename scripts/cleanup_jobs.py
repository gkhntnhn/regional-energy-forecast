"""Apply data retention policy — archive old jobs, delete old predictions.

Usage:
    uv run python scripts/cleanup_jobs.py --days 90
    uv run python scripts/cleanup_jobs.py --days 90 --dry-run

Retention rules:
- predictions older than N days → DELETE
- weather_snapshots (non-actual) older than N days → DELETE
- weather actuals (is_actual=True) → NEVER deleted (long-term analysis)
- jobs older than N days → status set to "archived" (metadata kept)
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import delete, func, select, update

from energy_forecast.utils import TZ_ISTANBUL


async def cleanup_old_data(
    session_factory: object,
    retention_days: int = 90,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Apply retention policy.

    Args:
        session_factory: Async session factory.
        retention_days: Number of days to retain.
        dry_run: If True, only report what would be deleted.

    Returns:
        Dict with counts of affected rows.
    """
    from energy_forecast.db.models import (
        JobModel,
        PredictionModel,
        WeatherSnapshotModel,
    )

    cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=retention_days)

    async with session_factory() as session:  # type: ignore[operator]
        # Count predictions to delete
        pred_count_stmt = (
            select(func.count())
            .select_from(PredictionModel)
            .where(PredictionModel.created_at < cutoff)
        )
        pred_count = (await session.execute(pred_count_stmt)).scalar() or 0

        # Count weather snapshots (non-actual) to delete
        weather_count_stmt = (
            select(func.count())
            .select_from(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(False),
                WeatherSnapshotModel.fetched_at < cutoff,
            )
        )
        weather_count = (
            await session.execute(weather_count_stmt)
        ).scalar() or 0

        # Count jobs to archive
        job_count_stmt = (
            select(func.count())
            .select_from(JobModel)
            .where(
                JobModel.created_at < cutoff,
                JobModel.status != "archived",
            )
        )
        job_count = (await session.execute(job_count_stmt)).scalar() or 0

        if dry_run:
            logger.info(
                "[DRY RUN] Would delete {} predictions, "
                "{} weather snapshots, archive {} jobs "
                "(cutoff: {})",
                pred_count,
                weather_count,
                job_count,
                cutoff.isoformat(),
            )
            return {
                "predictions_deleted": pred_count,
                "weather_deleted": weather_count,
                "jobs_archived": job_count,
                "dry_run": 1,
            }

        # Execute deletions
        pred_result = await session.execute(
            delete(PredictionModel).where(
                PredictionModel.created_at < cutoff
            )
        )
        weather_result = await session.execute(
            delete(WeatherSnapshotModel).where(
                WeatherSnapshotModel.is_actual.is_(False),
                WeatherSnapshotModel.fetched_at < cutoff,
            )
        )
        job_result = await session.execute(
            update(JobModel)
            .where(
                JobModel.created_at < cutoff,
                JobModel.status != "archived",
            )
            .values(status="archived")
        )

        await session.commit()

        result = {
            "predictions_deleted": pred_result.rowcount,
            "weather_deleted": weather_result.rowcount,
            "jobs_archived": job_result.rowcount,
        }
        logger.info("Cleanup completed: {}", result)
        return result


async def main(days: int, dry_run: bool) -> None:
    """Entry point."""
    from dotenv import load_dotenv

    load_dotenv()

    from energy_forecast.config import load_config
    from energy_forecast.db import create_db_engine, create_session_factory

    settings = load_config()
    if not settings.env.database_url:
        logger.error("DATABASE_URL not set — cannot run cleanup")
        return

    engine = create_db_engine(settings.env.database_url, settings.database)
    session_factory = create_session_factory(engine)

    try:
        result = await cleanup_old_data(
            session_factory, retention_days=days, dry_run=dry_run
        )
        if dry_run:
            logger.info("[DRY RUN] Would affect: {}", result)
        else:
            logger.info("Cleanup result: {}", result)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply data retention policy")
    parser.add_argument(
        "--days", type=int, default=90, help="Retention period in days"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without executing",
    )
    args = parser.parse_args()
    asyncio.run(main(args.days, args.dry_run))
