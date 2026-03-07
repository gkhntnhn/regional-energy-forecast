"""Job repository — CRUD operations for JobModel."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import JobModel
from energy_forecast.utils import TZ_ISTANBUL


class JobRepository:
    """Data access layer for jobs table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, job_data: dict[str, Any]) -> JobModel:
        """Insert a new job record."""
        job = JobModel(**job_data)
        self._session.add(job)
        await self._session.flush()
        return job

    async def get_by_id(self, job_id: str) -> JobModel | None:
        """Fetch a job by its ID."""
        return await self._session.get(JobModel, job_id)

    async def get_active_job(self) -> JobModel | None:
        """Return the currently active (pending/running) job, if any."""
        stmt = (
            select(JobModel)
            .where(JobModel.status.in_(["pending", "running"]))
            .order_by(JobModel.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self, job_id: str, status: str, **kwargs: Any
    ) -> None:
        """Update job status and optional fields."""
        job = await self.get_by_id(job_id)
        if job is None:
            return
        job.status = status
        if status == "completed":
            job.completed_at = datetime.now(tz=TZ_ISTANBUL)
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        await self._session.flush()

    async def update_progress(self, job_id: str, progress: str) -> None:
        """Update job progress text."""
        job = await self.get_by_id(job_id)
        if job is None:
            return
        job.progress = progress
        await self._session.flush()

    async def update_metadata(
        self, job_id: str, metadata: dict[str, Any]
    ) -> None:
        """Update job lineage metadata (JSONB fields)."""
        job = await self.get_by_id(job_id)
        if job is None:
            return
        if "metadata" in metadata:
            job.metadata_ = metadata["metadata"]
        if "feature_importance_top15" in metadata:
            existing = job.metadata_ or {}
            existing["feature_importance_top15"] = metadata[
                "feature_importance_top15"
            ]
            job.metadata_ = existing
        if "config_snapshot" in metadata:
            job.config_snapshot = metadata["config_snapshot"]
        if "model_versions" in metadata:
            job.model_versions = metadata["model_versions"]
        if "epias_snapshot" in metadata:
            job.epias_snapshot = metadata["epias_snapshot"]
        if "excel_hash" in metadata:
            job.excel_hash = metadata["excel_hash"]
        if "historical_path" in metadata:
            job.historical_path = metadata["historical_path"]
        if "forecast_path" in metadata:
            job.forecast_path = metadata["forecast_path"]
        if "archive_path" in metadata:
            job.archive_path = metadata["archive_path"]
        await self._session.flush()

    async def update_email_status(
        self,
        job_id: str,
        status: str,
        attempts: int = 0,
        error: str | None = None,
        sent_at: datetime | None = None,
    ) -> None:
        """Update email delivery tracking fields."""
        job = await self.get_by_id(job_id)
        if job is None:
            return
        job.email_status = status
        job.email_attempts = attempts
        if error is not None:
            job.email_error = error
        if sent_at is not None:
            job.email_sent_at = sent_at
        await self._session.flush()

    async def get_all(self) -> list[JobModel]:
        """Return all jobs ordered by creation time (newest first)."""
        stmt = select(JobModel).order_by(JobModel.created_at.desc())
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_stats(self) -> dict[str, int]:
        """Return job count by status."""
        jobs = await self.get_all()
        stats: dict[str, int] = {}
        for job in jobs:
            stats[job.status] = stats.get(job.status, 0) + 1
        return stats
