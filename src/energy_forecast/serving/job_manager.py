"""Job queue manager with single-worker guarantee.

Supports two modes:
- **Database mode** (DATABASE_URL set): Jobs persist in PostgreSQL via repositories.
- **In-memory mode** (DATABASE_URL empty): Jobs stored in a dict (dev/test only).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.serving.exceptions import JobNotFoundError, JobQueueFullError
from energy_forecast.serving.job_steps import (
    archive_step,
    create_output_step,
    match_previous_predictions_step,
    run_drift_check,
    run_prediction_step,
    send_email_step,
    store_metadata_step,
    store_predictions_step,
    store_weather_step,
    update_progress_db,
    update_status_db,
)
from energy_forecast.serving.schemas import JobStatus
from energy_forecast.utils import TZ_ISTANBUL

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from energy_forecast.serving.services.email_service import EmailService
    from energy_forecast.serving.services.file_service import FileService
    from energy_forecast.serving.services.prediction_service import PredictionService

# Re-export for backward compatibility (tests import from here)
_run_drift_check = run_drift_check


class Job(BaseModel):
    """In-memory job representation (used when DATABASE_URL is empty)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    email: str
    excel_path: Path
    file_stem: str
    status: JobStatus = JobStatus.PENDING
    progress: str | None = None
    error: str | None = None
    result_path: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=TZ_ISTANBUL))
    completed_at: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}


@dataclass(slots=True)
class _QueueItem:
    """Internal queue item for job processing."""

    job_id: str
    excel_path: str
    email: str
    file_stem: str
    created_at: datetime
    is_db_mode: bool
    prediction_service: PredictionService
    file_service: FileService
    email_service: EmailService
    session_factory: async_sessionmaker[AsyncSession] | None = field(default=None)
    job_ref: Job | None = field(default=None)


class JobManager:
    """Manages prediction jobs with queue-based sequential processing.

    Uses asyncio.Queue to accept multiple jobs and processes them one at a time
    via a background worker loop. Replaces the old Lock + reject pattern.
    """

    _MAX_WORKER_RESTARTS = 3

    def __init__(self, max_queue_size: int = 5) -> None:
        self._jobs: dict[str, Job] = {}
        self._active_job_id: str | None = None
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._enqueue_lock = asyncio.Lock()
        self.enqueue_lock = self._enqueue_lock  # Public alias for atomic create+enqueue
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_restart_count = 0

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def start_worker(self) -> None:
        """Start the background worker loop. Call from lifespan startup."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._worker_task.add_done_callback(self._on_worker_done)
        logger.info(
            "Job queue worker started (max_queue_size={})",
            self._max_queue_size,
        )

    def _on_worker_done(self, task: asyncio.Task[None]) -> None:
        """Auto-restart worker if it crashes unexpectedly (max 3 restarts)."""
        if task.cancelled():
            return  # graceful shutdown via stop_worker()
        exc = task.exception()
        if exc is not None:
            self._worker_restart_count += 1
            if self._worker_restart_count <= self._MAX_WORKER_RESTARTS:
                logger.error(
                    "Worker loop crashed (restart {}/{}): {}",
                    self._worker_restart_count,
                    self._MAX_WORKER_RESTARTS,
                    exc,
                )
                self._worker_task = None
                self.start_worker()
            else:
                logger.critical(
                    "Worker loop crashed {} times — NOT restarting. Manual intervention required.",
                    self._worker_restart_count,
                )

    async def stop_worker(self, drain_timeout: float = 30.0) -> None:
        """Stop worker with graceful drain -- waits for current job to finish.

        Args:
            drain_timeout: Max seconds to wait for queue drain before cancelling.
        """
        if self._worker_task is None:
            return
        # Wait for current job to finish (don't accept new ones)
        if not self._queue.empty():
            logger.info(
                "Draining queue ({} items, timeout={}s)...",
                self._queue.qsize(),
                drain_timeout,
            )
            try:
                await asyncio.wait_for(self._queue.join(), timeout=drain_timeout)
            except TimeoutError:
                logger.warning(
                    "Queue drain timeout after {}s -- cancelling worker",
                    drain_timeout,
                )
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None
        logger.info("Job queue worker stopped (remaining: {})", self._queue.qsize())

    async def _worker_loop(self) -> None:
        """Background loop: dequeue jobs and process them sequentially."""
        while True:
            item = await self._queue.get()
            try:
                if item.is_db_mode:
                    if item.session_factory is None:
                        raise RuntimeError("DB mode requires session_factory")
                    await self.process_job_db(
                        job_id=item.job_id,
                        excel_path=item.excel_path,
                        email=item.email,
                        file_stem=item.file_stem,
                        created_at=item.created_at,
                        session_factory=item.session_factory,
                        prediction_service=item.prediction_service,
                        file_service=item.file_service,
                        email_service=item.email_service,
                    )
                else:
                    if item.job_ref is None:
                        raise RuntimeError("In-memory mode requires job_ref")
                    await self.process_job_in_memory(
                        job=item.job_ref,
                        prediction_service=item.prediction_service,
                        file_service=item.file_service,
                        email_service=item.email_service,
                    )
            except Exception:
                logger.opt(exception=True).error("Worker loop error for job {}", item.job_id)
            finally:
                self._queue.task_done()

    def enqueue(self, **kwargs: Any) -> int:
        """Add a job to the processing queue.

        Returns:
            Queue position (1-based).

        Raises:
            JobQueueFullError: If queue is at max capacity.
        """
        try:
            item = _QueueItem(**kwargs)
            self._queue.put_nowait(item)
        except asyncio.QueueFull as err:
            raise JobQueueFullError(
                f"Queue is full (max {self._max_queue_size} jobs). Please try again later."
            ) from err
        position = self._queue.qsize()
        logger.info(
            "Job {} enqueued at position {}",
            kwargs.get("job_id", "?"),
            position,
        )
        return position

    @property
    def queue_size(self) -> int:
        """Current number of jobs waiting in the queue."""
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        """Maximum queue capacity."""
        return self._max_queue_size

    # ------------------------------------------------------------------
    # In-memory helpers (dev mode fallback)
    # ------------------------------------------------------------------

    def has_active_job_in_memory(self) -> bool:
        """Check if a job is currently running (in-memory mode)."""
        return self._active_job_id is not None

    def get_active_job_in_memory(self) -> Job | None:
        """Get the currently running job (in-memory mode)."""
        if self._active_job_id:
            return self._jobs.get(self._active_job_id)
        return None

    def create_job_in_memory(self, email: str, excel_path: Path, file_stem: str) -> Job:
        """Create a new job in memory (queued if worker is active)."""
        job = Job(email=email, excel_path=excel_path, file_stem=file_stem)
        if self.has_active_job_in_memory():
            job.status = JobStatus.QUEUED
        self._jobs[job.id] = job
        logger.info("Created job: {} for {}", job.id, email)
        return job

    def get_job_in_memory(self, job_id: str) -> Job:
        """Get job by ID (in-memory mode)."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job

    def get_all_jobs_in_memory(self) -> list[Job]:
        """Get all jobs (in-memory mode)."""
        return list(self._jobs.values())

    def get_stats_in_memory(self) -> dict[str, int]:
        """Get job statistics (in-memory mode)."""
        stats: dict[str, int] = {
            "total": len(self._jobs),
            "pending": 0,
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for job in self._jobs.values():
            stats[job.status.value] += 1
        return stats

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def has_active_job_db(self, session: AsyncSession) -> bool:
        """Check if a job is currently running (DB mode)."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        active = await repo.get_active_job()
        return active is not None

    async def get_active_job_db(self, session: AsyncSession) -> Any:
        """Get the currently active job from DB."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        return await repo.get_active_job()

    async def create_job_db(
        self,
        session: AsyncSession,
        email: str,
        excel_path: Path,
        file_stem: str,
    ) -> Any:
        """Create a new job in DB (queued if other active jobs exist)."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        active_count = await repo.count_active_jobs()

        # Determine initial status
        status = "queued" if active_count > 0 else "pending"

        job_id = uuid.uuid4().hex[:12]
        job_data = {
            "id": job_id,
            "email": email,
            "excel_path": str(excel_path),
            "file_stem": file_stem,
            "status": status,
            "email_status": "pending",
            "created_at": datetime.now(tz=TZ_ISTANBUL),
        }
        job = await repo.create(job_data)
        await session.commit()
        logger.info("Created job: {} for {} (status={})", job_id, email, status)
        return job

    async def get_job_db(self, session: AsyncSession, job_id: str) -> Any:
        """Get job by ID from DB."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job

    # ------------------------------------------------------------------
    # DB checkpoint helpers — delegate to job_steps module
    # ------------------------------------------------------------------

    @staticmethod
    async def _update_progress_db(
        session_factory: async_sessionmaker[AsyncSession],
        job_id: str,
        message: str,
    ) -> None:
        """Update job progress message via a short-lived DB session."""
        await update_progress_db(session_factory, job_id, message)

    @staticmethod
    async def _update_status_db(
        session_factory: async_sessionmaker[AsyncSession],
        job_id: str,
        status: str,
        *,
        result_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update job status via a short-lived DB session."""
        await update_status_db(
            session_factory,
            job_id,
            status,
            result_path=result_path,
            error=error,
        )

    # ------------------------------------------------------------------
    # Process job (DB mode) — orchestrator
    # ------------------------------------------------------------------

    async def process_job_db(
        self,
        job_id: str,
        excel_path: str,
        email: str,
        file_stem: str,
        created_at: datetime,
        session_factory: async_sessionmaker[AsyncSession],
        prediction_service: PredictionService,
        file_service: FileService,
        email_service: EmailService,
    ) -> None:
        """Process a job using DB for persistence.

        Each checkpoint uses a separate session to avoid holding connections.
        Steps are delegated to standalone functions in job_steps module.
        """
        # Mark running (no lock needed — worker loop guarantees sequential)
        await update_status_db(session_factory, job_id, "running")

        try:
            # Progress: data analysis
            await update_progress_db(session_factory, job_id, "Veri analizi yapiliyor...")

            # Step 1: Match previous predictions with actuals
            logger.info("[Job {}] Step 1/8: Matching previous predictions", job_id)
            await match_previous_predictions_step(
                job_id,
                excel_path,
                session_factory,
                prediction_service,
                email_service,
            )

            # Step 2: Run prediction pipeline
            logger.info("[Job {}] Step 2/8: Running prediction pipeline", job_id)
            predictions = await run_prediction_step(
                excel_path,
                prediction_service,
            )

            # Step 3: Store predictions in DB
            logger.info("[Job {}] Step 3/8: Storing predictions in DB", job_id)
            await store_predictions_step(
                job_id,
                predictions,
                session_factory,
            )

            # Step 4: Store weather snapshot
            logger.info("[Job {}] Step 4/8: Storing weather snapshot", job_id)
            await store_weather_step(
                job_id,
                predictions,
                session_factory,
            )

            # Step 5: Store EPIAS + feature importance metadata
            logger.info("[Job {}] Step 5/8: Storing metadata", job_id)
            await store_metadata_step(
                job_id,
                predictions,
                prediction_service,
                session_factory,
            )

            # Step 6: Create output file
            logger.info("[Job {}] Step 6/8: Creating output file", job_id)
            await update_progress_db(
                session_factory,
                job_id,
                "Rapor dosyasi olusturuluyor...",
            )

            output_path = await create_output_step(
                predictions,
                file_stem,
                file_service,
            )

            # Step 7: Archive features + GDrive upload
            logger.info("[Job {}] Step 7/8: Archiving artifacts", job_id)
            await archive_step(
                job_id,
                file_stem,
                output_path,
                created_at,
                predictions,
                prediction_service,
                session_factory,
            )

            # Step 8: Send email (NON-FATAL — last step, after GDrive)
            logger.info("[Job {}] Step 8/8: Sending email", job_id)
            await update_progress_db(session_factory, job_id, "E-posta gonderiliyor...")

            await send_email_step(
                job_id,
                email,
                output_path,
                created_at,
                session_factory,
                email_service,
            )

            # Mark complete
            await update_status_db(
                session_factory,
                job_id,
                "completed",
                result_path=str(output_path),
            )

            logger.info("[Job {}] Completed successfully", job_id)

            # Audit: job_complete (non-fatal)
            try:
                from energy_forecast.db.repositories.audit_repo import (
                    AuditRepository,
                )

                async with session_factory() as session:
                    audit = AuditRepository(session)
                    await audit.log(
                        action="job_complete",
                        user_email=email,
                        details={"job_id": job_id},
                    )
                    await session.commit()
            except Exception as exc:
                logger.debug("Audit log (job_complete) failed: {}", exc)

        except Exception as e:
            error_msg = str(e)
            logger.opt(exception=True).error("Job {} failed: {}", job_id, error_msg)
            # Sanitize error for user — strip internal paths and DB details
            safe_msg = error_msg.split("\n")[0][:200] if error_msg else "Internal error"
            await update_status_db(
                session_factory,
                job_id,
                "failed",
                error=safe_msg,
            )

            # Audit: job_failed (non-fatal)
            try:
                from energy_forecast.db.repositories.audit_repo import (
                    AuditRepository,
                )

                async with session_factory() as session:
                    audit = AuditRepository(session)
                    await audit.log(
                        action="job_failed",
                        user_email=email,
                        details={
                            "job_id": job_id,
                            "error": safe_msg,
                        },
                    )
                    await session.commit()
            except Exception as exc:
                logger.debug("Audit log (job_failed) failed: {}", exc)

            try:
                email_service.send_error_notification(
                    to_email=email,
                    job_id=job_id,
                    error_message=safe_msg,
                )
            except Exception as email_err:
                logger.error("Failed to send error notification: {}", email_err)

    # ------------------------------------------------------------------
    # Process job (in-memory mode)
    # ------------------------------------------------------------------

    async def process_job_in_memory(
        self,
        job: Job,
        prediction_service: PredictionService,
        file_service: FileService,
        email_service: EmailService,
    ) -> None:
        """Process a job with in-memory storage (no lock -- worker loop is sequential)."""
        job.status = JobStatus.RUNNING
        self._active_job_id = job.id
        logger.info("Job {} started", job.id)

        try:
            self._jobs[job.id].progress = "Veri analizi yapiliyor..."
            predictions = await asyncio.to_thread(
                prediction_service.run_prediction,
                excel_path=job.excel_path,
                progress_callback=lambda msg: setattr(
                    self._jobs.get(job.id, job),
                    "progress",
                    msg,
                ),
            )

            self._jobs[job.id].progress = "Rapor dosyasi olusturuluyor..."
            output_path = file_service.create_output_xlsx(predictions, job.file_stem)

            self._jobs[job.id].progress = "E-posta gonderiliyor..."
            try:
                email_service.send_with_retry(
                    to_email=job.email,
                    attachment_path=output_path,
                    job_id=job.id,
                    created_at=job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                )
            except Exception as email_err:
                logger.opt(exception=True).warning(
                    "Email failed for job {} (non-fatal): {}",
                    job.id,
                    email_err,
                )

            job.status = JobStatus.COMPLETED
            job.result_path = output_path
            job.completed_at = datetime.now(tz=TZ_ISTANBUL)
            self._active_job_id = None
            logger.info("Job {} completed", job.id)

        except Exception as e:
            error_msg = str(e)
            # Sanitize error for user — strip internal paths and DB details (mirror DB-mode)
            safe_msg = error_msg.split("\n")[0][:200] if error_msg else "Internal error"
            job.status = JobStatus.FAILED
            job.error = safe_msg
            job.completed_at = datetime.now(tz=TZ_ISTANBUL)
            self._active_job_id = None
            logger.opt(exception=True).error("Job {} failed: {}", job.id, error_msg)

            try:
                email_service.send_error_notification(
                    to_email=job.email,
                    job_id=job.id,
                    error_message=safe_msg,
                )
            except Exception as email_err:
                logger.error("Failed to send error notification: {}", email_err)

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed/failed jobs from memory."""
        from datetime import timedelta

        threshold = datetime.now(tz=TZ_ISTANBUL) - timedelta(hours=max_age_hours)
        to_remove = []

        for job_id, job in self._jobs.items():
            is_finished = job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            )
            is_old = job.completed_at is not None and job.completed_at < threshold
            if is_finished and is_old:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]

        if to_remove:
            logger.info("Cleaned up {} old jobs", len(to_remove))
        return len(to_remove)
