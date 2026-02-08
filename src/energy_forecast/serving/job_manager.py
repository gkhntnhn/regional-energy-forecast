"""Job queue manager with single-worker guarantee."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.serving.exceptions import JobNotFoundError, JobQueueFullError
from energy_forecast.serving.schemas import JobStatus

if TYPE_CHECKING:
    from energy_forecast.serving.services.email_service import EmailService
    from energy_forecast.serving.services.file_service import FileService
    from energy_forecast.serving.services.prediction_service import PredictionService


class Job(BaseModel):
    """Prediction job data."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    email: str
    excel_path: Path
    status: JobStatus = JobStatus.PENDING
    progress: str | None = None
    error: str | None = None
    result_path: Path | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}


class JobManager:
    """Manages prediction jobs with single-worker queue.

    Uses asyncio.Lock to ensure only one prediction runs at a time.
    Jobs are stored in-memory (suitable for single-instance deployment).

    For multi-instance deployment, replace with Redis or database backend.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._active_job_id: str | None = None

    def has_active_job(self) -> bool:
        """Check if a job is currently running.

        Returns:
            True if a job is in RUNNING status.
        """
        return self._active_job_id is not None

    def get_active_job(self) -> Job | None:
        """Get the currently running job if any."""
        if self._active_job_id:
            return self._jobs.get(self._active_job_id)
        return None

    def create_job(self, email: str, excel_path: Path) -> Job:
        """Create a new pending job.

        Args:
            email: Recipient email address.
            excel_path: Path to uploaded Excel file.

        Returns:
            Created Job instance.

        Raises:
            JobQueueFullError: If a job is already running.
        """
        if self.has_active_job():
            active = self.get_active_job()
            raise JobQueueFullError(
                f"A job is already running (ID: {active.id if active else 'unknown'}). "
                "Please wait for it to complete."
            )

        job = Job(email=email, excel_path=excel_path)
        self._jobs[job.id] = job
        logger.info("Created job: {} for {}", job.id, email)
        return job

    def get_job(self, job_id: str) -> Job:
        """Get job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Job instance.

        Raises:
            JobNotFoundError: If job not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job

    def update_progress(self, job_id: str, progress: str) -> None:
        """Update job progress message.

        Args:
            job_id: Job identifier.
            progress: Progress message.
        """
        if job_id in self._jobs:
            self._jobs[job_id].progress = progress
            logger.debug("Job {} progress: {}", job_id, progress)

    def _set_running(self, job_id: str) -> None:
        """Mark job as running."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.RUNNING
            self._active_job_id = job_id
            logger.info("Job {} started", job_id)

    def _complete_job(self, job_id: str, result_path: Path) -> None:
        """Mark job as completed.

        Args:
            job_id: Job identifier.
            result_path: Path to output file.
        """
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].result_path = result_path
            self._jobs[job_id].completed_at = datetime.now()
            self._active_job_id = None
            logger.info("Job {} completed", job_id)

    def _fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed.

        Args:
            job_id: Job identifier.
            error: Error message.
        """
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].error = error
            self._jobs[job_id].completed_at = datetime.now()
            self._active_job_id = None
            logger.error("Job {} failed: {}", job_id, error)

    async def process_job(
        self,
        job: Job,
        prediction_service: PredictionService,
        file_service: FileService,
        email_service: EmailService,
    ) -> None:
        """Process a job with single-worker guarantee.

        Acquires lock before processing to ensure only one job runs at a time.
        Runs prediction, creates output file, and sends email.

        Args:
            job: Job to process.
            prediction_service: Prediction service instance.
            file_service: File service instance.
            email_service: Email service instance.
        """
        async with self._lock:
            self._set_running(job.id)

            try:
                # Run prediction (sync, but in thread pool via BackgroundTasks)
                self.update_progress(job.id, "Running prediction pipeline...")
                predictions = prediction_service.run_prediction(
                    excel_path=job.excel_path,
                    progress_callback=lambda msg: self.update_progress(job.id, msg),
                )

                # Create output file
                self.update_progress(job.id, "Creating output file...")
                output_path = file_service.create_output_xlsx(predictions, job.id)

                # Send email
                self.update_progress(job.id, "Sending email...")
                email_service.send_prediction_result(
                    to_email=job.email,
                    attachment_path=output_path,
                    job_id=job.id,
                    created_at=job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                )

                self._complete_job(job.id, output_path)

            except Exception as e:
                error_msg = str(e)
                self._fail_job(job.id, error_msg)

                # Try to send error notification
                try:
                    email_service.send_error_notification(
                        to_email=job.email,
                        job_id=job.id,
                        error_message=error_msg,
                    )
                except Exception as email_err:
                    logger.error("Failed to send error notification: {}", email_err)

                raise

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed/failed jobs from memory.

        Args:
            max_age_hours: Maximum age of jobs to keep.

        Returns:
            Number of jobs removed.
        """
        from datetime import timedelta

        threshold = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for job_id, job in self._jobs.items():
            is_finished = job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            is_old = job.completed_at is not None and job.completed_at < threshold
            if is_finished and is_old:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]

        if to_remove:
            logger.info("Cleaned up {} old jobs", len(to_remove))
        return len(to_remove)

    def get_all_jobs(self) -> list[Job]:
        """Get all jobs (for debugging/admin)."""
        return list(self._jobs.values())

    def get_stats(self) -> dict[str, int]:
        """Get job statistics."""
        stats: dict[str, int] = {
            "total": len(self._jobs),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for job in self._jobs.values():
            stats[job.status.value] += 1
        return stats
