"""Tests for job manager."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from energy_forecast.serving.exceptions import JobNotFoundError, JobQueueFullError
from energy_forecast.serving.job_manager import Job, JobManager
from energy_forecast.serving.schemas import JobStatus
from energy_forecast.utils import TZ_ISTANBUL


@pytest.fixture
def job_manager() -> JobManager:
    """Create fresh job manager."""
    return JobManager()


@pytest.fixture
def sample_job() -> Job:
    """Create sample job."""
    return Job(
        email="test@example.com",
        excel_path=Path("/tmp/test.xlsx"),
    )


class TestJob:
    """Tests for Job model."""

    def test_job_defaults(self) -> None:
        """Test job default values."""
        job = Job(email="test@test.com", excel_path=Path("/tmp/test.xlsx"))

        assert len(job.id) == 12
        assert job.status == JobStatus.PENDING
        assert job.progress is None
        assert job.error is None
        assert job.result_path is None
        assert job.completed_at is None

    def test_job_custom_values(self) -> None:
        """Test job with custom values."""
        now = datetime.now()
        job = Job(
            id="custom123",
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
            status=JobStatus.RUNNING,
            created_at=now,
        )

        assert job.id == "custom123"
        assert job.status == JobStatus.RUNNING


class TestJobManager:
    """Tests for JobManager."""

    def test_has_active_job_empty(self, job_manager: JobManager) -> None:
        """Test no active job initially."""
        assert job_manager.has_active_job() is False

    def test_create_job(self, job_manager: JobManager) -> None:
        """Test job creation."""
        job = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )

        assert job.id in job_manager._jobs
        assert job.status == JobStatus.PENDING

    def test_create_job_when_active_raises(self, job_manager: JobManager) -> None:
        """Test job creation fails when another is active."""
        # Create and mark as running
        job = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )
        job_manager._set_running(job.id)

        # Try to create another
        with pytest.raises(JobQueueFullError):
            job_manager.create_job(
                email="test2@test.com",
                excel_path=Path("/tmp/test2.xlsx"),
            )

    def test_get_job(self, job_manager: JobManager) -> None:
        """Test getting job by ID."""
        created = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )

        found = job_manager.get_job(created.id)
        assert found.id == created.id

    def test_get_job_not_found(self, job_manager: JobManager) -> None:
        """Test getting nonexistent job."""
        with pytest.raises(JobNotFoundError):
            job_manager.get_job("nonexistent")

    def test_update_progress(self, job_manager: JobManager) -> None:
        """Test progress update."""
        job = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )

        job_manager.update_progress(job.id, "Step 1 of 3")

        assert job_manager.get_job(job.id).progress == "Step 1 of 3"

    def test_set_running(self, job_manager: JobManager) -> None:
        """Test setting job to running."""
        job = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )

        job_manager._set_running(job.id)

        assert job_manager.get_job(job.id).status == JobStatus.RUNNING
        assert job_manager.has_active_job() is True
        assert job_manager._active_job_id == job.id

    def test_complete_job(self, job_manager: JobManager) -> None:
        """Test completing a job."""
        job = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )
        job_manager._set_running(job.id)

        result_path = Path("/tmp/result.xlsx")
        job_manager._complete_job(job.id, result_path)

        completed = job_manager.get_job(job.id)
        assert completed.status == JobStatus.COMPLETED
        assert completed.result_path == result_path
        assert completed.completed_at is not None
        assert job_manager.has_active_job() is False

    def test_fail_job(self, job_manager: JobManager) -> None:
        """Test failing a job."""
        job = job_manager.create_job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
        )
        job_manager._set_running(job.id)

        job_manager._fail_job(job.id, "Something went wrong")

        failed = job_manager.get_job(job.id)
        assert failed.status == JobStatus.FAILED
        assert failed.error == "Something went wrong"
        assert job_manager.has_active_job() is False

    def test_get_stats(self, job_manager: JobManager) -> None:
        """Test job statistics."""
        # Create jobs in different states
        job1 = job_manager.create_job("a@a.com", Path("/tmp/a.xlsx"))
        job_manager._set_running(job1.id)
        job_manager._complete_job(job1.id, Path("/tmp/out.xlsx"))

        job2 = job_manager.create_job("b@b.com", Path("/tmp/b.xlsx"))
        job_manager._set_running(job2.id)
        job_manager._fail_job(job2.id, "error")

        job_manager.create_job("c@c.com", Path("/tmp/c.xlsx"))  # pending job

        stats = job_manager.get_stats()

        assert stats["total"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 1

    def test_cleanup_old_jobs(self, job_manager: JobManager) -> None:
        """Test cleaning up old jobs."""
        # Create an old completed job
        job = job_manager.create_job("test@test.com", Path("/tmp/test.xlsx"))
        job_manager._set_running(job.id)
        job_manager._complete_job(job.id, Path("/tmp/out.xlsx"))

        # Make it old
        job_manager._jobs[job.id].completed_at = datetime.now(tz=TZ_ISTANBUL) - timedelta(hours=48)

        removed = job_manager.cleanup_old_jobs(max_age_hours=24)

        assert removed == 1
        assert job.id not in job_manager._jobs


class TestJobManagerProcessJob:
    """Tests for JobManager.process_job()."""

    @pytest.mark.asyncio
    async def test_process_job_success(self, job_manager: JobManager) -> None:
        """Test successful job processing."""
        import pandas as pd

        job = job_manager.create_job("test@test.com", Path("/tmp/test.xlsx"))

        # Mock services
        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pd.DataFrame(
            {"prediction": [1000, 1100]},
            index=pd.date_range("2025-01-01", periods=2, freq="h"),
        )

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_prediction_result.return_value = True

        await job_manager.process_job(
            job=job,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        assert job_manager.get_job(job.id).status == JobStatus.COMPLETED
        mock_prediction.run_prediction.assert_called_once()
        mock_file.create_output_xlsx.assert_called_once()
        mock_email.send_prediction_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_failure(self, job_manager: JobManager) -> None:
        """Test job processing with failure (error is captured, not re-raised)."""
        job = job_manager.create_job("test@test.com", Path("/tmp/test.xlsx"))

        # Mock services with error
        mock_prediction = MagicMock()
        mock_prediction.run_prediction.side_effect = Exception("Prediction failed")

        mock_file = MagicMock()
        mock_email = MagicMock()
        mock_email.send_error_notification.return_value = True

        # Background tasks should NOT re-raise — error is recorded in job state
        await job_manager.process_job(
            job=job,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        assert job_manager.get_job(job.id).status == JobStatus.FAILED
        assert "Prediction failed" in job_manager.get_job(job.id).error  # type: ignore
        mock_email.send_error_notification.assert_called_once()
