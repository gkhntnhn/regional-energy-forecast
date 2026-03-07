"""Tests for job manager (in-memory mode)."""

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
        file_stem="01-03-2026_12-00-00",
    )


class TestJob:
    """Tests for Job model."""

    def test_job_defaults(self) -> None:
        """Test job default values."""
        job = Job(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
            file_stem="01-03-2026_12-00-00",
        )

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
            file_stem="01-03-2026_12-00-00",
            status=JobStatus.RUNNING,
            created_at=now,
        )

        assert job.id == "custom123"
        assert job.status == JobStatus.RUNNING


class TestJobManager:
    """Tests for JobManager (in-memory mode)."""

    def test_has_active_job_empty(self, job_manager: JobManager) -> None:
        """Test no active job initially."""
        assert job_manager.has_active_job_in_memory() is False

    def test_create_job(self, job_manager: JobManager) -> None:
        """Test job creation."""
        job = job_manager.create_job_in_memory(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
            file_stem="01-03-2026_12-00-00",
        )

        assert job.id in job_manager._jobs
        assert job.status == JobStatus.PENDING

    def test_create_job_when_active_raises(self, job_manager: JobManager) -> None:
        """Test job creation fails when another is active."""
        job = job_manager.create_job_in_memory(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
            file_stem="01-03-2026_12-00-00",
        )
        job_manager._jobs[job.id].status = JobStatus.RUNNING
        job_manager._active_job_id = job.id

        with pytest.raises(JobQueueFullError):
            job_manager.create_job_in_memory(
                email="test2@test.com",
                excel_path=Path("/tmp/test2.xlsx"),
                file_stem="01-03-2026_12-00-01",
            )

    def test_get_job(self, job_manager: JobManager) -> None:
        """Test getting job by ID."""
        created = job_manager.create_job_in_memory(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
            file_stem="01-03-2026_12-00-00",
        )

        found = job_manager.get_job_in_memory(created.id)
        assert found.id == created.id

    def test_get_job_not_found(self, job_manager: JobManager) -> None:
        """Test getting nonexistent job."""
        with pytest.raises(JobNotFoundError):
            job_manager.get_job_in_memory("nonexistent")

    def test_get_stats(self, job_manager: JobManager) -> None:
        """Test job statistics."""
        job1 = job_manager.create_job_in_memory(
            "a@a.com", Path("/tmp/a.xlsx"), "01-03-2026_12-00-00"
        )
        job1.status = JobStatus.COMPLETED
        job1.completed_at = datetime.now(tz=TZ_ISTANBUL)
        job_manager._active_job_id = None

        job2 = job_manager.create_job_in_memory(
            "b@b.com", Path("/tmp/b.xlsx"), "01-03-2026_12-00-01"
        )
        job2.status = JobStatus.FAILED
        job2.completed_at = datetime.now(tz=TZ_ISTANBUL)

        job_manager.create_job_in_memory(
            "c@c.com", Path("/tmp/c.xlsx"), "01-03-2026_12-00-02"
        )

        stats = job_manager.get_stats_in_memory()

        assert stats["total"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 1

    def test_cleanup_old_jobs(self, job_manager: JobManager) -> None:
        """Test cleaning up old jobs."""
        job = job_manager.create_job_in_memory(
            "test@test.com", Path("/tmp/test.xlsx"), "01-03-2026_12-00-00"
        )
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(tz=TZ_ISTANBUL) - timedelta(hours=48)

        removed = job_manager.cleanup_old_jobs(max_age_hours=24)

        assert removed == 1
        assert job.id not in job_manager._jobs


class TestJobManagerProcessJob:
    """Tests for JobManager.process_job_in_memory()."""

    @pytest.mark.asyncio
    async def test_process_job_success(self, job_manager: JobManager) -> None:
        """Test successful job processing."""
        import pandas as pd

        job = job_manager.create_job_in_memory(
            "test@test.com", Path("/tmp/test.xlsx"), "01-03-2026_12-00-00"
        )

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pd.DataFrame(
            {"prediction": [1000, 1100]},
            index=pd.date_range("2025-01-01", periods=2, freq="h"),
        )

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_prediction_result.return_value = True

        await job_manager.process_job_in_memory(
            job=job,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        assert job_manager.get_job_in_memory(job.id).status == JobStatus.COMPLETED
        mock_prediction.run_prediction.assert_called_once()
        call_args = mock_file.create_output_xlsx.call_args
        assert call_args[0][1] == "01-03-2026_12-00-00"
        mock_email.send_prediction_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_failure(self, job_manager: JobManager) -> None:
        """Test job processing with failure."""
        job = job_manager.create_job_in_memory(
            "test@test.com", Path("/tmp/test.xlsx"), "01-03-2026_12-00-00"
        )

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.side_effect = Exception("Prediction failed")

        mock_file = MagicMock()
        mock_email = MagicMock()
        mock_email.send_error_notification.return_value = True

        await job_manager.process_job_in_memory(
            job=job,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        result = job_manager.get_job_in_memory(job.id)
        assert result.status == JobStatus.FAILED
        assert "Prediction failed" in result.error  # type: ignore[operator]
        mock_email.send_error_notification.assert_called_once()
