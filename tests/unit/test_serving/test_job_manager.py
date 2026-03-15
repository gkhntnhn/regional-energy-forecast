"""Tests for job manager (in-memory and DB modes)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
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
        now = datetime.now(tz=TZ_ISTANBUL)
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

    def test_create_job_when_active_queues(self, job_manager: JobManager) -> None:
        """Test second job gets QUEUED status when another is active."""
        job = job_manager.create_job_in_memory(
            email="test@test.com",
            excel_path=Path("/tmp/test.xlsx"),
            file_stem="01-03-2026_12-00-00",
        )
        job_manager._jobs[job.id].status = JobStatus.RUNNING
        job_manager._active_job_id = job.id

        job2 = job_manager.create_job_in_memory(
            email="test2@test.com",
            excel_path=Path("/tmp/test2.xlsx"),
            file_stem="01-03-2026_12-00-01",
        )
        assert job2.status == JobStatus.QUEUED

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

        job_manager.create_job_in_memory("c@c.com", Path("/tmp/c.xlsx"), "01-03-2026_12-00-02")

        stats = job_manager.get_stats_in_memory()

        assert stats["total"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 1

    def test_get_active_job_returns_running(self, job_manager: JobManager) -> None:
        """Test get_active_job_in_memory returns running job."""
        job = job_manager.create_job_in_memory(
            "test@test.com", Path("/tmp/test.xlsx"), "01-03-2026_12-00-00"
        )
        job_manager._active_job_id = job.id

        active = job_manager.get_active_job_in_memory()
        assert active is not None
        assert active.id == job.id

    def test_get_active_job_returns_none_when_inactive(self, job_manager: JobManager) -> None:
        """Test get_active_job_in_memory returns None when no active job."""
        assert job_manager.get_active_job_in_memory() is None

    def test_get_all_jobs(self, job_manager: JobManager) -> None:
        """Test get_all_jobs_in_memory returns all jobs."""
        job_manager.create_job_in_memory("a@a.com", Path("/tmp/a.xlsx"), "01-03-2026_12-00-00")
        job_manager.create_job_in_memory("b@b.com", Path("/tmp/b.xlsx"), "01-03-2026_12-00-01")

        all_jobs = job_manager.get_all_jobs_in_memory()
        assert len(all_jobs) == 2

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
        mock_email.send_with_retry.return_value = (True, 1, None)

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
        mock_email.send_with_retry.assert_called_once()

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

    @pytest.mark.asyncio
    async def test_process_job_failure_email_error_swallowed(self, job_manager: JobManager) -> None:
        """Test error notification failure is swallowed (not re-raised)."""
        job = job_manager.create_job_in_memory(
            "test@test.com", Path("/tmp/test.xlsx"), "01-03-2026_12-00-00"
        )

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.side_effect = Exception("Pipeline crash")

        mock_file = MagicMock()
        mock_email = MagicMock()
        mock_email.send_error_notification.side_effect = Exception("SMTP down")

        # Should NOT raise despite email failure
        await job_manager.process_job_in_memory(
            job=job,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        result = job_manager.get_job_in_memory(job.id)
        assert result.status == JobStatus.FAILED


# ---------------------------------------------------------------------------
# DB-aware tests
# ---------------------------------------------------------------------------


class TestJobManagerDB:
    """Tests for JobManager DB-aware methods (create, get, has_active, etc.)."""

    @pytest.mark.asyncio
    async def test_create_job_db_creates_record(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test create_job_db inserts a pending job into the database."""
        job = await job_manager.create_job_db(
            session=db_session,
            email="user@example.com",
            excel_path=Path("/tmp/input.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )

        assert job is not None
        assert len(job.id) == 12
        assert job.email == "user@example.com"
        assert job.status == "pending"
        assert job.excel_path == str(Path("/tmp/input.xlsx"))
        assert job.file_stem == "07-03-2026_10-00-00"
        assert job.email_status == "pending"

    @pytest.mark.asyncio
    async def test_create_job_db_queues_when_active_exists(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test create_job_db creates second job with 'queued' status."""
        await job_manager.create_job_db(
            session=db_session,
            email="first@example.com",
            excel_path=Path("/tmp/first.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )

        job2 = await job_manager.create_job_db(
            session=db_session,
            email="second@example.com",
            excel_path=Path("/tmp/second.xlsx"),
            file_stem="07-03-2026_10-00-01",
        )
        assert job2.status == "queued"

    @pytest.mark.asyncio
    async def test_create_job_db_allows_after_completion(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test create_job_db allows new job when previous is completed."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        job1 = await job_manager.create_job_db(
            session=db_session,
            email="first@example.com",
            excel_path=Path("/tmp/first.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )
        repo = JobRepository(db_session)
        await repo.update_status(job1.id, "completed")
        await db_session.commit()

        job2 = await job_manager.create_job_db(
            session=db_session,
            email="second@example.com",
            excel_path=Path("/tmp/second.xlsx"),
            file_stem="07-03-2026_10-00-01",
        )
        assert job2 is not None
        assert job2.id != job1.id

    @pytest.mark.asyncio
    async def test_get_job_db_returns_existing(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test get_job_db returns a job that exists in the database."""
        created = await job_manager.create_job_db(
            session=db_session,
            email="user@example.com",
            excel_path=Path("/tmp/input.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )

        found = await job_manager.get_job_db(db_session, created.id)
        assert found.id == created.id
        assert found.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_job_db_raises_not_found(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test get_job_db raises JobNotFoundError for missing job_id."""
        with pytest.raises(JobNotFoundError, match="Job not found"):
            await job_manager.get_job_db(db_session, "nonexistent99")

    @pytest.mark.asyncio
    async def test_has_active_job_db_false_when_empty(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test has_active_job_db returns False on empty database."""
        result = await job_manager.has_active_job_db(db_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_has_active_job_db_true_when_pending(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test has_active_job_db returns True when a pending job exists."""
        await job_manager.create_job_db(
            session=db_session,
            email="user@example.com",
            excel_path=Path("/tmp/input.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )
        result = await job_manager.has_active_job_db(db_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_has_active_job_db_false_when_completed(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test has_active_job_db returns False when all jobs are completed."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        job = await job_manager.create_job_db(
            session=db_session,
            email="user@example.com",
            excel_path=Path("/tmp/input.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )
        repo = JobRepository(db_session)
        await repo.update_status(job.id, "completed")
        await db_session.commit()

        result = await job_manager.has_active_job_db(db_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_active_job_db_returns_none_when_empty(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test get_active_job_db returns None with no active jobs."""
        result = await job_manager.get_active_job_db(db_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_active_job_db_returns_running_job(
        self, job_manager: JobManager, db_session: Any
    ) -> None:
        """Test get_active_job_db returns the running job."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        job = await job_manager.create_job_db(
            session=db_session,
            email="user@example.com",
            excel_path=Path("/tmp/input.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )
        repo = JobRepository(db_session)
        await repo.update_status(job.id, "running")
        await db_session.commit()

        active = await job_manager.get_active_job_db(db_session)
        assert active is not None
        assert active.id == job.id
        assert active.status == "running"


class TestProcessJobDB:
    """Tests for JobManager.process_job_db() — the main DB processing pipeline."""

    @pytest.mark.asyncio
    async def test_process_job_db_success(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db marks job completed on success path."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        # Seed a pending job
        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "success12345",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        # Build mock prediction result
        pred_index = pd.date_range("2026-03-08", periods=48, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0 + i for i in range(48)],
                "period": ["intraday"] * 24 + ["day_ahead"] * 24,
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {"model": "ensemble_v1"}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        created_at = datetime.now(tz=TZ_ISTANBUL)
        await job_manager.process_job_db(
            job_id="success12345",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=created_at,
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        # Verify job is completed in DB
        async with db_session_factory() as session:
            job = await session.get(JobModel, "success12345")
            assert job is not None
            assert job.status == "completed"
            assert job.result_path == str(Path("/tmp/output.xlsx"))
            assert job.completed_at is not None

        mock_prediction.run_prediction.assert_called_once()
        mock_file.create_output_xlsx.assert_called_once()
        mock_email.send_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_db_failure_marks_failed(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db marks job as failed when prediction raises."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        # Seed a pending job
        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "fail_job_123",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.side_effect = RuntimeError("Pipeline crashed")
        mock_prediction._data_loader = None

        mock_file = MagicMock()
        mock_email = MagicMock()
        mock_email.send_error_notification.return_value = True

        created_at = datetime.now(tz=TZ_ISTANBUL)
        await job_manager.process_job_db(
            job_id="fail_job_123",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=created_at,
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        # Verify job is failed in DB
        async with db_session_factory() as session:
            job = await session.get(JobModel, "fail_job_123")
            assert job is not None
            assert job.status == "failed"
            assert job.error is not None
            assert "Pipeline crashed" in job.error

        mock_email.send_error_notification.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_db_stores_predictions(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db stores prediction rows in predictions table."""
        from energy_forecast.db.models import PredictionModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        # Seed a pending job
        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "pred_store_1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        # Build prediction result with 4 rows for brevity
        pred_index = pd.date_range("2026-03-08", periods=4, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0, 1220.0, 1230.0],
                "period": ["intraday", "intraday", "day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        await job_manager.process_job_db(
            job_id="pred_store_1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        # Verify predictions were stored
        async with db_session_factory() as session:
            from sqlalchemy import select

            stmt = select(PredictionModel).where(PredictionModel.job_id == "pred_store_1")
            result = await session.execute(stmt)
            preds = list(result.scalars().all())
            assert len(preds) == 4
            assert all(p.model_source == "ensemble" for p in preds)
            assert preds[0].consumption_mwh == 1200.0

    @pytest.mark.asyncio
    async def test_process_job_db_stores_per_model_predictions(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db stores per-model raw predictions for analytics."""
        from energy_forecast.db.models import PredictionModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        # Seed a pending job
        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "model_preds1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )

        # Build raw_predictions with per-model columns
        raw_preds = pd.DataFrame(
            {
                "catboost_prediction": [1195.0, 1205.0],
                "prophet_prediction": [1210.0, 1220.0],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = raw_preds
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        await job_manager.process_job_db(
            job_id="model_preds1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        # 2 ensemble + 2 catboost + 2 prophet = 6 total
        async with db_session_factory() as session:
            from sqlalchemy import select

            stmt = select(PredictionModel).where(PredictionModel.job_id == "model_preds1")
            result = await session.execute(stmt)
            preds = list(result.scalars().all())
            assert len(preds) == 6

            sources = {p.model_source for p in preds}
            assert sources == {"ensemble", "catboost", "prophet"}

    @pytest.mark.asyncio
    async def test_process_job_db_email_status_updated(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db updates email_status to 'sent' on success."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "email_test_1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        await job_manager.process_job_db(
            job_id="email_test_1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        async with db_session_factory() as session:
            job = await session.get(JobModel, "email_test_1")
            assert job is not None
            assert job.email_status == "sent"
            assert job.email_attempts == 1

    @pytest.mark.asyncio
    async def test_process_job_db_failure_sends_error_email(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db sends error notification on failure."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "err_email_01",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.side_effect = ValueError("Bad data")
        mock_prediction._data_loader = None

        mock_file = MagicMock()
        mock_email = MagicMock()

        await job_manager.process_job_db(
            job_id="err_email_01",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        mock_email.send_error_notification.assert_called_once_with(
            to_email="user@example.com",
            job_id="err_email_01",
            error_message="Bad data",
        )

    @pytest.mark.asyncio
    async def test_process_job_db_weather_snapshot_failure_nonfatal(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test weather snapshot storage failure does not crash the job."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "weather_nf1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )

        # Provide a weather_data that will cause errors when bulk_create_forecast
        # is called, but since it's non-fatal, job should still complete.
        # We use a non-empty DataFrame to trigger the weather snapshot path.
        weather_df = pd.DataFrame(
            {"temperature_2m": [10.0, 11.0]},
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = weather_df
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        # Even if weather snapshot has issues, job should complete
        await job_manager.process_job_db(
            job_id="weather_nf1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        async with db_session_factory() as session:
            job = await session.get(JobModel, "weather_nf1")
            assert job is not None
            assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_process_job_db_stores_feature_importance(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db stores feature importance in metadata."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "fi_test_001",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = {"data_range": "2024-2026"}
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        fi_top = [
            {"feature": "consumption_lag_48", "importance": 25.0},
            {"feature": "temperature_2m", "importance": 15.0},
        ]

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = fi_top

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        await job_manager.process_job_db(
            job_id="fi_test_001",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        async with db_session_factory() as session:
            job = await session.get(JobModel, "fi_test_001")
            assert job is not None
            assert job.metadata_ is not None
            assert "feature_importance_top15" in job.metadata_
            assert len(job.metadata_["feature_importance_top15"]) == 2
            assert job.epias_snapshot is not None
            assert job.epias_snapshot["data_range"] == "2024-2026"

    @pytest.mark.asyncio
    async def test_process_job_db_failure_email_error_swallowed(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test error email failure during process_job_db is swallowed."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "swallow_err1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.side_effect = ValueError("Bad input")
        mock_prediction._data_loader = None

        mock_file = MagicMock()
        mock_email = MagicMock()
        mock_email.send_error_notification.side_effect = Exception("SMTP down")

        # Should not raise even though error email fails
        await job_manager.process_job_db(
            job_id="swallow_err1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        async with db_session_factory() as session:
            job = await session.get(JobModel, "swallow_err1")
            assert job is not None
            assert job.status == "failed"
            assert "Bad input" in (job.error or "")

    @pytest.mark.asyncio
    async def test_process_job_db_prediction_matching(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test process_job_db runs prediction matching with actuals."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "match_test1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        # Build a consumption DataFrame that load_excel returns
        consumption_idx = pd.date_range("2026-03-06", periods=48, freq="h", tz=TZ_ISTANBUL)
        consumption_df = pd.DataFrame(
            {"consumption": [1100.0 + i for i in range(48)]},
            index=consumption_idx,
        )

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        # Mock data_loader to return consumption
        mock_data_loader = MagicMock()
        mock_data_loader.load_excel.return_value = consumption_df

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = mock_data_loader
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        await job_manager.process_job_db(
            job_id="match_test1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        # Job should complete regardless
        async with db_session_factory() as session:
            job = await session.get(JobModel, "match_test1")
            assert job is not None
            assert job.status == "completed"

        # Verify load_excel was called
        mock_data_loader.load_excel.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_db_prediction_matching_failure_nonfatal(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test prediction matching failure does not crash the job."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "match_fail1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        # data_loader.load_excel raises
        mock_data_loader = MagicMock()
        mock_data_loader.load_excel.side_effect = Exception("Excel parse error")

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = mock_data_loader
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        # Should not raise despite matching failure
        await job_manager.process_job_db(
            job_id="match_fail1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        async with db_session_factory() as session:
            job = await session.get(JobModel, "match_fail1")
            assert job is not None
            assert job.status == "completed"


class TestEmailNonFatal:
    """Tests for email failure NOT causing job failure."""

    @pytest.mark.asyncio
    async def test_process_job_db_email_failure_still_completes(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test job completes even when email delivery fails."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "email_fail_1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        # send_with_retry returns (False, 2, "error msg") = all retries failed
        mock_email.send_with_retry.return_value = (
            False,
            2,
            "SMTP connection error: [Errno 22] Invalid argument",
        )

        await job_manager.process_job_db(
            job_id="email_fail_1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        # Job should be COMPLETED (not failed) — prediction succeeded
        async with db_session_factory() as session:
            job = await session.get(JobModel, "email_fail_1")
            assert job is not None
            assert job.status == "completed"
            assert job.email_status == "failed"
            assert job.result_path is not None

    @pytest.mark.asyncio
    async def test_process_job_db_email_exception_still_completes(
        self, job_manager: JobManager, db_session_factory: Any
    ) -> None:
        """Test job completes even when email step raises an exception."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                {
                    "id": "email_exc_1",
                    "email": "user@example.com",
                    "excel_path": "/tmp/input.xlsx",
                    "file_stem": "07-03-2026_10-00-00",
                    "status": "pending",
                    "email_status": "pending",
                }
            )
            await session.commit()

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {
                "consumption_mwh": [1200.0, 1210.0],
                "period": ["day_ahead", "day_ahead"],
            },
            index=pred_index,
        )
        pred_df.attrs["weather_data"] = None
        pred_df.attrs["raw_predictions"] = None
        pred_df.attrs["metadata"] = {}
        pred_df.attrs["epias_snapshot"] = None
        pred_df.attrs["features_df"] = None
        pred_df.attrs["forecast_mask"] = None

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df
        mock_prediction._data_loader = None
        mock_prediction.get_feature_importance_top.return_value = None

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        # send_with_retry raises an unexpected exception
        mock_email.send_with_retry.side_effect = OSError(22, "Invalid argument")

        await job_manager.process_job_db(
            job_id="email_exc_1",
            excel_path="/tmp/input.xlsx",
            email="user@example.com",
            file_stem="07-03-2026_10-00-00",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=db_session_factory,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        async with db_session_factory() as session:
            job = await session.get(JobModel, "email_exc_1")
            assert job is not None
            assert job.status == "completed"
            assert job.email_status == "failed"

    @pytest.mark.asyncio
    async def test_process_job_in_memory_email_failure_still_completes(
        self, job_manager: JobManager
    ) -> None:
        """Test in-memory job completes even when email fails."""
        job = Job(
            email="test@example.com",
            excel_path=Path("/tmp/test.xlsx"),
            file_stem="07-03-2026_10-00-00",
        )
        job_manager._jobs[job.id] = job

        pred_index = pd.date_range("2026-03-08", periods=2, freq="h", tz=TZ_ISTANBUL)
        pred_df = pd.DataFrame(
            {"consumption_mwh": [1200.0, 1210.0], "period": ["day_ahead", "day_ahead"]},
            index=pred_index,
        )

        mock_prediction = MagicMock()
        mock_prediction.run_prediction.return_value = pred_df

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/output.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.side_effect = OSError(22, "Invalid argument")

        await job_manager.process_job_in_memory(
            job=job,
            prediction_service=mock_prediction,
            file_service=mock_file,
            email_service=mock_email,
        )

        assert job.status == JobStatus.COMPLETED
        assert job.result_path == Path("/tmp/output.xlsx")
        assert job.completed_at is not None


class TestRunDriftCheck:
    """Tests for _run_drift_check standalone function."""

    @pytest.mark.asyncio
    async def test_drift_check_no_config_file(self, db_session_factory: Any) -> None:
        """Test _run_drift_check uses defaults when config file missing."""
        from unittest.mock import AsyncMock, patch

        from energy_forecast.serving.job_manager import _run_drift_check

        mock_email = MagicMock()

        # Patch check_model_drift at its source module (local import)
        with patch(
            "energy_forecast.monitoring.drift_detector.check_model_drift",
            new_callable=AsyncMock,
            return_value=[],
        ):
            # Should complete without error
            await _run_drift_check(db_session_factory, mock_email)

    @pytest.mark.asyncio
    async def test_drift_check_disabled(self, db_session_factory: Any) -> None:
        """Test _run_drift_check returns early when drift is disabled."""
        from unittest.mock import AsyncMock, patch

        from energy_forecast.serving.job_manager import _run_drift_check

        mock_email = MagicMock()

        # Provide a YAML-like config with enabled=false
        yaml_data = {"drift_detection": {"enabled": False}}

        with (
            patch("builtins.open", create=True),
            patch(
                "energy_forecast.serving.job_manager.Path.exists",
                return_value=True,
            ) as _p,
            patch("yaml.safe_load", return_value=yaml_data),
            patch(
                "energy_forecast.monitoring.drift_detector.check_model_drift",
                new_callable=AsyncMock,
            ) as mock_check,
        ):
            await _run_drift_check(db_session_factory, mock_email)
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_drift_check_exception_nonfatal(self, db_session_factory: Any) -> None:
        """Test _run_drift_check swallows exceptions gracefully."""
        from unittest.mock import AsyncMock, patch

        from energy_forecast.serving.job_manager import _run_drift_check

        mock_email = MagicMock()

        with patch(
            "energy_forecast.monitoring.drift_detector.check_model_drift",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB connection lost"),
        ):
            # Should not raise
            await _run_drift_check(db_session_factory, mock_email)

    @pytest.mark.asyncio
    async def test_drift_check_with_alerts_logs_warning(self, db_session_factory: Any) -> None:
        """Test _run_drift_check logs warnings when alerts are returned."""
        from unittest.mock import AsyncMock, patch

        from energy_forecast.serving.job_manager import _run_drift_check

        mock_alert = MagicMock()
        mock_alert.severity = "warning"
        mock_alert.message = "MAPE increasing"
        mock_alert.alert_type = "mape_threshold"
        mock_alert.current_value = 5.5
        mock_alert.threshold = 5.0

        mock_email = MagicMock()

        with patch(
            "energy_forecast.monitoring.drift_detector.check_model_drift",
            new_callable=AsyncMock,
            return_value=[mock_alert],
        ):
            # Should run through without error; warning is severity,
            # email_on_warning defaults to False in DriftConfig
            await _run_drift_check(db_session_factory, mock_email)


# ---------------------------------------------------------------------------
# Queue tests
# ---------------------------------------------------------------------------


class TestJobQueue:
    """Tests for asyncio.Queue-based job processing."""

    def test_enqueue_returns_position(self) -> None:
        """Test enqueue returns 1-based queue position."""
        jm = JobManager(max_queue_size=5)
        pos = jm.enqueue(
            job_id="q1",
            excel_path="/tmp/a.xlsx",
            email="a@a.com",
            file_stem="s1",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=None,
            prediction_service=MagicMock(),
            file_service=MagicMock(),
            email_service=MagicMock(),
            is_db_mode=False,
            job_ref=MagicMock(),
        )
        assert pos == 1

    def test_enqueue_full_raises(self) -> None:
        """Test enqueue raises JobQueueFullError when queue is at max."""
        jm = JobManager(max_queue_size=2)
        for i in range(2):
            jm.enqueue(
                job_id=f"q{i}",
                excel_path="/tmp/a.xlsx",
                email="a@a.com",
                file_stem="s",
                created_at=datetime.now(tz=TZ_ISTANBUL),
                session_factory=None,
                prediction_service=MagicMock(),
                file_service=MagicMock(),
                email_service=MagicMock(),
                is_db_mode=False,
                job_ref=MagicMock(),
            )
        with pytest.raises(JobQueueFullError, match="Queue is full"):
            jm.enqueue(
                job_id="q_overflow",
                excel_path="/tmp/a.xlsx",
                email="a@a.com",
                file_stem="s",
                created_at=datetime.now(tz=TZ_ISTANBUL),
                session_factory=None,
                prediction_service=MagicMock(),
                file_service=MagicMock(),
                email_service=MagicMock(),
                is_db_mode=False,
                job_ref=MagicMock(),
            )

    def test_queue_size_property(self) -> None:
        """Test queue_size reflects enqueued items."""
        jm = JobManager(max_queue_size=5)
        assert jm.queue_size == 0
        jm.enqueue(
            job_id="q1",
            excel_path="/tmp/a.xlsx",
            email="a@a.com",
            file_stem="s",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            session_factory=None,
            prediction_service=MagicMock(),
            file_service=MagicMock(),
            email_service=MagicMock(),
            is_db_mode=False,
            job_ref=MagicMock(),
        )
        assert jm.queue_size == 1

    @pytest.mark.asyncio
    async def test_worker_processes_sequentially(self) -> None:
        """Test worker loop processes jobs one at a time in order."""
        import asyncio

        jm = JobManager(max_queue_size=5)
        order: list[str] = []

        job1 = Job(
            id="seq1",
            email="a@a.com",
            excel_path=Path("/tmp/a.xlsx"),
            file_stem="s1",
        )
        job2 = Job(
            id="seq2",
            email="b@b.com",
            excel_path=Path("/tmp/b.xlsx"),
            file_stem="s2",
        )
        jm._jobs[job1.id] = job1
        jm._jobs[job2.id] = job2

        mock_pred = MagicMock()
        pred_df = pd.DataFrame(
            {"prediction": [1000]},
            index=pd.date_range("2025-01-01", periods=1, freq="h"),
        )
        mock_pred.run_prediction.return_value = pred_df

        mock_file = MagicMock()
        mock_file.create_output_xlsx.return_value = Path("/tmp/out.xlsx")

        mock_email = MagicMock()
        mock_email.send_with_retry.return_value = (True, 1, None)

        # Patch process_job_in_memory to track order
        original_process = jm.process_job_in_memory

        async def tracked_process(job: Job, **kwargs: Any) -> None:
            order.append(job.id)
            await original_process(job=job, **kwargs)

        jm.process_job_in_memory = tracked_process  # type: ignore[assignment]

        jm.start_worker()

        jm.enqueue(
            job_id=job1.id,
            excel_path="/tmp/a.xlsx",
            email="a@a.com",
            file_stem="s1",
            created_at=job1.created_at,
            session_factory=None,
            prediction_service=mock_pred,
            file_service=mock_file,
            email_service=mock_email,
            is_db_mode=False,
            job_ref=job1,
        )
        jm.enqueue(
            job_id=job2.id,
            excel_path="/tmp/b.xlsx",
            email="b@b.com",
            file_stem="s2",
            created_at=job2.created_at,
            session_factory=None,
            prediction_service=mock_pred,
            file_service=mock_file,
            email_service=mock_email,
            is_db_mode=False,
            job_ref=job2,
        )

        # Wait for queue to drain
        await asyncio.wait_for(jm._queue.join(), timeout=5.0)
        await jm.stop_worker()

        assert order == ["seq1", "seq2"]

    @pytest.mark.asyncio
    async def test_start_stop_worker(self) -> None:
        """Test worker start and stop lifecycle."""
        jm = JobManager(max_queue_size=3)
        assert jm._worker_task is None

        jm.start_worker()
        assert jm._worker_task is not None
        assert not jm._worker_task.done()

        await jm.stop_worker()
        assert jm._worker_task is None

    def test_stats_include_queued(self) -> None:
        """Test get_stats_in_memory includes queued status."""
        jm = JobManager(max_queue_size=5)
        job = jm.create_job_in_memory("a@a.com", Path("/tmp/a.xlsx"), "s1")
        job.status = JobStatus.RUNNING
        jm._active_job_id = job.id

        job2 = jm.create_job_in_memory("b@b.com", Path("/tmp/b.xlsx"), "s2")
        assert job2.status == JobStatus.QUEUED

        stats = jm.get_stats_in_memory()
        assert stats["queued"] == 1
        assert stats["running"] == 1


class TestUpdateProgressDb:
    """Tests for _update_progress_db static method."""

    @pytest.mark.asyncio
    async def test_update_progress_db_updates_job(
        self, db_session_factory: Any,
    ) -> None:
        """_update_progress_db writes progress to DB via real SQLite."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        # Seed a job
        async with db_session_factory() as session:
            session.add(JobModel(
                id="prog_test1",
                email="t@t.com",
                status="running",
                excel_path="/tmp/t.xlsx",
                file_stem="stem",
                email_status="pending",
            ))
            await session.commit()

        # Act
        await JobManager._update_progress_db(
            db_session_factory, "prog_test1", "Step 2/8",
        )

        # Assert
        async with db_session_factory() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id("prog_test1")
            assert job is not None
            assert job.progress == "Step 2/8"


class TestUpdateStatusDb:
    """Tests for _update_status_db static method."""

    @pytest.mark.asyncio
    async def test_update_status_db_sets_status(
        self, db_session_factory: Any,
    ) -> None:
        """_update_status_db changes job status in DB."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            session.add(JobModel(
                id="stat_test1",
                email="t@t.com",
                status="running",
                excel_path="/tmp/t.xlsx",
                file_stem="stem",
                email_status="pending",
            ))
            await session.commit()

        await JobManager._update_status_db(
            db_session_factory, "stat_test1", "completed",
        )

        async with db_session_factory() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id("stat_test1")
            assert job is not None
            assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_update_status_db_with_result_path(
        self, db_session_factory: Any,
    ) -> None:
        """_update_status_db stores result_path in DB."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            session.add(JobModel(
                id="stat_test2",
                email="t@t.com",
                status="running",
                excel_path="/tmp/t.xlsx",
                file_stem="stem",
                email_status="pending",
            ))
            await session.commit()

        await JobManager._update_status_db(
            db_session_factory, "stat_test2", "completed",
            result_path="/tmp/out.xlsx",
        )

        async with db_session_factory() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id("stat_test2")
            assert job is not None
            assert job.result_path == "/tmp/out.xlsx"

    @pytest.mark.asyncio
    async def test_update_status_db_with_error(
        self, db_session_factory: Any,
    ) -> None:
        """_update_status_db stores error message in DB."""
        from energy_forecast.db.models import JobModel
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with db_session_factory() as session:
            session.add(JobModel(
                id="stat_test3",
                email="t@t.com",
                status="running",
                excel_path="/tmp/t.xlsx",
                file_stem="stem",
                email_status="pending",
            ))
            await session.commit()

        await JobManager._update_status_db(
            db_session_factory, "stat_test3", "failed",
            error="Something broke",
        )

        async with db_session_factory() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id("stat_test3")
            assert job is not None
            assert job.status == "failed"
            assert job.error == "Something broke"
