"""Tests for repository CRUD operations."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import JobModel
from energy_forecast.db.repositories.job_repo import JobRepository
from energy_forecast.db.repositories.prediction_repo import PredictionRepository
from energy_forecast.utils import TZ_ISTANBUL


@pytest.fixture
def _seed_job(db_session: AsyncSession) -> JobModel:
    """Create a job for testing (not committed)."""
    job = JobModel(
        id="repo_test_01",
        email="test@example.com",
        status="pending",
        excel_path="/tmp/test.xlsx",
        file_stem="stem",
        email_status="pending",
    )
    db_session.add(job)
    return job


class TestJobRepository:
    """Tests for JobRepository."""

    @pytest.mark.asyncio
    async def test_create(self, db_session: AsyncSession) -> None:
        """Test creating a job via repository."""
        repo = JobRepository(db_session)
        job = await repo.create(
            {
                "id": "create_test1",
                "email": "test@example.com",
                "excel_path": "/tmp/test.xlsx",
                "file_stem": "stem",
                "status": "pending",
                "email_status": "pending",
            }
        )
        assert job.id == "create_test1"
        assert job.status == "pending"

    @pytest.mark.asyncio
    async def test_get_by_id(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test fetching a job by ID."""
        await db_session.flush()
        repo = JobRepository(db_session)
        job = await repo.get_by_id("repo_test_01")
        assert job is not None
        assert job.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, db_session: AsyncSession
    ) -> None:
        """Test fetching nonexistent job."""
        repo = JobRepository(db_session)
        job = await repo.get_by_id("nonexistent")
        assert job is None

    @pytest.mark.asyncio
    async def test_get_active_job(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test finding active job."""
        _seed_job.status = "running"
        await db_session.flush()

        repo = JobRepository(db_session)
        active = await repo.get_active_job()
        assert active is not None
        assert active.id == "repo_test_01"

    @pytest.mark.asyncio
    async def test_get_active_job_none(
        self, db_session: AsyncSession
    ) -> None:
        """Test no active job when all are completed."""
        repo = JobRepository(db_session)
        active = await repo.get_active_job()
        assert active is None

    @pytest.mark.asyncio
    async def test_update_status(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test status update."""
        await db_session.flush()
        repo = JobRepository(db_session)
        await repo.update_status("repo_test_01", "completed")
        await db_session.flush()

        job = await repo.get_by_id("repo_test_01")
        assert job is not None
        assert job.status == "completed"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_update_progress(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test progress update."""
        await db_session.flush()
        repo = JobRepository(db_session)
        await repo.update_progress("repo_test_01", "Step 2 of 5")

        job = await repo.get_by_id("repo_test_01")
        assert job is not None
        assert job.progress == "Step 2 of 5"

    @pytest.mark.asyncio
    async def test_update_metadata(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test metadata update."""
        await db_session.flush()
        repo = JobRepository(db_session)
        await repo.update_metadata(
            "repo_test_01",
            {
                "metadata": {"latency_ms": 5000},
                "excel_hash": "abc123",
            },
        )

        job = await repo.get_by_id("repo_test_01")
        assert job is not None
        assert job.excel_hash == "abc123"

    @pytest.mark.asyncio
    async def test_update_email_status(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test email status update."""
        await db_session.flush()
        repo = JobRepository(db_session)
        now = datetime.now(tz=TZ_ISTANBUL)
        await repo.update_email_status(
            "repo_test_01", "sent", attempts=1, sent_at=now
        )

        job = await repo.get_by_id("repo_test_01")
        assert job is not None
        assert job.email_status == "sent"
        assert job.email_attempts == 1

    @pytest.mark.asyncio
    async def test_get_all(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test listing all jobs."""
        await db_session.flush()
        repo = JobRepository(db_session)
        jobs = await repo.get_all()
        assert len(jobs) >= 1

    @pytest.mark.asyncio
    async def test_get_stats(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test job statistics."""
        await db_session.flush()
        repo = JobRepository(db_session)
        stats = await repo.get_stats()
        assert "pending" in stats
        assert stats["pending"] >= 1


class TestPredictionRepository:
    """Tests for PredictionRepository."""

    @pytest.mark.asyncio
    async def test_bulk_create(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test bulk inserting predictions."""
        await db_session.flush()
        repo = PredictionRepository(db_session)
        rows = [
            {
                "job_id": "repo_test_01",
                "forecast_dt": datetime(2026, 3, 7, h, 0, tzinfo=TZ_ISTANBUL),
                "consumption_mwh": 1000.0 + h * 10,
                "period": "day_ahead",
                "model_source": "ensemble",
            }
            for h in range(24)
        ]
        count = await repo.bulk_create(rows)
        assert count == 24

    @pytest.mark.asyncio
    async def test_get_by_job_id(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test fetching predictions by job ID."""
        await db_session.flush()
        repo = PredictionRepository(db_session)
        await repo.bulk_create(
            [
                {
                    "job_id": "repo_test_01",
                    "forecast_dt": datetime(
                        2026, 3, 7, 0, 0, tzinfo=TZ_ISTANBUL
                    ),
                    "consumption_mwh": 1234.5,
                    "period": "day_ahead",
                    "model_source": "ensemble",
                }
            ]
        )
        results = await repo.get_by_job_id("repo_test_01")
        assert len(results) == 1
        assert results[0].consumption_mwh == 1234.5

    @pytest.mark.asyncio
    async def test_get_ensemble_by_job_id(
        self, db_session: AsyncSession, _seed_job: JobModel
    ) -> None:
        """Test filtering ensemble-only predictions."""
        await db_session.flush()
        repo = PredictionRepository(db_session)
        await repo.bulk_create(
            [
                {
                    "job_id": "repo_test_01",
                    "forecast_dt": datetime(
                        2026, 3, 7, 0, 0, tzinfo=TZ_ISTANBUL
                    ),
                    "consumption_mwh": 1000.0,
                    "period": "day_ahead",
                    "model_source": "catboost",
                },
                {
                    "job_id": "repo_test_01",
                    "forecast_dt": datetime(
                        2026, 3, 7, 0, 0, tzinfo=TZ_ISTANBUL
                    ),
                    "consumption_mwh": 1100.0,
                    "period": "day_ahead",
                    "model_source": "ensemble",
                },
            ]
        )
        ensemble = await repo.get_ensemble_by_job_id("repo_test_01")
        assert len(ensemble) == 1
        assert ensemble[0].model_source == "ensemble"
