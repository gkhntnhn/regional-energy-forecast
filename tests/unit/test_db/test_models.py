"""Tests for ORM models."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import JobModel, PredictionModel
from energy_forecast.utils import TZ_ISTANBUL


class TestJobModel:
    """Tests for JobModel ORM."""

    @pytest.mark.asyncio
    async def test_create_job(self, db_session: AsyncSession) -> None:
        """Test creating a job record."""
        job = JobModel(
            id="test123456",
            email="test@example.com",
            status="pending",
            excel_path="/tmp/test.xlsx",
            file_stem="01-03-2026_12-00-00",
            email_status="pending",
        )
        db_session.add(job)
        await db_session.flush()

        result = await db_session.get(JobModel, "test123456")
        assert result is not None
        assert result.email == "test@example.com"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_job_defaults(self, db_session: AsyncSession) -> None:
        """Test job default values."""
        job = JobModel(
            id="default12345",
            email="test@example.com",
            excel_path="/tmp/test.xlsx",
            file_stem="stem",
        )
        db_session.add(job)
        await db_session.flush()

        result = await db_session.get(JobModel, "default12345")
        assert result is not None
        assert result.progress is None
        assert result.error is None
        assert result.result_path is None
        assert result.completed_at is None

    @pytest.mark.asyncio
    async def test_job_with_metadata(self, db_session: AsyncSession) -> None:
        """Test job with JSONB metadata fields."""
        job = JobModel(
            id="meta12345678",
            email="test@example.com",
            excel_path="/tmp/test.xlsx",
            file_stem="stem",
            metadata_={"latency_ms": 1234},
            config_snapshot={"method": "stacking"},
            model_versions={"catboost": "v1"},
        )
        db_session.add(job)
        await db_session.flush()

        result = await db_session.get(JobModel, "meta12345678")
        assert result is not None
        # SQLite stores JSONB as text, values are preserved
        assert result.model_versions is not None


class TestPredictionModel:
    """Tests for PredictionModel ORM."""

    @pytest.mark.asyncio
    async def test_create_prediction(self, db_session: AsyncSession) -> None:
        """Test creating a prediction linked to a job."""
        job = JobModel(
            id="jobpred12345",
            email="test@example.com",
            excel_path="/tmp/test.xlsx",
            file_stem="stem",
        )
        db_session.add(job)
        await db_session.flush()

        pred = PredictionModel(
            job_id="jobpred12345",
            forecast_dt=datetime(2026, 3, 7, 0, 0, tzinfo=TZ_ISTANBUL),
            consumption_mwh=1234.5,
            period="day_ahead",
            model_source="ensemble",
        )
        db_session.add(pred)
        await db_session.flush()

        assert pred.id is not None
        assert pred.actual_mwh is None  # Phase 2 column

    @pytest.mark.asyncio
    async def test_cascade_delete(self, db_session: AsyncSession) -> None:
        """Test predictions are deleted when job is deleted."""
        job = JobModel(
            id="cascade12345",
            email="test@example.com",
            excel_path="/tmp/test.xlsx",
            file_stem="stem",
        )
        db_session.add(job)
        await db_session.flush()

        pred = PredictionModel(
            job_id="cascade12345",
            forecast_dt=datetime(2026, 3, 7, 0, 0, tzinfo=TZ_ISTANBUL),
            consumption_mwh=1000.0,
            period="intraday",
        )
        db_session.add(pred)
        await db_session.flush()

        await db_session.delete(job)
        await db_session.flush()

        # Prediction should be cascade-deleted
        from sqlalchemy import select

        result = await db_session.execute(
            select(PredictionModel).where(
                PredictionModel.job_id == "cascade12345"
            )
        )
        assert result.scalars().all() == []
