"""Tests for serving schemas."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from energy_forecast.serving.schemas import (
    ErrorResponse,
    ForecastType,
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    PredictionItem,
    PredictionStatistics,
    PredictRequest,
)
from energy_forecast.utils import TZ_ISTANBUL


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"


class TestForecastType:
    """Tests for ForecastType enum."""

    def test_forecast_types(self) -> None:
        """Test forecast type values."""
        assert ForecastType.DAY_AHEAD == "day_ahead"
        assert ForecastType.DAY_AHEAD_AND_INTRADAY == "day_ahead_and_intraday"


class TestPredictRequest:
    """Tests for PredictRequest model."""

    def test_valid_email(self) -> None:
        """Test valid email is accepted."""
        req = PredictRequest(email="test@example.com")
        assert req.email == "test@example.com"

    def test_invalid_email(self) -> None:
        """Test invalid email is rejected."""
        with pytest.raises(ValidationError):
            PredictRequest(email="not-an-email")


class TestPredictionItem:
    """Tests for PredictionItem model."""

    def test_valid_item(self) -> None:
        """Test valid prediction item."""
        item = PredictionItem(
            datetime="2025-01-01T00:00:00",
            consumption_mwh=1234.5,
            period="day_ahead",
        )
        assert item.consumption_mwh == 1234.5
        assert item.period == "day_ahead"


class TestPredictionStatistics:
    """Tests for PredictionStatistics model."""

    def test_valid_stats(self) -> None:
        """Test valid statistics."""
        stats = PredictionStatistics(
            count=48,
            mean=1200.0,
            min=900.0,
            max=1500.0,
            std=150.0,
            date_start="2025-01-01T00:00:00",
            date_end="2025-01-02T23:00:00",
        )
        assert stats.count == 48
        assert stats.mean == 1200.0


class TestJobResponse:
    """Tests for JobResponse model."""

    def test_valid_response(self) -> None:
        """Test valid job response."""
        resp = JobResponse(
            job_id="abc123",
            status=JobStatus.PENDING,
            message="Job queued",
            created_at=datetime.now(tz=TZ_ISTANBUL),
        )
        assert resp.job_id == "abc123"
        assert resp.status == JobStatus.PENDING


class TestJobStatusResponse:
    """Tests for JobStatusResponse model."""

    def test_pending_response(self) -> None:
        """Test pending job status response."""
        resp = JobStatusResponse(
            job_id="abc123",
            status=JobStatus.PENDING,
            created_at=datetime.now(tz=TZ_ISTANBUL),
        )
        assert resp.progress is None
        assert resp.error is None
        assert resp.completed_at is None

    def test_completed_response(self) -> None:
        """Test completed job status response."""
        now = datetime.now(tz=TZ_ISTANBUL)
        resp = JobStatusResponse(
            job_id="abc123",
            status=JobStatus.COMPLETED,
            progress="Done",
            created_at=now,
            completed_at=now,
        )
        assert resp.status == JobStatus.COMPLETED
        assert resp.completed_at is not None


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_health(self) -> None:
        """Test valid health response."""
        resp = HealthResponse(
            status="ok",
            timestamp=datetime.now(tz=TZ_ISTANBUL),
            version="0.1.0",
        )
        assert resp.status == "ok"


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_response(self) -> None:
        """Test error response."""
        resp = ErrorResponse(error="ValidationError", detail="Invalid input")
        assert resp.success is False
        assert resp.error == "ValidationError"
