"""Pydantic request/response schemas for API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field


class ForecastType(StrEnum):
    """Forecast type selection."""

    DAY_AHEAD = "day_ahead"
    DAY_AHEAD_AND_INTRADAY = "day_ahead_and_intraday"


class JobStatus(StrEnum):
    """Job processing status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Prediction request data (email comes from form, file from upload)."""

    email: EmailStr = Field(..., description="Email address to send results")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class PredictionItem(BaseModel):
    """Single hourly prediction."""

    datetime: str
    consumption_mwh: float
    period: str


class PredictionStatistics(BaseModel):
    """Prediction summary statistics."""

    count: int
    mean: float
    min: float
    max: float
    std: float
    date_start: str
    date_end: str


class ForecastMetadata(BaseModel):
    """Forecast metadata."""

    model: str = "ensemble_v1"
    weights: dict[str, float] = Field(default_factory=dict)
    last_data_point: str
    forecast_start: str
    forecast_end: str


class ForecastResponse(BaseModel):
    """Full forecast API response."""

    success: bool
    forecast_type: ForecastType
    predictions: list[PredictionItem]
    metadata: ForecastMetadata
    statistics: PredictionStatistics
    download_url: str | None = None


class JobResponse(BaseModel):
    """Response when a job is submitted."""

    job_id: str
    status: JobStatus
    message: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: JobStatus
    progress: str | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str


class ErrorResponse(BaseModel):
    """Error response for API errors."""

    success: bool = False
    error: str
    detail: str | None = None
