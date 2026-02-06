"""Pydantic request/response schemas for API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ForecastType(StrEnum):
    """Forecast type selection."""

    DAY_AHEAD = "day_ahead"
    DAY_AHEAD_AND_INTRADAY = "day_ahead_and_intraday"


class PredictionItem(BaseModel):
    """Single hourly prediction."""

    datetime: str
    consumption_mwh: float
    period: str


class ForecastResponse(BaseModel):
    """Full forecast API response."""

    success: bool
    forecast_type: ForecastType
    predictions: list[PredictionItem]
    metadata: dict[str, object]
    statistics: dict[str, float]
    download_url: str | None = None
