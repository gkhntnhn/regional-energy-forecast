"""Prediction repository — CRUD operations for PredictionModel."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import PredictionModel


class PredictionRepository:
    """Data access layer for predictions table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(self, predictions: list[dict[str, Any]]) -> int:
        """Insert multiple prediction records. Returns row count."""
        models = [PredictionModel(**p) for p in predictions]
        self._session.add_all(models)
        await self._session.flush()
        return len(models)

    async def get_by_job_id(self, job_id: str) -> list[PredictionModel]:
        """Return all predictions for a given job."""
        stmt = (
            select(PredictionModel)
            .where(PredictionModel.job_id == job_id)
            .order_by(PredictionModel.forecast_dt)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_ensemble_by_job_id(
        self, job_id: str
    ) -> list[PredictionModel]:
        """Return only ensemble predictions for a given job."""
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.job_id == job_id,
                PredictionModel.model_source == "ensemble",
            )
            .order_by(PredictionModel.forecast_dt)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
