"""Prediction repository — CRUD operations for PredictionModel."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import PredictionModel
from energy_forecast.utils import TZ_ISTANBUL


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

    async def match_predictions_with_actuals(
        self, consumption_df: pd.DataFrame
    ) -> int:
        """Match previous predictions with actual consumption from Excel.

        When a new Excel is uploaded, it contains actual consumption up to
        T-1 23:00. This method finds unmatched ensemble predictions whose
        forecast_dt falls within the consumption data range and fills in
        actual_mwh, error_pct, and matched_at.

        Args:
            consumption_df: DataFrame with DatetimeIndex and 'consumption' column.

        Returns:
            Number of predictions matched.
        """
        if consumption_df.empty or "consumption" not in consumption_df.columns:
            return 0

        dt_max = consumption_df.index.max()
        if hasattr(dt_max, "tzinfo") and dt_max.tzinfo is None:
            dt_max = dt_max.tz_localize(TZ_ISTANBUL)

        # Find unmatched predictions within the consumption data range
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.actual_mwh.is_(None),
                PredictionModel.forecast_dt <= dt_max,
            )
        )
        result = await self._session.execute(stmt)

        # Normalize consumption index to naive for matching
        idx = consumption_df.index
        naive_index = (
            idx.tz_localize(None)  # type: ignore[attr-defined]
            if hasattr(idx, "tz") and idx.tz is not None
            else idx
        )
        consumption_naive = consumption_df.set_index(naive_index)

        now = datetime.now(tz=TZ_ISTANBUL)
        matched_count = 0
        for pred in result.scalars():
            pred_dt = pred.forecast_dt
            # Normalize to naive for comparison
            naive_dt = pred_dt.replace(tzinfo=None) if pred_dt.tzinfo else pred_dt

            actual_val: float | None = None
            if naive_dt in consumption_naive.index:
                raw = consumption_naive.at[naive_dt, "consumption"]
                actual_val = float(raw)  # type: ignore[arg-type]

            if actual_val is not None and actual_val > 0:
                pred.actual_mwh = actual_val
                pred.error_pct = (
                    abs(actual_val - pred.consumption_mwh) / actual_val * 100
                )
                pred.matched_at = now
                matched_count += 1

        await self._session.flush()
        return matched_count
