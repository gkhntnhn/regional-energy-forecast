"""Tests for prediction-actual matching in PredictionRepository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import JobModel, PredictionModel
from energy_forecast.db.repositories.prediction_repo import PredictionRepository

TZ = timezone(timedelta(hours=3))


@pytest_asyncio.fixture
async def _seed_job_with_predictions(
    db_session: AsyncSession,
) -> tuple[str, list[datetime]]:
    """Create a job with 24 unmatched predictions."""
    job = JobModel(
        id="match_test01",
        email="test@example.com",
        excel_path="/tmp/test.xlsx",
        file_stem="test",
        status="completed",
    )
    db_session.add(job)

    base_dt = datetime(2026, 3, 5, 0, 0, tzinfo=TZ)
    forecast_dts: list[datetime] = []
    for h in range(24):
        dt = base_dt + timedelta(hours=h)
        forecast_dts.append(dt)
        pred = PredictionModel(
            job_id=job.id,
            forecast_dt=dt,
            consumption_mwh=1000.0 + h * 10,
            period="day_ahead",
            model_source="ensemble",
        )
        db_session.add(pred)

    await db_session.flush()
    return job.id, forecast_dts


@pytest.mark.asyncio
async def test_match_predictions_basic(
    db_session: AsyncSession,
    _seed_job_with_predictions: tuple[str, list[datetime]],
) -> None:
    """Predictions are matched with actual consumption values."""
    job_id, forecast_dts = _seed_job_with_predictions
    repo = PredictionRepository(db_session)

    # Create consumption DataFrame covering those datetimes
    consumption_df = pd.DataFrame(
        {"consumption": [1050.0 + h * 10 for h in range(24)]},
        index=pd.DatetimeIndex(forecast_dts),
    )

    matched = await repo.match_predictions_with_actuals(consumption_df)
    assert matched == 24

    preds = await repo.get_ensemble_by_job_id(job_id)
    for pred in preds:
        assert pred.actual_mwh is not None
        assert pred.error_pct is not None
        assert pred.matched_at is not None


@pytest.mark.asyncio
async def test_match_error_pct_calculation(
    db_session: AsyncSession,
    _seed_job_with_predictions: tuple[str, list[datetime]],
) -> None:
    """Error percentage is computed as |actual - predicted| / actual * 100."""
    _, forecast_dts = _seed_job_with_predictions
    repo = PredictionRepository(db_session)

    # First prediction: predicted=1000, actual=1100 → error=(100/1100)*100 ≈ 9.09%
    consumption_df = pd.DataFrame(
        {"consumption": [1100.0]},
        index=pd.DatetimeIndex([forecast_dts[0]]),
    )

    await repo.match_predictions_with_actuals(consumption_df)

    preds = await repo.get_ensemble_by_job_id("match_test01")
    matched = [p for p in preds if p.actual_mwh is not None]
    assert len(matched) == 1
    assert matched[0].error_pct == pytest.approx(
        abs(1100.0 - 1000.0) / 1100.0 * 100, rel=1e-3,
    )


@pytest.mark.asyncio
async def test_match_skips_already_matched(
    db_session: AsyncSession,
    _seed_job_with_predictions: tuple[str, list[datetime]],
) -> None:
    """Already matched predictions are not re-matched."""
    _, forecast_dts = _seed_job_with_predictions
    repo = PredictionRepository(db_session)

    consumption_df = pd.DataFrame(
        {"consumption": [1050.0 + h * 10 for h in range(24)]},
        index=pd.DatetimeIndex(forecast_dts),
    )

    first = await repo.match_predictions_with_actuals(consumption_df)
    assert first == 24

    # Running again should match 0 (all already matched)
    second = await repo.match_predictions_with_actuals(consumption_df)
    assert second == 0


@pytest.mark.asyncio
async def test_match_empty_df_returns_zero(
    db_session: AsyncSession,
) -> None:
    """Empty consumption DataFrame returns 0 matches."""
    repo = PredictionRepository(db_session)
    empty_df = pd.DataFrame(columns=["consumption"])
    assert await repo.match_predictions_with_actuals(empty_df) == 0


@pytest.mark.asyncio
async def test_match_no_consumption_col_returns_zero(
    db_session: AsyncSession,
) -> None:
    """DataFrame without 'consumption' column returns 0."""
    repo = PredictionRepository(db_session)
    df = pd.DataFrame({"other": [1.0]}, index=pd.DatetimeIndex(["2026-03-05"]))
    assert await repo.match_predictions_with_actuals(df) == 0
