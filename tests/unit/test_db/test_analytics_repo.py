"""Tests for AnalyticsRepository — admin dashboard queries."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import (
    AuditLogModel,
    JobModel,
    ModelRunModel,
    PredictionModel,
    WeatherSnapshotModel,
)
from energy_forecast.db.repositories.analytics_repo import AnalyticsRepository
from energy_forecast.utils import TZ_ISTANBUL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prediction(
    job_id: str,
    hour: int,
    model_source: str = "ensemble",
    consumption: float = 1200.0,
    actual: float | None = 1250.0,
    day_offset: int = 0,
) -> PredictionModel:
    dt = datetime(2026, 3, 1, hour, tzinfo=TZ_ISTANBUL) + timedelta(
        days=day_offset
    )
    error = (
        abs(consumption - actual) / actual * 100
        if actual is not None and actual > 0
        else None
    )
    return PredictionModel(
        job_id=job_id,
        forecast_dt=dt,
        consumption_mwh=consumption,
        period="day_ahead",
        model_source=model_source,
        actual_mwh=actual,
        error_pct=error,
        matched_at=datetime.now(tz=TZ_ISTANBUL) if actual else None,
        created_at=dt,
    )


def _make_job(
    job_id: str,
    day_offset: int = 0,
    metadata: dict[str, Any] | None = None,
    epias_snapshot: dict[str, Any] | None = None,
) -> JobModel:
    dt = datetime(2026, 3, 1, 10, tzinfo=TZ_ISTANBUL) + timedelta(
        days=day_offset
    )
    return JobModel(
        id=job_id,
        email="test@test.com",
        status="completed",
        excel_path="/tmp/test.xlsx",
        file_stem="test",
        created_at=dt,
        completed_at=dt,
        metadata_=metadata,
        epias_snapshot=epias_snapshot,
    )


def _make_weather(
    forecast_dt: datetime,
    is_actual: bool,
    job_id: str | None = None,
    temperature: float = 15.0,
    wind: float = 5.0,
    fetched_at: datetime | None = None,
) -> WeatherSnapshotModel:
    return WeatherSnapshotModel(
        job_id=job_id,
        forecast_dt=forecast_dt,
        fetched_at=fetched_at or forecast_dt,
        is_actual=is_actual,
        temperature_2m=temperature,
        wind_speed_10m=wind,
    )


# ---------------------------------------------------------------------------
# 4.1 — MAPE Trending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_daily_mape(db_session: AsyncSession) -> None:
    job = _make_job("j1")
    db_session.add(job)
    for h in range(24):
        db_session.add(
            _make_prediction("j1", h, consumption=1200.0, actual=1250.0)
        )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_daily_mape(days=30)
    assert len(result) == 1
    assert result[0]["day"] == "2026-03-01"
    assert result[0]["n_hours"] == 24
    assert result[0]["mape"] > 0


@pytest.mark.asyncio
async def test_get_daily_mape_empty(db_session: AsyncSession) -> None:
    repo = AnalyticsRepository(db_session)
    result = await repo.get_daily_mape(days=30)
    assert result == []


@pytest.mark.asyncio
async def test_get_weekly_mape(db_session: AsyncSession) -> None:
    job = _make_job("j1")
    db_session.add(job)
    for h in range(24):
        db_session.add(
            _make_prediction("j1", h, consumption=1200.0, actual=1250.0)
        )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_weekly_mape(weeks=12)
    assert len(result) >= 1
    assert "weekly_mape" in result[0]
    assert result[0]["n_hours"] == 24


@pytest.mark.asyncio
async def test_get_hourly_mape(db_session: AsyncSession) -> None:
    job = _make_job("j1")
    db_session.add(job)
    db_session.add(
        _make_prediction("j1", 10, consumption=1200.0, actual=1250.0)
    )
    db_session.add(
        _make_prediction("j1", 14, consumption=1300.0, actual=1250.0)
    )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_hourly_mape()
    assert len(result) == 2
    hours = {r["hour"] for r in result}
    assert hours == {10, 14}


# ---------------------------------------------------------------------------
# 4.2 — Per-Model Performance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_per_model_mape(db_session: AsyncSession) -> None:
    job = _make_job("j1")
    db_session.add(job)
    db_session.add(
        _make_prediction("j1", 10, model_source="ensemble", actual=1250.0)
    )
    db_session.add(
        _make_prediction("j1", 10, model_source="catboost", actual=1250.0)
    )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_per_model_mape(days=30)
    models = {r["model_source"] for r in result}
    assert "ensemble" in models
    assert "catboost" in models


@pytest.mark.asyncio
async def test_get_hourly_model_performance(
    db_session: AsyncSession,
) -> None:
    job = _make_job("j1")
    db_session.add(job)
    db_session.add(
        _make_prediction("j1", 10, model_source="ensemble", actual=1250.0)
    )
    db_session.add(
        _make_prediction("j1", 10, model_source="catboost", actual=1250.0)
    )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_hourly_model_performance()
    assert len(result) == 2
    assert all(r["hour"] == 10 for r in result)


@pytest.mark.asyncio
async def test_get_model_comparison_stats(
    db_session: AsyncSession,
) -> None:
    job = _make_job("j1")
    db_session.add(job)
    for h in range(24):
        db_session.add(_make_prediction("j1", h, model_source="ensemble"))
        db_session.add(_make_prediction("j1", h, model_source="catboost"))
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_model_comparison_stats(days=30)
    assert len(result) == 2
    assert all("avg_mape" in r for r in result)
    assert all("median_mape" in r for r in result)
    assert all("p95_mape" in r for r in result)


# ---------------------------------------------------------------------------
# 4.3 — Weather Horizon
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_weather_variable_accuracy(
    db_session: AsyncSession,
) -> None:
    dt = datetime(2026, 3, 1, 12, tzinfo=TZ_ISTANBUL)
    db_session.add(
        _make_weather(dt, is_actual=False, job_id=None, temperature=16.0)
    )
    db_session.add(
        _make_weather(dt, is_actual=True, temperature=15.0)
    )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_weather_variable_accuracy()
    assert len(result) >= 1
    temp_row = next(r for r in result if r["variable"] == "temperature_2m")
    assert temp_row["mae"] == 1.0


@pytest.mark.asyncio
async def test_get_weather_variable_accuracy_empty(
    db_session: AsyncSession,
) -> None:
    repo = AnalyticsRepository(db_session)
    result = await repo.get_weather_variable_accuracy()
    assert result == []


# ---------------------------------------------------------------------------
# 4.4 — Feature Importance Trend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_feature_importance_trend(
    db_session: AsyncSession,
) -> None:
    fi_data = [
        {"feature": "consumption_lag_48", "importance": 25.0},
        {"feature": "temperature_2m", "importance": 15.0},
    ]
    job = _make_job("j1", metadata={"feature_importance_top15": fi_data})
    db_session.add(job)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_feature_importance_trend(days=30, top_n=10)
    assert len(result) == 2
    features = {r["feature"] for r in result}
    assert "consumption_lag_48" in features


@pytest.mark.asyncio
async def test_get_feature_importance_no_metadata(
    db_session: AsyncSession,
) -> None:
    job = _make_job("j1")
    db_session.add(job)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_feature_importance_trend(days=30)
    assert result == []


# ---------------------------------------------------------------------------
# 4.7 — EPIAS Accuracy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_epias_forecast_accuracy(
    db_session: AsyncSession,
) -> None:
    snap = {
        "last_values": {
            "Load_Forecast": 4500.0,
            "Real_Time_Consumption": 4400.0,
        }
    }
    job = _make_job("j1", epias_snapshot=snap)
    db_session.add(job)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_epias_forecast_accuracy(days=30)
    assert len(result) == 1
    assert result[0]["epias_error_pct"] > 0


@pytest.mark.asyncio
async def test_get_epias_no_snapshot(
    db_session: AsyncSession,
) -> None:
    job = _make_job("j1")
    db_session.add(job)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_epias_forecast_accuracy(days=30)
    assert result == []


# ---------------------------------------------------------------------------
# Job History (paginated)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_history_pagination(
    db_session: AsyncSession,
) -> None:
    for i in range(5):
        db_session.add(_make_job(f"j{i}", day_offset=i))
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_job_history(page=1, size=2)
    assert result["total"] == 5
    assert result["pages"] == 3
    assert len(result["jobs"]) == 2


@pytest.mark.asyncio
async def test_get_job_history_empty(
    db_session: AsyncSession,
) -> None:
    repo = AnalyticsRepository(db_session)
    result = await repo.get_job_history()
    assert result["total"] == 0
    assert result["jobs"] == []


# ---------------------------------------------------------------------------
# Model Runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_runs(db_session: AsyncSession) -> None:
    run = ModelRunModel(
        model_type="catboost",
        status="completed",
        val_mape=2.5,
        test_mape=2.8,
        n_trials=50,
        n_splits=12,
    )
    db_session.add(run)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_model_runs(model_type="catboost")
    assert len(result) == 1
    assert result[0]["val_mape"] == 2.5


@pytest.mark.asyncio
async def test_get_model_runs_filter(
    db_session: AsyncSession,
) -> None:
    db_session.add(
        ModelRunModel(model_type="catboost", status="completed")
    )
    db_session.add(
        ModelRunModel(model_type="prophet", status="completed")
    )
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_model_runs(model_type="catboost")
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_promoted_models(db_session: AsyncSession) -> None:
    run = ModelRunModel(
        model_type="catboost",
        status="completed",
        is_promoted=True,
        promoted_at=datetime.now(tz=TZ_ISTANBUL),
        model_path="/models/catboost/model.cbm",
    )
    db_session.add(run)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_promoted_models()
    assert len(result) == 1
    assert result[0]["model_type"] == "catboost"


# ---------------------------------------------------------------------------
# Drift Status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_drift_status(db_session: AsyncSession) -> None:
    log = AuditLogModel(
        action="drift_mape_threshold",
        details={"mape": 5.5, "threshold": 5.0},
    )
    db_session.add(log)
    await db_session.flush()

    repo = AnalyticsRepository(db_session)
    result = await repo.get_drift_status()
    assert len(result) == 1
    assert result[0]["action"] == "drift_mape_threshold"
