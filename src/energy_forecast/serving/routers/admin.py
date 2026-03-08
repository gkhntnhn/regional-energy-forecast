"""Admin API router — analytics endpoints for the admin dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from loguru import logger

from energy_forecast.db.repositories.analytics_repo import AnalyticsRepository
from energy_forecast.db.repositories.job_repo import JobRepository
from energy_forecast.db.repositories.prediction_repo import PredictionRepository

admin_router = APIRouter(prefix="/admin", tags=["admin"])


def _get_session_factory(request: Request) -> Any:
    """Extract session factory from app state, or None if DB disabled."""
    if not getattr(request.app.state, "use_db", False):
        return None
    return request.app.state.session_factory


# ---------------------------------------------------------------------------
# Admin Dashboard page
# ---------------------------------------------------------------------------


@admin_router.get("/", include_in_schema=False)
async def admin_dashboard() -> FileResponse:
    """Serve the admin dashboard HTML."""
    from pathlib import Path

    return FileResponse(
        Path(__file__).parent.parent / "static" / "admin.html",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# MAPE Analytics (4.1)
# ---------------------------------------------------------------------------


@admin_router.get("/analytics/mape/daily")
async def get_daily_mape(
    request: Request, days: int = 30
) -> list[dict[str, Any]]:
    """Daily production MAPE trend."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_daily_mape(days)


@admin_router.get("/analytics/mape/weekly")
async def get_weekly_mape(
    request: Request, weeks: int = 12
) -> list[dict[str, Any]]:
    """Weekly MAPE trend with T vs T+1 breakdown."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_weekly_mape(weeks)


@admin_router.get("/analytics/mape/hourly")
async def get_hourly_mape(request: Request) -> list[dict[str, Any]]:
    """Hourly MAPE pattern."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_hourly_mape()


# ---------------------------------------------------------------------------
# Model Analytics (4.2)
# ---------------------------------------------------------------------------


@admin_router.get("/analytics/models/mape")
async def get_per_model_mape(
    request: Request, days: int = 30
) -> list[dict[str, Any]]:
    """Per-model production MAPE comparison."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_per_model_mape(days)


@admin_router.get("/analytics/models/hourly")
async def get_hourly_model_performance(
    request: Request,
) -> list[dict[str, Any]]:
    """Hourly model MAPE matrix (heatmap data)."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_hourly_model_performance()


@admin_router.get("/analytics/models/comparison")
async def get_model_comparison(
    request: Request, days: int = 30
) -> list[dict[str, Any]]:
    """Ensemble vs individual model stats."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_model_comparison_stats(days)


# ---------------------------------------------------------------------------
# Weather Analytics (4.3)
# ---------------------------------------------------------------------------


@admin_router.get("/analytics/weather/horizon")
async def get_weather_horizon(
    request: Request, weeks: int = 8
) -> list[dict[str, Any]]:
    """T vs T+1 weather forecast accuracy."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_weather_horizon_accuracy(weeks)


@admin_router.get("/analytics/weather/variables")
async def get_weather_variables(
    request: Request,
) -> list[dict[str, Any]]:
    """Per-variable weather forecast MAE."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_weather_variable_accuracy()


# ---------------------------------------------------------------------------
# Feature Analytics (4.4)
# ---------------------------------------------------------------------------


@admin_router.get("/analytics/features/trend")
async def get_feature_trend(
    request: Request, days: int = 30
) -> list[dict[str, Any]]:
    """Feature importance change over time."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_feature_importance_trend(days)


# ---------------------------------------------------------------------------
# EPIAS Analytics (4.7)
# ---------------------------------------------------------------------------


@admin_router.get("/analytics/epias/accuracy")
async def get_epias_accuracy(
    request: Request, days: int = 30
) -> list[dict[str, Any]]:
    """EPIAS Load_Forecast vs Real_Time_Consumption accuracy."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_epias_forecast_accuracy(days)


# ---------------------------------------------------------------------------
# Job History
# ---------------------------------------------------------------------------


@admin_router.get("/jobs/history")
async def get_job_history(
    request: Request, page: int = 1, size: int = 20
) -> dict[str, Any]:
    """Paginated job history."""
    sf = _get_session_factory(request)
    if sf is None:
        return {"total": 0, "page": page, "size": size, "pages": 0, "jobs": []}
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_job_history(page, size)


@admin_router.get("/jobs/{job_id}/details")
async def get_job_details(
    request: Request, job_id: str
) -> dict[str, Any]:
    """Detailed job info with predictions and weather snapshots."""
    sf = _get_session_factory(request)
    if sf is None:
        return {"error": "Database not configured"}
    async with sf() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)
        if job is None:
            return {"error": "Job not found"}

        pred_repo = PredictionRepository(session)
        preds = await pred_repo.get_by_job_id(job_id)

        return {
            "id": job.id,
            "email": job.email[:3] + "***" if job.email else "",
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "completed_at": (
                job.completed_at.isoformat() if job.completed_at else None
            ),
            "metadata": job.metadata_,
            "epias_snapshot": job.epias_snapshot,
            "predictions": [
                {
                    "forecast_dt": p.forecast_dt.isoformat(),
                    "consumption_mwh": p.consumption_mwh,
                    "period": p.period,
                    "model_source": p.model_source,
                    "actual_mwh": p.actual_mwh,
                    "error_pct": p.error_pct,
                }
                for p in preds
            ],
        }


# ---------------------------------------------------------------------------
# Model Runs (Faz 3 data)
# ---------------------------------------------------------------------------


@admin_router.get("/models/runs")
async def get_model_runs(
    request: Request, model_type: str | None = None
) -> list[dict[str, Any]]:
    """Training history from model_runs table."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_model_runs(model_type)


@admin_router.get("/models/promoted")
async def get_promoted_models(
    request: Request,
) -> list[dict[str, Any]]:
    """Currently promoted models."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_promoted_models()


@admin_router.get("/models/drift/status")
async def get_drift_status(
    request: Request,
) -> list[dict[str, Any]]:
    """Recent drift alerts."""
    sf = _get_session_factory(request)
    if sf is None:
        return []
    async with sf() as session:
        repo = AnalyticsRepository(session)
        return await repo.get_drift_status()


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


@admin_router.get("/system/health")
async def system_health(request: Request) -> dict[str, Any]:
    """Detailed system health check."""
    sf = _get_session_factory(request)
    db_ok = False
    if sf is not None:
        try:
            from sqlalchemy import text

            async with sf() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception as e:
            logger.warning("DB health check failed: {}", e)

    prediction_service = getattr(request.app.state, "prediction_service", None)
    models_loaded = (
        prediction_service.is_ready if prediction_service else False
    )

    return {
        "database": "connected" if db_ok else "disconnected",
        "models_loaded": models_loaded,
        "model_info": (
            prediction_service.get_model_info()
            if prediction_service and models_loaded
            else {}
        ),
    }
