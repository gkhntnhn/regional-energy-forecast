"""Analytics repository — read-only queries for admin dashboard.

Dual-mode: PostgreSQL uses SQL aggregation, SQLite falls back to Python-side grouping.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import Integer, Numeric, String

from energy_forecast.db.models import (
    AuditLogModel,
    JobModel,
    ModelRunModel,
    PredictionModel,
    WeatherSnapshotModel,
)
from energy_forecast.utils import TZ_ISTANBUL


class AnalyticsRepository:
    """Read-only analytics queries for the admin dashboard.

    PostgreSQL mode: SQL aggregation (GROUP BY, func.avg, date_trunc).
    SQLite mode: ORM fetch + Python-side grouping (test compatibility).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def _is_pg(self) -> bool:
        """Check if the current database is PostgreSQL."""
        bind = self._session.get_bind()
        return bind.dialect.name == "postgresql"

    # ------------------------------------------------------------------
    # 4.1 — Production MAPE Trending
    # ------------------------------------------------------------------

    async def get_daily_mape(self, days: int = 30) -> list[dict[str, Any]]:
        """Daily production MAPE for ensemble predictions."""
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)

        if self._is_pg:
            day_trunc = func.date_trunc("day", PredictionModel.forecast_dt)
            day_label = func.to_char(day_trunc, "YYYY-MM-DD").label("day")
            stmt = (
                select(
                    day_label,
                    func.round(cast(func.avg(PredictionModel.error_pct), Numeric), 2).label(
                        "mape"
                    ),
                    func.round(cast(func.min(PredictionModel.error_pct), Numeric), 2).label(
                        "min_error"
                    ),
                    func.round(cast(func.max(PredictionModel.error_pct), Numeric), 2).label(
                        "max_error"
                    ),
                    func.count().label("n_hours"),
                )
                .where(
                    PredictionModel.actual_mwh.is_not(None),
                    PredictionModel.model_source == "ensemble",
                    PredictionModel.forecast_dt >= cutoff,
                    PredictionModel.error_pct.is_not(None),
                )
                .group_by(day_label)
                .order_by(day_label)
            )
            result = await self._session.execute(stmt)
            return [
                {
                    "day": row.day,
                    "mape": float(row.mape),
                    "min_error": float(row.min_error),
                    "max_error": float(row.max_error),
                    "n_hours": row.n_hours,
                }
                for row in result
            ]

        # SQLite fallback: Python-side grouping
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.actual_mwh.is_not(None),
                PredictionModel.model_source == "ensemble",
                PredictionModel.forecast_dt >= cutoff,
            )
            .order_by(PredictionModel.forecast_dt)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        daily: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            day = r.forecast_dt.strftime("%Y-%m-%d")
            if r.error_pct is not None:
                daily[day].append(r.error_pct)

        return [
            {
                "day": day,
                "mape": round(sum(errs) / len(errs), 2),
                "min_error": round(min(errs), 2),
                "max_error": round(max(errs), 2),
                "n_hours": len(errs),
            }
            for day, errs in sorted(daily.items())
        ]

    async def get_weekly_mape(self, weeks: int = 12) -> list[dict[str, Any]]:
        """Weekly MAPE trend with T vs T+1 breakdown."""
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(weeks=weeks)

        if self._is_pg:
            # PostgreSQL: use date_trunc('week') + EXTRACT for ISO week key
            iso_year = func.extract(
                "isoyear", PredictionModel.forecast_dt
            )
            iso_week = func.extract(
                "week", PredictionModel.forecast_dt
            )
            week_label = func.concat(
                cast(iso_year, String),
                "-W",
                func.lpad(cast(cast(iso_week, Integer), String), 2, "0"),
            )
            stmt = (
                select(
                    week_label.label("week"),
                    func.round(cast(func.avg(PredictionModel.error_pct), Numeric), 2).label(
                        "weekly_mape"
                    ),
                    func.count().label("n_hours"),
                    func.round(
                        cast(func.avg(
                            case(
                                (
                                    PredictionModel.period == "intraday",
                                    PredictionModel.error_pct,
                                ),
                                else_=None,
                            )
                        ), Numeric),
                        2,
                    ).label("t_mape"),
                    func.round(
                        cast(func.avg(
                            case(
                                (
                                    PredictionModel.period == "day_ahead",
                                    PredictionModel.error_pct,
                                ),
                                else_=None,
                            )
                        ), Numeric),
                        2,
                    ).label("t1_mape"),
                )
                .where(
                    PredictionModel.actual_mwh.is_not(None),
                    PredictionModel.model_source == "ensemble",
                    PredictionModel.forecast_dt >= cutoff,
                    PredictionModel.error_pct.is_not(None),
                )
                .group_by(week_label)
                .order_by(week_label)
            )
            result = await self._session.execute(stmt)
            out: list[dict[str, Any]] = []
            for row in result:
                entry: dict[str, Any] = {
                    "week": row.week,
                    "weekly_mape": float(row.weekly_mape),
                    "n_hours": row.n_hours,
                }
                if row.t_mape is not None:
                    entry["t_mape"] = float(row.t_mape)
                if row.t1_mape is not None:
                    entry["t1_mape"] = float(row.t1_mape)
                out.append(entry)
            return out

        # SQLite fallback: Python-side grouping
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.actual_mwh.is_not(None),
                PredictionModel.model_source == "ensemble",
                PredictionModel.forecast_dt >= cutoff,
            )
            .order_by(PredictionModel.forecast_dt)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        weekly: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"all": [], "intraday": [], "day_ahead": []}
        )
        for r in rows:
            iso = r.forecast_dt.isocalendar()
            week_key = f"{iso[0]}-W{iso[1]:02d}"
            if r.error_pct is not None:
                weekly[week_key]["all"].append(r.error_pct)
                weekly[week_key][r.period].append(r.error_pct)

        out = []
        for week, buckets in sorted(weekly.items()):
            row_dict: dict[str, Any] = {
                "week": week,
                "weekly_mape": _safe_mean(buckets["all"]),
                "n_hours": len(buckets["all"]),
            }
            if buckets["intraday"]:
                row_dict["t_mape"] = _safe_mean(buckets["intraday"])
            if buckets["day_ahead"]:
                row_dict["t1_mape"] = _safe_mean(buckets["day_ahead"])
            out.append(row_dict)
        return out

    async def get_hourly_mape(self) -> list[dict[str, Any]]:
        """Hourly MAPE pattern — which hours have highest error."""
        if self._is_pg:
            hour_col = cast(
                func.extract("hour", PredictionModel.forecast_dt),
                Integer,
            )
            stmt = (
                select(
                    hour_col.label("hour"),
                    func.round(cast(func.avg(PredictionModel.error_pct), Numeric), 2).label(
                        "avg_mape"
                    ),
                    func.count().label("n_samples"),
                )
                .where(
                    PredictionModel.actual_mwh.is_not(None),
                    PredictionModel.model_source == "ensemble",
                    PredictionModel.error_pct.is_not(None),
                )
                .group_by(hour_col)
                .order_by(hour_col)
            )
            result = await self._session.execute(stmt)
            return [
                {
                    "hour": row.hour,
                    "avg_mape": float(row.avg_mape),
                    "n_samples": row.n_samples,
                }
                for row in result
            ]

        # SQLite fallback
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.actual_mwh.is_not(None),
                PredictionModel.model_source == "ensemble",
            )
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        hourly: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            hour = r.forecast_dt.hour
            if r.error_pct is not None:
                hourly[hour].append(r.error_pct)

        return [
            {
                "hour": h,
                "avg_mape": _safe_mean(errs),
                "n_samples": len(errs),
            }
            for h, errs in sorted(hourly.items())
        ]

    # ------------------------------------------------------------------
    # 4.2 — Per-Model Performance
    # ------------------------------------------------------------------

    async def get_per_model_mape(self, days: int = 30) -> list[dict[str, Any]]:
        """Model-level production MAPE comparison."""
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)

        if self._is_pg:
            stmt = (
                select(
                    PredictionModel.model_source,
                    func.round(cast(func.avg(PredictionModel.error_pct), Numeric), 2).label(
                        "mape"
                    ),
                    func.count().label("n_hours"),
                )
                .where(
                    PredictionModel.actual_mwh.is_not(None),
                    PredictionModel.forecast_dt >= cutoff,
                    PredictionModel.model_source.is_not(None),
                    PredictionModel.error_pct.is_not(None),
                )
                .group_by(PredictionModel.model_source)
                .order_by(func.avg(PredictionModel.error_pct))
            )
            result = await self._session.execute(stmt)
            return [
                {
                    "model_source": row.model_source,
                    "mape": float(row.mape),
                    "n_hours": row.n_hours,
                }
                for row in result
            ]

        # SQLite fallback
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.actual_mwh.is_not(None),
                PredictionModel.forecast_dt >= cutoff,
            )
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        by_model: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r.model_source and r.error_pct is not None:
                by_model[r.model_source].append(r.error_pct)

        return sorted(
            [
                {
                    "model_source": model,
                    "mape": _safe_mean(errs),
                    "n_hours": len(errs),
                }
                for model, errs in by_model.items()
            ],
            key=lambda x: x["mape"],
        )

    async def get_hourly_model_performance(self) -> list[dict[str, Any]]:
        """Hourly model MAPE matrix for heatmap visualization."""
        if self._is_pg:
            hour_col = cast(
                func.extract("hour", PredictionModel.forecast_dt),
                Integer,
            )
            stmt = (
                select(
                    hour_col.label("hour"),
                    PredictionModel.model_source,
                    func.round(cast(func.avg(PredictionModel.error_pct), Numeric), 2).label(
                        "mape"
                    ),
                )
                .where(
                    PredictionModel.actual_mwh.is_not(None),
                    PredictionModel.model_source.is_not(None),
                    PredictionModel.error_pct.is_not(None),
                )
                .group_by(hour_col, PredictionModel.model_source)
                .order_by(hour_col, PredictionModel.model_source)
            )
            result = await self._session.execute(stmt)
            return [
                {
                    "hour": row.hour,
                    "model_source": row.model_source,
                    "mape": float(row.mape),
                }
                for row in result
            ]

        # SQLite fallback
        stmt = (
            select(PredictionModel)
            .where(PredictionModel.actual_mwh.is_not(None))
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        matrix: dict[tuple[int, str], list[float]] = defaultdict(list)
        for r in rows:
            if r.model_source and r.error_pct is not None:
                matrix[(r.forecast_dt.hour, r.model_source)].append(
                    r.error_pct
                )

        return [
            {
                "hour": hour,
                "model_source": model,
                "mape": _safe_mean(errs),
            }
            for (hour, model), errs in sorted(matrix.items())
        ]

    async def get_model_comparison_stats(
        self, days: int = 30
    ) -> list[dict[str, Any]]:
        """Ensemble vs individual model stats (avg, median, p95).

        Note: median and p95 require Python-side computation even for PG,
        since percentile_cont is not available in all PG versions.
        Falls back to Python-side for all dialects.
        """
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)
        stmt = (
            select(PredictionModel)
            .where(
                PredictionModel.actual_mwh.is_not(None),
                PredictionModel.forecast_dt >= cutoff,
            )
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        by_model: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r.model_source and r.error_pct is not None:
                by_model[r.model_source].append(r.error_pct)

        out: list[dict[str, Any]] = []
        for model, errs in sorted(by_model.items()):
            s = sorted(errs)
            n = len(s)
            median = s[n // 2] if n else 0.0
            p95_idx = min(int(n * 0.95), n - 1) if n else 0
            out.append(
                {
                    "model_source": model,
                    "avg_mape": _safe_mean(s),
                    "median_mape": round(median, 2),
                    "p95_mape": round(s[p95_idx], 2) if s else 0.0,
                    "n_hours": n,
                }
            )
        return out

    # ------------------------------------------------------------------
    # 4.3 — Weather Forecast Horizon Accuracy
    # ------------------------------------------------------------------

    async def get_weather_horizon_accuracy(
        self, weeks: int = 8
    ) -> list[dict[str, Any]]:
        """T (0-24h) vs T+1 (24-48h) weather forecast accuracy."""
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(weeks=weeks)

        # Load forecasts
        fc_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(False),
                WeatherSnapshotModel.forecast_dt >= cutoff,
            )
        )
        fc_result = await self._session.execute(fc_stmt)
        forecasts = {s.forecast_dt: s for s in fc_result.scalars().all()}

        if not forecasts:
            return []

        # Load actuals for same datetime range
        act_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(True),
                WeatherSnapshotModel.forecast_dt.in_(list(forecasts.keys())),
            )
        )
        act_result = await self._session.execute(act_stmt)
        actuals = {s.forecast_dt: s for s in act_result.scalars().all()}

        # Classify T vs T+1 by comparing forecast_dt date to fetched_at date
        buckets: dict[str, dict[str, list[float]]] = {
            "T": defaultdict(list),
            "T+1": defaultdict(list),
        }
        compare_cols = [
            "temperature_2m", "apparent_temperature",
            "wind_speed_10m", "shortwave_radiation", "precipitation",
        ]
        for dt, fc in forecasts.items():
            act = actuals.get(dt)
            if act is None:
                continue
            # T if forecast_dt is same day as fetched_at, else T+1
            fc_date = fc.forecast_dt.date() if hasattr(fc.forecast_dt, "date") else fc.forecast_dt
            fetch_date = fc.fetched_at.date() if hasattr(fc.fetched_at, "date") else fc.fetched_at
            horizon = "T" if fc_date == fetch_date else "T+1"
            for col in compare_cols:
                fc_val = getattr(fc, col, None)
                act_val = getattr(act, col, None)
                if fc_val is not None and act_val is not None:
                    buckets[horizon][col].append(abs(fc_val - act_val))

        return [
            {
                "horizon": h,
                **{
                    f"{col}_mae": _safe_mean(vals)
                    for col, vals in cols.items()
                },
                "n_hours": max(len(v) for v in cols.values()) if cols else 0,
            }
            for h, cols in buckets.items()
            if cols
        ]

    async def get_weather_variable_accuracy(self) -> list[dict[str, Any]]:
        """Per-variable weather forecast MAE (all time)."""
        fc_stmt = (
            select(WeatherSnapshotModel)
            .where(WeatherSnapshotModel.is_actual.is_(False))
        )
        fc_result = await self._session.execute(fc_stmt)
        forecasts = {s.forecast_dt: s for s in fc_result.scalars().all()}

        if not forecasts:
            return []

        act_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(True),
                WeatherSnapshotModel.forecast_dt.in_(list(forecasts.keys())),
            )
        )
        act_result = await self._session.execute(act_stmt)
        actuals = {s.forecast_dt: s for s in act_result.scalars().all()}

        compare_cols = [
            "temperature_2m", "apparent_temperature",
            "wind_speed_10m", "shortwave_radiation", "precipitation",
        ]
        errors: dict[str, list[float]] = defaultdict(list)
        for dt, fc in forecasts.items():
            act = actuals.get(dt)
            if act is None:
                continue
            for col in compare_cols:
                fc_val = getattr(fc, col, None)
                act_val = getattr(act, col, None)
                if fc_val is not None and act_val is not None:
                    errors[col].append(abs(fc_val - act_val))

        return [
            {
                "variable": col,
                "mae": _safe_mean(errs),
                "n_samples": len(errs),
            }
            for col, errs in errors.items()
        ]

    # ------------------------------------------------------------------
    # 4.4 — Feature Importance Trend
    # ------------------------------------------------------------------

    async def get_feature_importance_trend(
        self, days: int = 30, top_n: int = 10
    ) -> list[dict[str, Any]]:
        """Feature importance change over time from jobs.metadata_ JSONB."""
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)
        stmt = (
            select(JobModel)
            .where(
                JobModel.status == "completed",
                JobModel.created_at >= cutoff,
            )
            .order_by(JobModel.created_at)
        )
        result = await self._session.execute(stmt)
        jobs = result.scalars().all()

        # Extract feature importance from JSONB metadata
        out: list[dict[str, Any]] = []
        all_features: dict[str, float] = defaultdict(float)
        for job in jobs:
            meta = job.metadata_ or {}
            fi_list = meta.get("feature_importance_top15")
            if not fi_list:
                continue
            day = job.created_at.strftime("%Y-%m-%d")
            for entry in fi_list:
                feat = entry.get("feature", "")
                imp = entry.get("importance", 0.0)
                out.append({"day": day, "feature": feat, "importance": imp})
                all_features[feat] += imp

        # If caller wants only top-N features, filter
        top_features = sorted(
            all_features, key=all_features.get, reverse=True  # type: ignore[arg-type]
        )[:top_n]
        return [r for r in out if r["feature"] in top_features]

    # ------------------------------------------------------------------
    # 4.7 — EPIAS Forecast vs Actual
    # ------------------------------------------------------------------

    async def get_epias_forecast_accuracy(
        self, days: int = 30
    ) -> list[dict[str, Any]]:
        """EPIAS Load_Forecast vs Real_Time_Consumption from job snapshots."""
        cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)
        stmt = (
            select(JobModel)
            .where(
                JobModel.status == "completed",
                JobModel.created_at >= cutoff,
            )
            .order_by(JobModel.created_at)
        )
        result = await self._session.execute(stmt)
        jobs = result.scalars().all()

        out: list[dict[str, Any]] = []
        for job in jobs:
            snap = job.epias_snapshot
            if not snap:
                continue
            last_vals = snap.get("last_values", {})
            lf = last_vals.get("Load_Forecast")
            rtc = last_vals.get("Real_Time_Consumption")
            if lf is None or rtc is None or rtc == 0:
                continue
            out.append(
                {
                    "day": job.created_at.strftime("%Y-%m-%d"),
                    "load_forecast": float(lf),
                    "rtc": float(rtc),
                    "epias_error_pct": round(
                        abs(float(lf) - float(rtc)) / float(rtc) * 100, 2
                    ),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Job History (paginated)
    # ------------------------------------------------------------------

    async def get_job_history(
        self, page: int = 1, size: int = 20
    ) -> dict[str, Any]:
        """Paginated job history for admin dashboard."""
        count_stmt = (
            select(func.count()).select_from(JobModel)
        )
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar() or 0

        offset = (page - 1) * size
        stmt = (
            select(JobModel)
            .order_by(JobModel.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        result = await self._session.execute(stmt)
        jobs = result.scalars().all()

        return {
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size if size > 0 else 0,
            "jobs": [
                {
                    "id": j.id,
                    "email": j.email[:3] + "***" if j.email else "",
                    "status": j.status,
                    "progress": j.progress,
                    "created_at": j.created_at.isoformat(),
                    "completed_at": (
                        j.completed_at.isoformat() if j.completed_at else None
                    ),
                }
                for j in jobs
            ],
        }

    # ------------------------------------------------------------------
    # Model Runs (from Faz 3)
    # ------------------------------------------------------------------

    async def get_model_runs(
        self,
        model_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Training history from model_runs table."""
        stmt = select(ModelRunModel).order_by(ModelRunModel.id.desc())
        if model_type:
            stmt = stmt.where(ModelRunModel.model_type == model_type)
        stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        runs = result.scalars().all()
        return [
            {
                "id": r.id,
                "model_type": r.model_type,
                "status": r.status,
                "val_mape": r.val_mape,
                "test_mape": r.test_mape,
                "n_trials": r.n_trials,
                "n_splits": r.n_splits,
                "feature_count": r.feature_count,
                "is_promoted": r.is_promoted,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": (
                    r.completed_at.isoformat() if r.completed_at else None
                ),
                "duration_seconds": r.duration_seconds,
            }
            for r in runs
        ]

    async def get_promoted_models(self) -> list[dict[str, Any]]:
        """Currently promoted models by type."""
        stmt = (
            select(ModelRunModel)
            .where(ModelRunModel.is_promoted.is_(True))
            .order_by(ModelRunModel.model_type)
        )
        result = await self._session.execute(stmt)
        runs = result.scalars().all()
        return [
            {
                "model_type": r.model_type,
                "val_mape": r.val_mape,
                "test_mape": r.test_mape,
                "model_path": r.model_path,
                "promoted_at": (
                    r.promoted_at.isoformat() if r.promoted_at else None
                ),
            }
            for r in runs
        ]

    async def get_drift_status(self) -> list[dict[str, Any]]:
        """Recent drift alerts from audit_logs."""
        stmt = (
            select(AuditLogModel)
            .where(AuditLogModel.action.like("drift_%"))
            .order_by(AuditLogModel.created_at.desc())
            .limit(20)
        )
        result = await self._session.execute(stmt)
        logs = result.scalars().all()
        return [
            {
                "action": log.action,
                "details": log.details,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    """Compute mean with empty-list safety."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)
