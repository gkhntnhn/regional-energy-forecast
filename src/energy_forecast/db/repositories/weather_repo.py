"""Weather snapshot repository — CRUD + accuracy queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import WeatherSnapshotModel

# Weather columns in the OpenMeteo DataFrame
_WEATHER_COLS: list[str] = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "weather_code",
]

# HDD/CDD base temperature (same as weather feature engineer)
_HDD_BASE = 18.0
_CDD_BASE = 22.0


class WeatherSnapshotRepository:
    """Data access layer for weather_snapshots table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def bulk_create_forecast(
        self,
        job_id: str,
        weather_df: pd.DataFrame,
        fetched_at: datetime,
    ) -> int:
        """Store forecast weather snapshot from prediction pipeline.

        Args:
            job_id: Associated prediction job ID.
            weather_df: Weather DataFrame with DatetimeIndex (48 rows).
            fetched_at: Timestamp when data was fetched from OpenMeteo.

        Returns:
            Number of rows inserted.
        """
        rows = self._build_snapshot_rows(
            weather_df, fetched_at, is_actual=False, job_id=job_id,
        )
        self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    async def bulk_create_actuals(
        self,
        weather_df: pd.DataFrame,
        fetched_at: datetime,
    ) -> int:
        """Store actual weather data from OpenMeteo Archive API.

        Args:
            weather_df: Weather DataFrame with DatetimeIndex (24 rows/day).
            fetched_at: Timestamp when data was fetched.

        Returns:
            Number of rows inserted.
        """
        rows = self._build_snapshot_rows(
            weather_df, fetched_at, is_actual=True, job_id=None,
        )

        self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    @staticmethod
    def _build_snapshot_rows(
        weather_df: pd.DataFrame,
        fetched_at: datetime,
        *,
        is_actual: bool,
        job_id: str | None,
    ) -> list[WeatherSnapshotModel]:
        """Build WeatherSnapshotModel rows from a weather DataFrame."""
        from energy_forecast.utils import TZ_ISTANBUL

        rows: list[WeatherSnapshotModel] = []
        for idx, row in weather_df.iterrows():
            dt = pd.Timestamp(idx)  # type: ignore[arg-type]
            if dt.tzinfo is None:
                dt = dt.tz_localize(TZ_ISTANBUL)

            snapshot = WeatherSnapshotModel(
                job_id=job_id,
                forecast_dt=dt.to_pydatetime(),
                fetched_at=fetched_at,
                is_actual=is_actual,
            )
            for col in _WEATHER_COLS:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        setattr(
                            snapshot, col,
                            int(val) if col == "weather_code" else float(val),
                        )

            if "temperature_2m" in row.index and pd.notna(row["temperature_2m"]):
                temp = float(row["temperature_2m"])
                snapshot.wth_hdd = max(0.0, _HDD_BASE - temp)
                snapshot.wth_cdd = max(0.0, temp - _CDD_BASE)

            rows.append(snapshot)
        return rows

    async def has_actuals_for_date(self, target_date: datetime) -> bool:
        """Check if actual weather data exists for a given date."""
        from datetime import timedelta

        start = datetime(
            target_date.year, target_date.month, target_date.day,
            tzinfo=target_date.tzinfo,
        )
        end = start + timedelta(days=1)
        stmt = (
            select(func.count())
            .select_from(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(True),
                WeatherSnapshotModel.forecast_dt >= start,
                WeatherSnapshotModel.forecast_dt < end,
            )
        )
        result = await self._session.execute(stmt)
        count = result.scalar() or 0
        return count > 0

    # ------------------------------------------------------------------
    # Read / query operations
    # ------------------------------------------------------------------

    async def get_forecast_vs_actual(
        self, job_id: str
    ) -> list[dict[str, Any]]:
        """Get forecast and actual weather side by side for a job.

        Returns list of dicts with forecast_dt, forecast_*, actual_* columns.
        """
        # Get forecasts for this job
        forecast_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.job_id == job_id,
                WeatherSnapshotModel.is_actual.is_(False),
            )
            .order_by(WeatherSnapshotModel.forecast_dt)
        )
        forecast_result = await self._session.execute(forecast_stmt)
        forecasts = {s.forecast_dt: s for s in forecast_result.scalars().all()}

        if not forecasts:
            return []

        # Get actuals for the same datetime range
        dt_min = min(forecasts.keys())
        dt_max = max(forecasts.keys())
        actual_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(True),
                WeatherSnapshotModel.forecast_dt >= dt_min,
                WeatherSnapshotModel.forecast_dt <= dt_max,
            )
        )
        actual_result = await self._session.execute(actual_stmt)
        actuals = {s.forecast_dt: s for s in actual_result.scalars().all()}

        comparison: list[dict[str, Any]] = []
        compare_cols = ["temperature_2m", "apparent_temperature", "wind_speed_10m"]
        for dt, fc in sorted(forecasts.items()):
            row: dict[str, Any] = {"forecast_dt": dt}
            act = actuals.get(dt)
            for col in compare_cols:
                fc_val = getattr(fc, col, None)
                act_val = getattr(act, col, None) if act else None
                row[f"forecast_{col}"] = fc_val
                row[f"actual_{col}"] = act_val
                if fc_val is not None and act_val is not None:
                    row[f"error_{col}"] = abs(fc_val - act_val)
            comparison.append(row)

        return comparison

    async def get_weekly_accuracy(
        self, weeks: int = 4
    ) -> list[dict[str, Any]]:
        """Get weekly aggregate MAE for key weather variables.

        Returns list of dicts with week_start, mae_temperature, mae_wind, etc.
        """
        from datetime import timedelta

        from energy_forecast.utils import TZ_ISTANBUL

        now = datetime.now(tz=TZ_ISTANBUL)
        cutoff = now - timedelta(weeks=weeks)

        # Get all actuals in range
        actual_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(True),
                WeatherSnapshotModel.forecast_dt >= cutoff,
            )
            .order_by(WeatherSnapshotModel.forecast_dt)
        )
        actual_result = await self._session.execute(actual_stmt)
        actuals = {s.forecast_dt: s for s in actual_result.scalars().all()}

        if not actuals:
            return []

        # Get all forecasts that have matching actuals
        forecast_stmt = (
            select(WeatherSnapshotModel)
            .where(
                WeatherSnapshotModel.is_actual.is_(False),
                WeatherSnapshotModel.forecast_dt.in_(list(actuals.keys())),
            )
        )
        forecast_result = await self._session.execute(forecast_stmt)

        # Group errors by week
        weekly_errors: dict[str, list[dict[str, float]]] = {}
        compare_cols = ["temperature_2m", "wind_speed_10m"]
        for fc in forecast_result.scalars().all():
            act = actuals.get(fc.forecast_dt)
            if act is None:
                continue
            # ISO week key
            week_key = fc.forecast_dt.strftime("%Y-W%W")
            if week_key not in weekly_errors:
                weekly_errors[week_key] = []
            errors: dict[str, float] = {}
            for col in compare_cols:
                fc_val = getattr(fc, col, None)
                act_val = getattr(act, col, None)
                if fc_val is not None and act_val is not None:
                    errors[col] = abs(fc_val - act_val)
            if errors:
                weekly_errors[week_key].append(errors)

        # Aggregate
        result: list[dict[str, Any]] = []
        for week, errors_list in sorted(weekly_errors.items()):
            row: dict[str, Any] = {"week": week, "sample_count": len(errors_list)}
            for col in compare_cols:
                col_errors = [e[col] for e in errors_list if col in e]
                if col_errors:
                    row[f"mae_{col}"] = sum(col_errors) / len(col_errors)
            result.append(row)

        return result

    async def get_by_job_id(
        self, job_id: str, is_actual: bool | None = None
    ) -> list[WeatherSnapshotModel]:
        """Get all weather snapshots for a job."""
        stmt = (
            select(WeatherSnapshotModel)
            .where(WeatherSnapshotModel.job_id == job_id)
            .order_by(WeatherSnapshotModel.forecast_dt)
        )
        if is_actual is not None:
            stmt = stmt.where(
                WeatherSnapshotModel.is_actual == is_actual
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
