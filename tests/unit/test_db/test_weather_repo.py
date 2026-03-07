"""Tests for WeatherSnapshotRepository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import JobModel
from energy_forecast.db.repositories.weather_repo import WeatherSnapshotRepository

TZ = timezone(timedelta(hours=3))


def _make_weather_df(start: str, hours: int = 24) -> pd.DataFrame:
    """Create a sample weather DataFrame."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=hours, freq="h", tz=TZ)
    return pd.DataFrame(
        {
            "temperature_2m": rng.uniform(5, 30, hours),
            "apparent_temperature": rng.uniform(3, 28, hours),
            "relative_humidity_2m": rng.uniform(30, 90, hours),
            "wind_speed_10m": rng.uniform(0, 20, hours),
            "weather_code": rng.choice([0, 1, 2, 3, 61], hours),
        },
        index=idx,
    )


@pytest_asyncio.fixture
async def _seed_job(db_session: AsyncSession) -> str:
    """Seed a job and return its ID."""
    job = JobModel(
        id="weather_test1",
        email="test@example.com",
        excel_path="/tmp/test.xlsx",
        file_stem="test",
        status="completed",
    )
    db_session.add(job)
    await db_session.flush()
    return job.id


# ------------------------------------------------------------------
# bulk_create_forecast
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_create_forecast(
    db_session: AsyncSession, _seed_job: str
) -> None:
    """Forecast snapshots are created and linked to job."""
    repo = WeatherSnapshotRepository(db_session)
    weather_df = _make_weather_df("2026-03-05")
    fetched = datetime.now(tz=TZ)

    count = await repo.bulk_create_forecast(
        job_id=_seed_job, weather_df=weather_df, fetched_at=fetched,
    )
    assert count == 24

    rows = await repo.get_by_job_id(_seed_job, is_actual=False)
    assert len(rows) == 24
    assert all(not r.is_actual for r in rows)
    assert all(r.job_id == _seed_job for r in rows)


@pytest.mark.asyncio
async def test_forecast_computes_hdd_cdd(
    db_session: AsyncSession, _seed_job: str
) -> None:
    """HDD and CDD are computed from temperature."""
    idx = pd.date_range("2026-03-05", periods=2, freq="h", tz=TZ)
    weather_df = pd.DataFrame(
        {"temperature_2m": [10.0, 30.0]},  # cold and hot
        index=idx,
    )
    repo = WeatherSnapshotRepository(db_session)
    await repo.bulk_create_forecast(
        job_id=_seed_job,
        weather_df=weather_df,
        fetched_at=datetime.now(tz=TZ),
    )

    rows = await repo.get_by_job_id(_seed_job)
    # 10C: HDD = 18-10 = 8, CDD = 0
    assert rows[0].wth_hdd == pytest.approx(8.0)
    assert rows[0].wth_cdd == pytest.approx(0.0)
    # 30C: HDD = 0, CDD = 30-22 = 8
    assert rows[1].wth_hdd == pytest.approx(0.0)
    assert rows[1].wth_cdd == pytest.approx(8.0)


# ------------------------------------------------------------------
# bulk_create_actuals + idempotent check
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_create_actuals(db_session: AsyncSession) -> None:
    """Actual snapshots are created with job_id=None."""
    repo = WeatherSnapshotRepository(db_session)
    weather_df = _make_weather_df("2026-03-03")
    fetched = datetime.now(tz=TZ)

    count = await repo.bulk_create_actuals(
        weather_df=weather_df, fetched_at=fetched,
    )
    assert count == 24


@pytest.mark.asyncio
async def test_has_actuals_for_date(db_session: AsyncSession) -> None:
    """has_actuals_for_date returns True after actuals are stored."""
    repo = WeatherSnapshotRepository(db_session)
    target = datetime(2026, 3, 3, tzinfo=TZ)

    assert await repo.has_actuals_for_date(target) is False

    weather_df = _make_weather_df("2026-03-03")
    await repo.bulk_create_actuals(
        weather_df=weather_df, fetched_at=datetime.now(tz=TZ),
    )

    assert await repo.has_actuals_for_date(target) is True


# ------------------------------------------------------------------
# forecast_vs_actual comparison
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forecast_vs_actual(
    db_session: AsyncSession, _seed_job: str
) -> None:
    """Forecast vs actual comparison returns error values."""
    repo = WeatherSnapshotRepository(db_session)

    # Create forecast snapshot
    fc_df = pd.DataFrame(
        {"temperature_2m": [20.0], "wind_speed_10m": [5.0]},
        index=pd.date_range("2026-03-05T00:00", periods=1, freq="h", tz=TZ),
    )
    await repo.bulk_create_forecast(
        job_id=_seed_job, weather_df=fc_df,
        fetched_at=datetime.now(tz=TZ),
    )

    # Create actual snapshot
    act_df = pd.DataFrame(
        {"temperature_2m": [22.0], "wind_speed_10m": [3.0]},
        index=pd.date_range("2026-03-05T00:00", periods=1, freq="h", tz=TZ),
    )
    await repo.bulk_create_actuals(
        weather_df=act_df, fetched_at=datetime.now(tz=TZ),
    )

    comparison = await repo.get_forecast_vs_actual(_seed_job)
    assert len(comparison) == 1
    row = comparison[0]
    assert row["forecast_temperature_2m"] == pytest.approx(20.0)
    assert row["actual_temperature_2m"] == pytest.approx(22.0)
    assert row["error_temperature_2m"] == pytest.approx(2.0)


# ------------------------------------------------------------------
# weather_code stored as integer
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weather_code_stored_as_int(
    db_session: AsyncSession, _seed_job: str
) -> None:
    """Weather code is stored as integer, not float."""
    idx = pd.date_range("2026-03-05", periods=1, freq="h", tz=TZ)
    weather_df = pd.DataFrame(
        {"weather_code": [61.0]},  # float from OpenMeteo
        index=idx,
    )
    repo = WeatherSnapshotRepository(db_session)
    await repo.bulk_create_forecast(
        job_id=_seed_job,
        weather_df=weather_df,
        fetched_at=datetime.now(tz=TZ),
    )
    rows = await repo.get_by_job_id(_seed_job)
    assert rows[0].weather_code == 61
    assert isinstance(rows[0].weather_code, int)
