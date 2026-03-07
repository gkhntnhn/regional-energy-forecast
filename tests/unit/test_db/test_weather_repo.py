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


# ------------------------------------------------------------------
# _build_snapshot_rows — naive timestamp branch (line 104)
# ------------------------------------------------------------------


def test_build_snapshot_rows_naive_index() -> None:
    """Naive DatetimeIndex is tz_localized to Europe/Istanbul in _build_snapshot_rows."""
    from zoneinfo import ZoneInfo

    tz_istanbul = ZoneInfo("Europe/Istanbul")

    # Create DataFrame with naive (no tz) DatetimeIndex
    naive_idx = pd.date_range("2026-03-05", periods=3, freq="h")
    assert naive_idx.tz is None  # confirm naive

    weather_df = pd.DataFrame(
        {"temperature_2m": [15.0, 20.0, 25.0], "wind_speed_10m": [3.0, 5.0, 7.0]},
        index=naive_idx,
    )

    # Call the static method directly (no DB round-trip needed)
    rows = WeatherSnapshotRepository._build_snapshot_rows(
        weather_df,
        fetched_at=datetime.now(tz=TZ),
        is_actual=False,
        job_id="test_job",
    )
    assert len(rows) == 3

    # Verify each forecast_dt has timezone info (was localized, not left naive)
    for row in rows:
        dt = row.forecast_dt
        assert dt.tzinfo is not None, "Expected tz-aware datetime after tz_localize"
        assert dt.tzinfo == tz_istanbul

    # Verify the hours are correct (localized, not converted)
    assert rows[0].forecast_dt.hour == 0
    assert rows[1].forecast_dt.hour == 1
    assert rows[2].forecast_dt.hour == 2

    # Verify temperature values are set correctly
    assert rows[0].temperature_2m == pytest.approx(15.0)
    assert rows[2].temperature_2m == pytest.approx(25.0)

    # Verify HDD/CDD are computed
    # 15C: HDD = 18-15 = 3, CDD = 0
    assert rows[0].wth_hdd == pytest.approx(3.0)
    assert rows[0].wth_cdd == pytest.approx(0.0)
    # 25C: HDD = 0, CDD = 25-22 = 3
    assert rows[2].wth_hdd == pytest.approx(0.0)
    assert rows[2].wth_cdd == pytest.approx(3.0)


# ------------------------------------------------------------------
# get_forecast_vs_actual — empty forecasts case (line 175)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_forecast_vs_actual_empty(
    db_session: AsyncSession, _seed_job: str
) -> None:
    """get_forecast_vs_actual returns empty list when no forecasts exist for a job."""
    repo = WeatherSnapshotRepository(db_session)

    # No forecast snapshots created for this job — should return []
    result = await repo.get_forecast_vs_actual(_seed_job)
    assert result == []


# ------------------------------------------------------------------
# get_weekly_accuracy (lines 214-276)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_weekly_accuracy_basic(
    db_session: AsyncSession, _seed_job: str
) -> None:
    """get_weekly_accuracy returns weekly MAE when forecast+actual snapshots exist."""
    repo = WeatherSnapshotRepository(db_session)

    # Use recent dates (relative to now) so they fall within the weeks=52 window.
    # We create data for 2 separate ISO weeks.
    now = datetime.now(tz=TZ)

    # Week A: 7 days ago — 3 hours of data
    week_a_start = now - timedelta(days=7)
    week_a_idx = pd.date_range(
        week_a_start.strftime("%Y-%m-%dT00:00"), periods=3, freq="h", tz=TZ,
    )
    fc_df_a = pd.DataFrame(
        {"temperature_2m": [10.0, 12.0, 14.0], "wind_speed_10m": [5.0, 6.0, 7.0]},
        index=week_a_idx,
    )
    act_df_a = pd.DataFrame(
        {"temperature_2m": [11.0, 14.0, 16.0], "wind_speed_10m": [4.0, 8.0, 10.0]},
        index=week_a_idx,
    )

    # Week B: 14 days ago — 2 hours of data
    week_b_start = now - timedelta(days=14)
    week_b_idx = pd.date_range(
        week_b_start.strftime("%Y-%m-%dT00:00"), periods=2, freq="h", tz=TZ,
    )
    fc_df_b = pd.DataFrame(
        {"temperature_2m": [20.0, 22.0], "wind_speed_10m": [10.0, 12.0]},
        index=week_b_idx,
    )
    act_df_b = pd.DataFrame(
        {"temperature_2m": [18.0, 20.0], "wind_speed_10m": [11.0, 14.0]},
        index=week_b_idx,
    )

    # Insert forecasts (linked to job)
    await repo.bulk_create_forecast(
        job_id=_seed_job, weather_df=fc_df_a, fetched_at=datetime.now(tz=TZ),
    )
    # Need a second job for week B forecasts
    job2 = JobModel(
        id="weather_test2", email="t@t.com",
        excel_path="/tmp/t2.xlsx", file_stem="t2", status="completed",
    )
    db_session.add(job2)
    await db_session.flush()
    await repo.bulk_create_forecast(
        job_id="weather_test2", weather_df=fc_df_b, fetched_at=datetime.now(tz=TZ),
    )

    # Insert actuals (no job_id)
    await repo.bulk_create_actuals(weather_df=act_df_a, fetched_at=datetime.now(tz=TZ))
    await repo.bulk_create_actuals(weather_df=act_df_b, fetched_at=datetime.now(tz=TZ))

    # Use weeks=52 to ensure all test data is included
    result = await repo.get_weekly_accuracy(weeks=52)

    # We should have at least 1 week (possibly 2 if the dates span different ISO weeks,
    # but they could land in the same ISO week depending on when the test runs).
    assert len(result) >= 1

    # Verify structure: each row has week, sample_count, mae_* keys
    for row in result:
        assert "week" in row
        assert "sample_count" in row
        assert row["sample_count"] > 0
        assert "mae_temperature_2m" in row
        assert "mae_wind_speed_10m" in row
        assert row["mae_temperature_2m"] > 0  # errors are non-zero
        assert row["mae_wind_speed_10m"] > 0

    # Verify total sample count across all weeks equals 5 (3 + 2)
    total_samples = sum(r["sample_count"] for r in result)
    assert total_samples == 5


@pytest.mark.asyncio
async def test_get_weekly_accuracy_no_actuals(
    db_session: AsyncSession,
) -> None:
    """get_weekly_accuracy returns empty list when no actuals exist in DB."""
    repo = WeatherSnapshotRepository(db_session)

    # No data at all — should return []
    result = await repo.get_weekly_accuracy(weeks=4)

    assert result == []
