"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from energy_forecast.config import Settings, get_default_config, load_config
from energy_forecast.config.settings import (
    DataLoaderConfig,
    OpenMeteoConfig,
    RegionConfig,
)
from energy_forecast.db.base import Base

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture()
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory path."""
    return project_root / "configs"


@pytest.fixture()
def data_dir(project_root: Path) -> Path:
    """Return the data directory path."""
    return project_root / "data"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(configs_dir: Path) -> Settings:
    """Load Settings from project YAML configs."""
    return load_config(configs_dir)


@pytest.fixture()
def default_settings() -> Settings:
    """Get default Settings (no YAML files needed)."""
    return get_default_config()


@pytest.fixture()
def data_loader_config(settings: Settings) -> DataLoaderConfig:
    """DataLoaderConfig from project settings."""
    return settings.data_loader


@pytest.fixture()
def openmeteo_config(settings: Settings) -> OpenMeteoConfig:
    """OpenMeteoConfig from project settings."""
    return settings.openmeteo


@pytest.fixture()
def region_config(settings: Settings) -> RegionConfig:
    """RegionConfig from project settings."""
    return settings.region


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_excel_df() -> pd.DataFrame:
    """Minimal valid consumption DataFrame (3 days = 72 rows)."""
    dates: list[str] = []
    times: list[int] = []
    consumptions: list[float] = []

    rng = np.random.default_rng(42)
    for day in range(1, 4):
        for hour in range(24):
            dates.append(f"2024-01-{day:02d}")
            times.append(hour)
            consumptions.append(round(800.0 + rng.random() * 400, 1))

    return pd.DataFrame(
        {
            "date": dates,
            "time": times,
            "consumption": consumptions,
        }
    )


@pytest.fixture()
def sample_excel_path(tmp_path: Path, sample_excel_df: pd.DataFrame) -> Path:
    """Write sample consumption to an Excel file and return the path."""
    path = tmp_path / "consumption.xlsx"
    sample_excel_df.to_excel(path, index=False, engine="openpyxl")
    return path


@pytest.fixture()
def sample_epias_response() -> dict[str, Any]:
    """Mock EPIAS API response for a single variable (24 hours)."""
    items = []
    for hour in range(24):
        items.append(
            {
                "date": f"2024-01-01T{hour:02d}:00:00+03:00",
                "toplam": 1000.0 + hour * 10,
            }
        )
    return {"body": {"content": items}}


@pytest.fixture()
def sample_openmeteo_response() -> dict[str, Any]:
    """Mock OpenMeteo API response with 11 variables (24 hours)."""
    times = [f"2024-01-01T{h:02d}:00" for h in range(24)]
    rng = np.random.default_rng(42)

    hourly: dict[str, Any] = {"time": times}
    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "snow_depth",
        "weather_code",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "shortwave_radiation",
    ]
    for var in variables:
        hourly[var] = (rng.random(24) * 30).tolist()

    return {"hourly": hourly}


# ---------------------------------------------------------------------------
# Feature engineering fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_consumption_df() -> pd.DataFrame:
    """30 days × 24 hours = 720 rows with consumption + DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=720, freq="h")
    return pd.DataFrame(
        {"consumption": 800.0 + rng.random(720) * 400},
        index=idx,
    ).rename_axis("datetime")


@pytest.fixture()
def sample_weather_df() -> pd.DataFrame:
    """7 days × 24 hours = 168 rows with 11 weather columns + DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=168, freq="h")
    data: dict[str, Any] = {}
    data["temperature_2m"] = rng.uniform(-5, 35, 168)
    data["relative_humidity_2m"] = rng.uniform(20, 95, 168)
    data["dew_point_2m"] = rng.uniform(-10, 20, 168)
    data["apparent_temperature"] = rng.uniform(-10, 40, 168)
    data["precipitation"] = rng.exponential(1.0, 168)
    data["snow_depth"] = rng.uniform(0, 5, 168)
    data["weather_code"] = rng.choice([0, 1, 2, 3, 45, 61, 71, 95], 168)
    data["surface_pressure"] = rng.uniform(990, 1030, 168)
    data["wind_speed_10m"] = rng.uniform(0, 60, 168)
    data["wind_direction_10m"] = rng.uniform(0, 360, 168)
    data["shortwave_radiation"] = rng.uniform(0, 800, 168)
    return pd.DataFrame(data, index=idx).rename_axis("datetime")


@pytest.fixture()
def sample_epias_df() -> pd.DataFrame:
    """30 days × 24 hours = 720 rows with 5 EPIAS columns + DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=720, freq="h")
    data: dict[str, Any] = {}
    for var in [
        "FDPP",
        "Real_Time_Consumption",
        "DAM_Purchase",
        "Bilateral_Agreement_Purchase",
        "Load_Forecast",
    ]:
        data[var] = 500.0 + rng.random(720) * 1000
    return pd.DataFrame(data, index=idx).rename_axis("datetime")


@pytest.fixture()
def sample_full_df(
    sample_consumption_df: pd.DataFrame,
    sample_weather_df: pd.DataFrame,
    sample_epias_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combined DataFrame: consumption + weather + EPIAS (720 rows)."""
    weather_extended = sample_weather_df.reindex(sample_consumption_df.index, method="ffill")
    return pd.concat(
        [sample_consumption_df, weather_extended, sample_epias_df],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Database fixtures (SQLite in-memory via aiosqlite)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_engine():  # type: ignore[no-untyped-def]
    """Create an async SQLite in-memory engine with schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: Any) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session for testing."""
    factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def db_session_factory(db_engine: Any) -> async_sessionmaker[AsyncSession]:
    """Return a session factory for testing process_job_db."""
    return async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
