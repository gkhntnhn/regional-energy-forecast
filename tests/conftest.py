"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import Settings, get_default_config, load_config
from energy_forecast.config.settings import (
    DataLoaderConfig,
    OpenMeteoConfig,
    RegionConfig,
)

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
