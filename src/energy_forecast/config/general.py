"""General, pipeline, data-loader and OpenMeteo configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CityConfig",
    "DataLoaderConfig",
    "DataValidationConfig",
    "EpiasApiConfig",
    "ExcelColumnsConfig",
    "ForecastConfig",
    "GeocodingConfig",
    "LoggingConfig",
    "OpenMeteoApiConfig",
    "OpenMeteoConfig",
    "PathsConfig",
    "PipelineConfig",
    "ProjectConfig",
    "RegionConfig",
    "TrainingPathsConfig",
    "WeatherCacheConfig",
    "_load_yaml",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file contains invalid syntax.
        TypeError: If the YAML content is not a dictionary.
    """
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML syntax in {path}: {e}"
        raise yaml.YAMLError(msg) from e

    if not isinstance(data, dict):
        msg = f"Expected dict from {path}, got {type(data).__name__}"
        raise TypeError(msg)
    return data


# ---------------------------------------------------------------------------
# General configs
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel, frozen=True):
    """Project metadata."""

    name: str = "energy-forecast"
    version: str = "0.1.0"
    timezone: str = "Europe/Istanbul"


class LoggingConfig(BaseModel, frozen=True):
    """Loguru logging parameters."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level>"
        " | <cyan>{name}</cyan> - <level>{message}</level>"
    )
    rotation: str = "10 MB"
    retention: str = "30 days"


class CityConfig(BaseModel, frozen=True):
    """Single city location and weight."""

    name: str
    weight: float = Field(ge=0.0, le=1.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class RegionConfig(BaseModel, frozen=True):
    """Regional grouping with weighted cities."""

    name: str = "Uludag"
    cities: list[CityConfig]

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Self:
        total = sum(c.weight for c in self.cities)
        if abs(total - 1.0) > 1e-6:
            msg = f"City weights must sum to 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self


class ForecastConfig(BaseModel, frozen=True):
    """Forecast horizon and frequency settings."""

    horizon_hours: int = Field(default=48, ge=1)
    frequency: str = "1h"
    min_lag: int = Field(default=48, ge=48)


class TrainingPathsConfig(BaseModel, frozen=True):
    """Default file paths for training and models."""

    raw_excel: str = "data/raw/Consumption_Input_Format.xlsx"
    features_historical: str = "data/processed/features_historical.parquet"
    features_forecast: str = "data/processed/features_forecast.parquet"
    features_data: str = "data/processed/features_historical.parquet"  # Alias for training
    models_dir: str = "models"
    ensemble_weights: str = "models/ensemble_weights.json"


class EpiasApiConfig(BaseModel, frozen=True):
    """EPIAS Transparency Platform API client configuration."""

    auth_url: str = "https://giris.epias.com.tr/cas/v1/tickets"
    base_url: str = "https://seffaflik.epias.com.tr/electricity-service/v1"
    cache_dir: str = "data/external/epias"
    file_pattern: str = "epias_market_{year}.parquet"
    generation_file_pattern: str = "epias_generation_{year}.parquet"
    rate_limit_seconds: float = Field(default=10.0, ge=0.0)
    token_ttl_seconds: float = Field(default=3600.0, ge=0.0)
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    retry_attempts: int = Field(default=3, ge=1)
    retry_min_wait: int = Field(default=4, ge=1)
    retry_max_wait: int = Field(default=60, ge=1)


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel, frozen=True):
    """Feature pipeline orchestration."""

    modules: list[str] = Field(
        default_factory=lambda: ["calendar", "consumption", "weather", "solar", "epias"],
    )
    merge_strategy: Literal["left", "inner", "outer"] = "left"
    drop_raw_epias: bool = True
    validate_output: bool = True
    check_duplicate_columns: bool = True


# ---------------------------------------------------------------------------
# Data loader config
# ---------------------------------------------------------------------------


class ExcelColumnsConfig(BaseModel, frozen=True):
    """Excel column name mapping."""

    date: str = "date"
    time: str = "time"
    consumption: str = "consumption"


class DataValidationConfig(BaseModel, frozen=True):
    """Input data validation rules."""

    min_consumption: float = Field(default=0.0, ge=0.0)
    max_consumption: float = Field(default=10000.0, gt=0.0)
    max_missing_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    expected_frequency: str = "1h"


class PathsConfig(BaseModel, frozen=True):
    """Data directory paths."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    static: Path = Path("data/static")
    holidays: Path = Path("data/static/turkish_holidays.parquet")


class DataLoaderConfig(BaseModel, frozen=True):
    """Data loading and validation settings."""

    excel: ExcelColumnsConfig = Field(default_factory=ExcelColumnsConfig)
    validation: DataValidationConfig = Field(default_factory=DataValidationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    date_format: str = "%Y-%m-%d"
    time_range: list[int] = Field(default_factory=lambda: [0, 23])
    unit: str = "MWh"


# ---------------------------------------------------------------------------
# OpenMeteo config
# ---------------------------------------------------------------------------


class OpenMeteoApiConfig(BaseModel, frozen=True):
    """OpenMeteo API connection settings."""

    base_url_historical: str = "https://archive-api.open-meteo.com/v1/archive"
    base_url_historical_forecast: str = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )
    base_url_forecast: str = "https://api.open-meteo.com/v1/forecast"
    timeout: int = Field(default=30, ge=1)
    retry_attempts: int = Field(default=3, ge=1)
    backoff_factor: float = Field(default=0.2, ge=0.0)


class GeocodingConfig(BaseModel, frozen=True):
    """Geocoding API configuration."""

    enabled: bool = False
    api_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    language: str = "tr"
    count: int = Field(default=1, ge=1)


class WeatherCacheConfig(BaseModel, frozen=True):
    """Weather data cache settings."""

    backend: Literal["sqlite"] = "sqlite"
    path: str = "data/external/weather_cache.db"
    ttl_hours: int = Field(default=6, ge=1)


class OpenMeteoConfig(BaseModel, frozen=True):
    """OpenMeteo full configuration."""

    api: OpenMeteoApiConfig = Field(default_factory=OpenMeteoApiConfig)
    variables: list[str] = Field(
        default_factory=lambda: [
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
    )
    cache: WeatherCacheConfig = Field(default_factory=WeatherCacheConfig)
    geocoding: GeocodingConfig = Field(default_factory=GeocodingConfig)
