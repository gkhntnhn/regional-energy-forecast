"""Root Settings aggregator and YAML config loader functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from energy_forecast.config.api import (
    ApiConfig,
    DatabaseConfig,
    EnvConfig,
    MonitoringConfig,
)
from energy_forecast.config.features import FeaturesConfig
from energy_forecast.config.general import (
    CityConfig,
    DataLoaderConfig,
    EpiasApiConfig,
    ForecastConfig,
    LoggingConfig,
    OpenMeteoConfig,
    PipelineConfig,
    ProjectConfig,
    RegionConfig,
    TrainingPathsConfig,
    _load_yaml,
)
from energy_forecast.config.models import (
    CatBoostConfig,
    EnsembleConfig,
    HyperparameterConfig,
    ProphetConfig,
    TFTConfig,
)

__all__ = [
    "_DEFAULT_REGION",
    "Settings",
    "get_default_config",
    "load_config",
]


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------


class Settings(BaseModel, frozen=True):
    """Root config — aggregates all YAML configs + env vars."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    region: RegionConfig
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    paths: TrainingPathsConfig = Field(default_factory=TrainingPathsConfig)
    epias_api: EpiasApiConfig = Field(default_factory=EpiasApiConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    data_loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    openmeteo: OpenMeteoConfig = Field(default_factory=OpenMeteoConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    prophet: ProphetConfig = Field(default_factory=ProphetConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    hyperparameters: HyperparameterConfig = Field(
        default_factory=HyperparameterConfig,
    )
    api: ApiConfig = Field(default_factory=ApiConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

_DEFAULT_REGION = RegionConfig(
    name="Uludag",
    cities=[
        CityConfig(name="Bursa", weight=0.60, latitude=40.183, longitude=29.050),
        CityConfig(name="Balikesir", weight=0.24, latitude=39.653, longitude=27.886),
        CityConfig(name="Yalova", weight=0.10, latitude=40.655, longitude=29.272),
        CityConfig(name="Canakkale", weight=0.06, latitude=40.146, longitude=26.402),
    ],
)


def _load_monitoring(config_dir: Path) -> dict[str, Any]:
    """Load monitoring config (optional, defaults if missing)."""
    path = config_dir / "monitoring.yaml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def _build_settings_dict(config_dir: Path) -> dict[str, Any]:
    """Read all YAML files from config_dir and build a merged dict."""
    # Top-level YAML files
    settings_data = _load_yaml(config_dir / "settings.yaml")
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    data_loader_data = _load_yaml(config_dir / "data_loader.yaml")
    openmeteo_data = _load_yaml(config_dir / "openmeteo.yaml")

    # API config (optional, use defaults if not found)
    api_yaml_path = config_dir / "api.yaml"
    api_data = _load_yaml(api_yaml_path) if api_yaml_path.exists() else {}

    # Feature configs
    features_dir = config_dir / "features"
    calendar_data = _load_yaml(features_dir / "calendar.yaml")
    consumption_data = _load_yaml(features_dir / "consumption.yaml")
    weather_data = _load_yaml(features_dir / "weather.yaml")
    solar_data = _load_yaml(features_dir / "solar.yaml")
    epias_data = _load_yaml(features_dir / "epias.yaml")

    # Model configs
    models_dir = config_dir / "models"
    catboost_data = _load_yaml(models_dir / "catboost.yaml")
    prophet_data = _load_yaml(models_dir / "prophet.yaml")
    tft_data = _load_yaml(models_dir / "tft.yaml")
    ensemble_data = _load_yaml(models_dir / "ensemble.yaml")
    hyperparams_data = _load_yaml(models_dir / "hyperparameters.yaml")

    return {
        "project": settings_data.get("project", {}),
        "logging": settings_data.get("logging", {}),
        "region": settings_data.get("region", {}),
        "forecast": settings_data.get("forecast", {}),
        "paths": settings_data.get("paths", {}),
        "epias_api": settings_data.get("epias_api", {}),
        "pipeline": pipeline_data.get("pipeline", {}),
        "data_loader": {
            "excel": data_loader_data.get("excel", {}),
            "validation": data_loader_data.get("validation", {}),
            "paths": data_loader_data.get("paths", {}),
            "date_format": data_loader_data.get("excel", {}).get("date_format", "%Y-%m-%d"),
            "time_range": data_loader_data.get("excel", {}).get("time_range", [0, 23]),
            "unit": data_loader_data.get("excel", {}).get("unit", "MWh"),
        },
        "openmeteo": openmeteo_data,
        "features": {
            "calendar": calendar_data,
            "consumption": consumption_data,
            "weather": weather_data,
            "solar": solar_data,
            "epias": epias_data,
        },
        "catboost": catboost_data,
        "prophet": prophet_data,
        "tft": tft_data,
        "ensemble": ensemble_data,
        "hyperparameters": {
            "catboost": {
                "n_trials": hyperparams_data.get("catboost", {}).get("n_trials", 50),
                "search_space": hyperparams_data.get("catboost", {}).get(
                    "search_space", {}
                ),
            },
            "prophet": {
                "n_trials": hyperparams_data.get("prophet", {}).get("n_trials", 30),
                "search_space": hyperparams_data.get("prophet", {}).get(
                    "search_space", {}
                ),
            },
            "tft": {
                "n_trials": hyperparams_data.get("tft", {}).get("n_trials", 20),
                "search_space": hyperparams_data.get("tft", {}).get(
                    "search_space", {}
                ),
            },
            "cross_validation": hyperparams_data.get("cross_validation", {}),
            "target_col": hyperparams_data.get("target_col", "consumption"),
        },
        "api": {
            "host": api_data.get("api", {}).get("host", "0.0.0.0"),
            "port": api_data.get("api", {}).get("port", 8000),
            "rate_limit": api_data.get("api", {}).get("rate_limit", "10/minute"),
            "cors_origins": api_data.get("api", {}).get("cors_origins", ["*"]),
            "files": api_data.get("files", {}),
            "email": api_data.get("email", {}),
        },
        "database": api_data.get("database", {}),
        "monitoring": _load_monitoring(config_dir),
    }


def load_config(config_dir: Path | None = None) -> Settings:
    """Load all YAML configs and return a validated Settings object.

    Args:
        config_dir: Path to the ``configs/`` directory.
            Defaults to ``configs/`` relative to cwd.

    Returns:
        Fully validated Settings instance.
    """
    config_dir = config_dir or Path("configs")
    merged = _build_settings_dict(config_dir)
    return Settings(**merged)


def get_default_config() -> Settings:
    """Create Settings with all default values (no YAML files required).

    Useful for tests and development.
    """
    return Settings(region=_DEFAULT_REGION)
