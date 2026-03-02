"""Pydantic V2 config system — YAML loader with validation.

Loads all configs from ``configs/`` directory, validates via Pydantic models,
and provides a single ``Settings`` object for the entire application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


# ---------------------------------------------------------------------------
# Feature configs
# ---------------------------------------------------------------------------

# -- Calendar --


class CyclicalPeriod(BaseModel, frozen=True):
    """Period for cyclical encoding."""

    period: int = Field(ge=1)


class BusinessHoursConfig(BaseModel, frozen=True):
    """Business and peak hour definitions."""

    start: int = Field(default=8, ge=0, le=23)
    end: int = Field(default=18, ge=0, le=23)
    peak_start: int = Field(default=17, ge=0, le=23)
    peak_end: int = Field(default=22, ge=0, le=23)


class HolidaysConfig(BaseModel, frozen=True):
    """Holiday feature settings."""

    path: str = "data/static/turkish_holidays.parquet"
    include_ramadan: bool = True
    bridge_days: bool = True


class AnticipationConfig(BaseModel, frozen=True):
    """Holiday anticipation feature settings."""

    enabled: bool = False
    windows: list[int] = Field(default_factory=lambda: [3, 7, 15])


class SplineSeasonalityConfig(BaseModel, frozen=True):
    """Periodic spline encoding settings."""

    enabled: bool = False
    n_splines: int = Field(default=12, ge=2)


class CalendarConfig(BaseModel, frozen=True):
    """Calendar feature engineering parameters."""

    datetime: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "extract": [
                "hour",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "week_of_year",
                "month",
                "quarter",
                "year",
            ],
        }
    )
    cyclical: dict[str, CyclicalPeriod] = Field(
        default_factory=lambda: {
            "hour": CyclicalPeriod(period=24),
            "day_of_week": CyclicalPeriod(period=7),
            "month": CyclicalPeriod(period=12),
            "day_of_year": CyclicalPeriod(period=365),
        }
    )
    holidays: HolidaysConfig = Field(default_factory=HolidaysConfig)
    anticipation: AnticipationConfig = Field(default_factory=AnticipationConfig)
    spline_seasonality: SplineSeasonalityConfig = Field(
        default_factory=SplineSeasonalityConfig,
    )
    business_hours: BusinessHoursConfig = Field(default_factory=BusinessHoursConfig)
    disabled_features: list[str] = Field(default_factory=list)


# -- Consumption --


class ConsumptionLagConfig(BaseModel, frozen=True):
    """Consumption lag feature parameters. min_lag >= 48 enforced."""

    min_lag: int = Field(default=48, ge=48)
    values: list[int] = Field(default_factory=lambda: [48, 72, 96, 168, 336, 720])

    @field_validator("values")
    @classmethod
    def _all_lags_ge_min(cls, v: list[int]) -> list[int]:
        for lag in v:
            if lag < 48:
                msg = f"Lag {lag} < 48 — data leakage risk!"
                raise ValueError(msg)
        return v


class RollingConfig(BaseModel, frozen=True):
    """Rolling window parameters."""

    windows: list[int] = Field(default_factory=lambda: [24, 48, 168, 336, 720])
    functions: list[str] = Field(default_factory=lambda: ["mean", "std", "min", "max"])


class EwmaConfig(BaseModel, frozen=True):
    """Exponential weighted moving average parameters."""

    spans: list[int] = Field(default_factory=lambda: [24, 48, 168])


class ExpandingConfig(BaseModel, frozen=True):
    """Expanding window parameters. min_periods >= 48 enforced."""

    min_periods: int = Field(default=48, ge=48)
    functions: list[str] = Field(default_factory=lambda: ["mean", "std"])

    @field_validator("min_periods")
    @classmethod
    def _min_periods_ge_48(cls, v: int) -> int:
        if v < 48:
            msg = f"Expanding min_periods {v} < 48 — data leakage risk!"
            raise ValueError(msg)
        return v


class MomentumConfig(BaseModel, frozen=True):
    """Momentum feature parameters."""

    periods: list[int] = Field(default_factory=lambda: [24, 168])


class QuantileConfig(BaseModel, frozen=True):
    """Quantile feature parameters."""

    quantiles: list[float] = Field(default_factory=lambda: [0.25, 0.50, 0.75])
    window: int = Field(default=168, ge=1)


class TrendRatioPairConfig(BaseModel, frozen=True):
    """Single trend ratio pair (numerator_lag / denominator_lag)."""

    numerator_lag: int = Field(ge=48)
    denominator_lag: int = Field(ge=48)


class TrendRatioConfig(BaseModel, frozen=True):
    """Trend ratio feature parameters."""

    pairs: list[TrendRatioPairConfig] = Field(
        default_factory=lambda: [TrendRatioPairConfig(numerator_lag=168, denominator_lag=336)]
    )


class TargetEncodingConfig(BaseModel, frozen=True):
    """Hour×DayOfWeek target encoding feature parameters."""

    enabled: bool = False


class ConsumptionConfig(BaseModel, frozen=True):
    """Consumption feature engineering parameters."""

    lags: ConsumptionLagConfig = Field(default_factory=ConsumptionLagConfig)
    rolling: RollingConfig = Field(default_factory=RollingConfig)
    ewma: EwmaConfig = Field(default_factory=EwmaConfig)
    expanding: ExpandingConfig = Field(default_factory=ExpandingConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    quantile: QuantileConfig = Field(default_factory=QuantileConfig)
    trend_ratio: TrendRatioConfig = Field(default_factory=TrendRatioConfig)
    target_encoding: TargetEncodingConfig = Field(default_factory=TargetEncodingConfig)


# -- Weather features --


class ComfortIndexConfig(BaseModel, frozen=True):
    """Comfort index calculation method."""

    method: str = "apparent_temperature"


class WeatherSeverityConfig(BaseModel, frozen=True):
    """Weather severity thresholds."""

    enabled: bool = True
    wind_threshold: float = 25.0
    precip_threshold: float = 5.0


class ExtremeFlagsConfig(BaseModel, frozen=True):
    """Selective enable/disable for extreme weather flags."""

    cold: bool = True
    hot: bool = True
    wind: bool = True
    precip: bool = True


class WeatherThresholdsConfig(BaseModel, frozen=True):
    """Temperature and weather thresholds."""

    hdd_base: float = 18.0
    cdd_base: float = 24.0
    extreme_cold: float = 0.0
    extreme_hot: float = 35.0
    high_wind: float = 25.0
    extreme_flags: ExtremeFlagsConfig = Field(default_factory=ExtremeFlagsConfig)


class WeatherRollingConfig(BaseModel, frozen=True):
    """Weather rolling window parameters."""

    windows: list[int] = Field(default_factory=lambda: [6, 12, 24])
    functions: list[str] = Field(default_factory=lambda: ["mean", "min", "max"])


class WeatherLagsConfig(BaseModel, frozen=True):
    """Weather lag feature settings."""

    enabled: bool = False
    hours: list[int] = Field(default_factory=lambda: [6, 12, 18, 24, 30, 36, 42, 48])
    columns: list[str] = Field(
        default_factory=lambda: ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    )


class QuadraticTemperatureConfig(BaseModel, frozen=True):
    """Quadratic temperature feature settings."""

    enabled: bool = False


class HeatIndexConfig(BaseModel, frozen=True):
    """Heat index (Steadman) feature settings."""

    enabled: bool = False
    threshold: float = Field(default=27.0, ge=20.0, le=40.0)


class TempDeviationConfig(BaseModel, frozen=True):
    """Temperature deviation from expanding mean."""

    enabled: bool = False


class WeatherInteractionsConfig(BaseModel, frozen=True):
    """Selective enable/disable for weather x calendar interactions."""

    cdd_x_is_peak: bool = True


class WeatherFeaturesConfig(BaseModel, frozen=True):
    """Weather feature engineering parameters."""

    thresholds: WeatherThresholdsConfig = Field(
        default_factory=WeatherThresholdsConfig,
    )
    comfort_index: ComfortIndexConfig = Field(default_factory=ComfortIndexConfig)
    rolling: WeatherRollingConfig = Field(default_factory=WeatherRollingConfig)
    weather_lags: WeatherLagsConfig = Field(default_factory=WeatherLagsConfig)
    quadratic_temperature: QuadraticTemperatureConfig = Field(
        default_factory=QuadraticTemperatureConfig,
    )
    severity: WeatherSeverityConfig = Field(default_factory=WeatherSeverityConfig)
    interactions: WeatherInteractionsConfig = Field(
        default_factory=WeatherInteractionsConfig,
    )
    heat_index: HeatIndexConfig = Field(default_factory=HeatIndexConfig)
    temp_deviation: TempDeviationConfig = Field(default_factory=TempDeviationConfig)


# -- Solar --


class SolarLocationConfig(BaseModel, frozen=True):
    """Solar calculation location."""

    latitude: float = 40.183
    longitude: float = 29.050
    altitude: int = 100
    timezone: str = "Europe/Istanbul"


class SolarPanelConfig(BaseModel, frozen=True):
    """Solar panel orientation."""

    tilt: int = 35
    azimuth: int = 180


class SolarLagRangeConfig(BaseModel, frozen=True):
    """Solar lag range for lead/lag features."""

    min: int = -10
    max: int = 10


class SolarLeadConfig(BaseModel, frozen=True):
    """Solar lead feature settings."""

    enabled: bool = True
    hours: list[int] = Field(default_factory=lambda: [1, 2, 3])
    lag_range: SolarLagRangeConfig = Field(default_factory=SolarLagRangeConfig)
    lag_columns: list[str] = Field(
        default_factory=lambda: ["sol_ghi", "sol_dni", "sol_dhi"]
    )


class SolarConfig(BaseModel, frozen=True):
    """Solar feature engineering parameters."""

    location: SolarLocationConfig = Field(default_factory=SolarLocationConfig)
    panel: SolarPanelConfig = Field(default_factory=SolarPanelConfig)
    features: list[str] = Field(
        default_factory=lambda: [
            "sol_elevation",
            "sol_azimuth",
            "sol_ghi",
            "sol_dni",
            "sol_dhi",
            "sol_poa_global",
            "sol_clearness_index",
            "sol_cloud_proxy",
            "sol_is_daylight",
            "sol_daylight_hours",
        ]
    )
    lead: SolarLeadConfig = Field(default_factory=SolarLeadConfig)
    disabled_features: list[str] = Field(default_factory=list)


# -- EPIAS --


class EpiasSourceConfig(BaseModel, frozen=True):
    """EPIAS data source paths."""

    cache_dir: str = "data/external/epias"
    file_pattern: str = "epias_market_{year}.parquet"


class EpiasLagConfig(BaseModel, frozen=True):
    """EPIAS lag feature parameters. min_lag >= 48 enforced."""

    min_lag: int = Field(default=48, ge=48)
    values: list[int] = Field(default_factory=lambda: [48, 72, 168])

    @field_validator("values")
    @classmethod
    def _all_lags_ge_min(cls, v: list[int]) -> list[int]:
        for lag in v:
            if lag < 48:
                msg = f"EPIAS lag {lag} < 48 — data leakage risk!"
                raise ValueError(msg)
        return v


class EpiasRollingConfig(BaseModel, frozen=True):
    """EPIAS rolling window parameters."""

    windows: list[int] = Field(default_factory=lambda: [24, 48, 168])
    functions: list[str] = Field(default_factory=lambda: ["mean", "std"])


class EpiasExpandingConfig(BaseModel, frozen=True):
    """EPIAS expanding window parameters."""

    min_periods: int = Field(default=48, ge=48)
    functions: list[str] = Field(default_factory=lambda: ["mean"])

    @field_validator("min_periods")
    @classmethod
    def _min_periods_ge_48(cls, v: int) -> int:
        if v < 48:
            msg = f"EPIAS expanding min_periods {v} < 48 — data leakage risk!"
            raise ValueError(msg)
        return v


class GenerationLagConfig(BaseModel, frozen=True):
    """Generation lag feature parameters. min_lag >= 48 enforced."""

    min_lag: int = Field(default=48, ge=48)
    values: list[int] = Field(default_factory=lambda: [48, 168])

    @field_validator("values")
    @classmethod
    def _all_lags_ge_min(cls, v: list[int]) -> list[int]:
        for lag in v:
            if lag < 48:
                msg = f"Generation lag {lag} < 48 — data leakage risk!"
                raise ValueError(msg)
        return v


class GenerationRollingConfig(BaseModel, frozen=True):
    """Generation rolling window parameters."""

    windows: list[int] = Field(default_factory=lambda: [24, 168])
    functions: list[str] = Field(default_factory=lambda: ["mean"])


class GenerationExpandingConfig(BaseModel, frozen=True):
    """Generation expanding window parameters."""

    min_periods: int = Field(default=48, ge=48)
    functions: list[str] = Field(default_factory=lambda: ["mean"])

    @field_validator("min_periods")
    @classmethod
    def _min_periods_ge_48(cls, v: int) -> int:
        if v < 48:
            msg = f"Generation expanding min_periods {v} < 48 — data leakage risk!"
            raise ValueError(msg)
        return v


class GenerationCompositesConfig(BaseModel, frozen=True):
    """Generation composite ratio feature settings."""

    enabled: bool = False
    renewable_vars: list[str] = Field(
        default_factory=lambda: ["gen_wind", "gen_sun", "gen_river", "gen_dammed_hydro"]
    )
    thermal_vars: list[str] = Field(
        default_factory=lambda: ["gen_natural_gas", "gen_lignite", "gen_import_coal"]
    )
    total_var: str = "gen_total"
    lag: int = Field(default=48, ge=48)


class GenerationConfig(BaseModel, frozen=True):
    """Generation feature engineering parameters."""

    variables: list[str] = Field(
        default_factory=lambda: [
            "gen_asphaltite_coal",
            "gen_biomass",
            "gen_black_coal",
            "gen_dammed_hydro",
            "gen_fueloil",
            "gen_geothermal",
            "gen_import_coal",
            "gen_import_export",
            "gen_lignite",
            "gen_lng",
            "gen_naphta",
            "gen_natural_gas",
            "gen_river",
            "gen_sun",
            "gen_total",
            "gen_wasteheat",
            "gen_wind",
        ]
    )
    lags: GenerationLagConfig = Field(default_factory=GenerationLagConfig)
    rolling: GenerationRollingConfig = Field(default_factory=GenerationRollingConfig)
    expanding: GenerationExpandingConfig = Field(default_factory=GenerationExpandingConfig)
    drop_raw: bool = True
    composites: GenerationCompositesConfig = Field(
        default_factory=GenerationCompositesConfig,
    )


class EpiasConfig(BaseModel, frozen=True):
    """EPIAS feature engineering parameters."""

    source: EpiasSourceConfig = Field(default_factory=EpiasSourceConfig)
    variables: list[str] = Field(
        default_factory=lambda: [
            "FDPP",
            "Real_Time_Consumption",
            "DAM_Purchase",
            "Bilateral_Agreement_Purchase",
            "Load_Forecast",
        ]
    )
    lags: EpiasLagConfig = Field(default_factory=EpiasLagConfig)
    rolling: EpiasRollingConfig = Field(default_factory=EpiasRollingConfig)
    expanding: EpiasExpandingConfig = Field(default_factory=EpiasExpandingConfig)
    drop_raw: bool = True
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# -- Combined features --


class FeaturesConfig(BaseModel, frozen=True):
    """All feature engineering configs combined."""

    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    consumption: ConsumptionConfig = Field(default_factory=ConsumptionConfig)
    weather: WeatherFeaturesConfig = Field(default_factory=WeatherFeaturesConfig)
    solar: SolarConfig = Field(default_factory=SolarConfig)
    epias: EpiasConfig = Field(default_factory=EpiasConfig)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


class CatBoostTrainingConfig(BaseModel, frozen=True):
    """CatBoost training parameters."""

    task_type: Literal["CPU", "GPU"] = "CPU"
    iterations: int = Field(default=2000, ge=100)
    learning_rate: float = Field(default=0.05, gt=0.0, lt=1.0)
    depth: int = Field(default=6, ge=1, le=16)
    loss_function: str = "MAE"
    eval_metric: str = "MAPE"
    early_stopping_rounds: int = Field(default=200, ge=1)
    has_time: bool = True
    random_seed: int = 42
    verbose: int = 100


class CatBoostNanHandling(BaseModel, frozen=True):
    """CatBoost NaN handling strategy."""

    categorical: str = "missing"


class CatBoostConfig(BaseModel, frozen=True):
    """CatBoost model configuration."""

    training: CatBoostTrainingConfig = Field(default_factory=CatBoostTrainingConfig)
    categorical_features: list[str] = Field(
        default_factory=lambda: [
            # Time
            "hour",
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "month",
            "quarter",
            "season",
            "year",
            # Holiday / special days
            "is_holiday",
            "is_weekend",
            "is_ramadan",
            "is_bridge_day",
            "tatil_tipi",
            "bayram_gun_no",
            # Interaction (flag x hour)
            "is_holiday_x_hour",
            "is_ramadan_x_hour",
            "is_weekend_x_hour",
            # Time-period flags
            "is_business_hours",
            "is_peak",
            "is_ramp_morning",
            "is_ramp_evening",
            "is_friday",
            "is_monday",
            "is_sunday",
            # Weather
            "weather_code",
            "weather_group",
            "wth_extreme_cold",
            "wth_extreme_hot",
            "wth_extreme_wind",
            "wth_heavy_precip",
            "wth_is_severe",
            "wth_severity",
            # Season / solar
            "is_cooling_season",
            "is_heating_season",
            "sol_is_daylight",
            "sol_daylight_hours",
        ]
    )
    nan_handling: CatBoostNanHandling = Field(default_factory=CatBoostNanHandling)


# -- Prophet --


class SeasonalityPeriodConfig(BaseModel, frozen=True):
    """Prophet Fourier order for a seasonality period."""

    fourier_order: int = Field(ge=1)


class ProphetSeasonalityConfig(BaseModel, frozen=True):
    """Prophet seasonality settings."""

    mode: Literal["additive", "multiplicative"] = "multiplicative"
    daily: SeasonalityPeriodConfig = Field(
        default_factory=lambda: SeasonalityPeriodConfig(fourier_order=15),
    )
    weekly: SeasonalityPeriodConfig = Field(
        default_factory=lambda: SeasonalityPeriodConfig(fourier_order=8),
    )
    yearly: SeasonalityPeriodConfig = Field(
        default_factory=lambda: SeasonalityPeriodConfig(fourier_order=12),
    )


class ProphetHolidaysConfig(BaseModel, frozen=True):
    """Prophet holiday settings."""

    country: str = "TR"
    include_ramadan: bool = True


class ProphetRegressorConfig(BaseModel, frozen=True):
    """Single Prophet regressor definition."""

    name: str
    mode: Literal["additive", "multiplicative"] = "additive"


class ProphetChangepointConfig(BaseModel, frozen=True):
    """Prophet changepoint settings."""

    prior_scale: float = Field(default=0.05, gt=0.0)
    n_changepoints: int = Field(default=25, ge=1)


class ProphetUncertaintyConfig(BaseModel, frozen=True):
    """Prophet uncertainty interval settings."""

    interval_width: float = Field(default=0.95, gt=0.0, lt=1.0)
    mcmc_samples: int = Field(default=0, ge=0)


class ProphetOptimizationConfig(BaseModel, frozen=True):
    """Prophet optimization settings."""

    random_seed: int = Field(default=42, ge=0)


class ProphetConfig(BaseModel, frozen=True):
    """Prophet model configuration."""

    seasonality: ProphetSeasonalityConfig = Field(
        default_factory=ProphetSeasonalityConfig,
    )
    optimization: ProphetOptimizationConfig = Field(
        default_factory=ProphetOptimizationConfig,
    )
    holidays: ProphetHolidaysConfig = Field(default_factory=ProphetHolidaysConfig)
    regressors: list[ProphetRegressorConfig] = Field(
        default_factory=lambda: [
            # Consumption lags (autoregressive signal)
            ProphetRegressorConfig(name="consumption_lag_168", mode="multiplicative"),
            ProphetRegressorConfig(name="consumption_lag_48", mode="multiplicative"),
            ProphetRegressorConfig(name="consumption_lag_720", mode="multiplicative"),
            # Weather
            ProphetRegressorConfig(name="temperature_2m", mode="multiplicative"),
            ProphetRegressorConfig(name="relative_humidity_2m", mode="additive"),
            ProphetRegressorConfig(name="wind_speed_10m", mode="additive"),
            ProphetRegressorConfig(name="shortwave_radiation", mode="multiplicative"),
            ProphetRegressorConfig(name="wth_cdd", mode="multiplicative"),
            ProphetRegressorConfig(name="wth_hdd", mode="multiplicative"),
            # Deterministic (calendar/solar)
            ProphetRegressorConfig(name="is_weekend", mode="multiplicative"),
            ProphetRegressorConfig(name="is_holiday", mode="multiplicative"),
            ProphetRegressorConfig(name="is_business_hours", mode="multiplicative"),
            ProphetRegressorConfig(name="is_peak", mode="multiplicative"),
            ProphetRegressorConfig(name="sol_elevation", mode="multiplicative"),
        ]
    )
    changepoint: ProphetChangepointConfig = Field(
        default_factory=ProphetChangepointConfig,
    )
    uncertainty: ProphetUncertaintyConfig = Field(
        default_factory=ProphetUncertaintyConfig,
    )


# -- TFT --


class TFTArchitectureConfig(BaseModel, frozen=True):
    """TFT network architecture."""

    hidden_size: int = Field(default=64, ge=1)
    attention_head_size: int = Field(default=4, ge=1)
    lstm_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    hidden_continuous_size: int = Field(default=16, ge=1)


class TFTTrainingConfig(BaseModel, frozen=True):
    """TFT training parameters."""

    encoder_length: int = Field(default=168, ge=1)
    prediction_length: int = Field(default=48, ge=1)
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    early_stop_patience: int = Field(default=10, ge=1)
    gradient_clip_val: float = Field(default=0.1, gt=0.0)
    random_seed: int = 42
    accelerator: Literal["cpu", "gpu", "auto"] = "auto"
    num_workers: int = Field(default=0, ge=0)
    enable_progress_bar: bool = True
    enable_model_summary: bool = False
    precision: str = "32-true"


class TFTCovariatesConfig(BaseModel, frozen=True):
    """TFT covariate specification for TimeSeriesDataSet."""

    time_varying_known: list[str] = Field(
        default_factory=lambda: [
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "day_of_year_sin",
            "day_of_year_cos",
            "is_weekend",
            "is_sunday",
            "is_bridge_day",
            "tatil_tipi",
            "holiday_duration",
            "bayrama_kalan_gun",
            "bayram_gun_no",
            "days_since_holiday",
            "days_until_holiday",
            "temperature_2m",
            "apparent_temperature",
            "shortwave_radiation",
            "sol_elevation",
        ]
    )
    time_varying_unknown: list[str] = Field(
        default_factory=lambda: [
            "consumption_lag_48",
            "consumption_lag_168",
            "consumption_lag_336",
            "consumption_lag_720",
            "consumption_week_ratio",
            "consumption_hourly_profile",
            "consumption_momentum_168",
            "consumption_pct_change_168",
            "consumption_trend_ratio_168_336",
            "consumption_trend_ratio_48_168",
            "consumption_window_720_std",
            "consumption_window_48_max",
            "consumption_window_336_max",
            "consumption_window_720_mean",
            "consumption_window_336_min",
            "consumption_q75_168",
            "temperature_2m_window_24_max",
            "temperature_2m_window_12_max",
            "temperature_2m_window_6_mean",
            "hdd_x_hour",
        ]
    )


class TFTOptimizationConfig(BaseModel, frozen=True):
    """TFT optimization settings."""

    fast_epochs: int = Field(default=10, ge=1)  # Deprecated: epoch-level pruning replaces this
    optuna_splits: int = Field(default=2, ge=1)
    n_jobs: int = Field(default=1, ge=1)  # Parallel Optuna trials (1=serial, 8=RunPod A100)
    val_size_hours: int = Field(default=720, ge=24)  # ~1 month (24 * 30)


class TFTConfig(BaseModel, frozen=True):
    """TFT model configuration."""

    architecture: TFTArchitectureConfig = Field(
        default_factory=TFTArchitectureConfig,
    )
    training: TFTTrainingConfig = Field(default_factory=TFTTrainingConfig)
    covariates: TFTCovariatesConfig = Field(default_factory=TFTCovariatesConfig)
    optimization: TFTOptimizationConfig = Field(
        default_factory=TFTOptimizationConfig,
    )
    quantiles: list[float] = Field(
        default_factory=lambda: [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98],
    )
    loss: str = "quantile"


# -- Ensemble --


class EnsembleWeightsConfig(BaseModel, frozen=True):
    """Default weights for ensemble models.

    Weights are auto-normalized to sum=1 based on active models at runtime.
    """

    catboost: float = Field(default=0.45, ge=0.0, le=1.0)
    prophet: float = Field(default=0.30, ge=0.0, le=1.0)
    tft: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_valid(self) -> Self:
        total = self.catboost + self.prophet + self.tft
        if total > 1.0 + 1e-6:
            msg = f"Ensemble weights cannot exceed 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self

    def get_normalized(self, active_models: list[str]) -> dict[str, float]:
        """Get weights normalized to sum=1 for active models only.

        Args:
            active_models: List of active model names.

        Returns:
            Dict mapping model name to normalized weight.
        """
        raw_weights = {
            "catboost": self.catboost,
            "prophet": self.prophet,
            "tft": self.tft,
        }
        active_weights = {m: raw_weights[m] for m in active_models if m in raw_weights}

        total = sum(active_weights.values())
        if total < 1e-6:
            # Equal weights if all are zero
            n = len(active_weights)
            return {m: 1.0 / n for m in active_weights}

        return {m: w / total for m, w in active_weights.items()}


class EnsembleWeightBoundsConfig(BaseModel, frozen=True):
    """Per-model weight bounds for optimization."""

    catboost: tuple[float, float] = (0.1, 0.8)
    prophet: tuple[float, float] = (0.1, 0.6)
    tft: tuple[float, float] = (0.1, 0.6)


class EnsembleOptimizationConfig(BaseModel, frozen=True):
    """Weight optimization settings."""

    enabled: bool = True
    metric: str = "mape"
    bounds: EnsembleWeightBoundsConfig = Field(
        default_factory=EnsembleWeightBoundsConfig
    )


class EnsembleFallbackConfig(BaseModel, frozen=True):
    """Fallback behavior when one model fails."""

    enabled: bool = True


class StackingMetaLearnerConfig(BaseModel, frozen=True):
    """CatBoost meta-learner hyperparameters for stacking ensemble."""

    depth: int = 2
    iterations: int = 500
    early_stopping_rounds: int = 30
    learning_rate: float = 0.05
    loss_function: str = "RMSE"
    l2_leaf_reg: float = 3.0
    task_type: str = "CPU"


class StackingConfig(BaseModel, frozen=True):
    """Stacking ensemble configuration."""

    meta_learner: StackingMetaLearnerConfig = Field(
        default_factory=StackingMetaLearnerConfig
    )
    context_features: list[str] = Field(
        default_factory=lambda: [
            "hour", "day_of_week", "is_weekend", "is_holiday", "month",
        ]
    )


class EnsembleConfig(BaseModel, frozen=True):
    """Ensemble model configuration."""

    mode: str = "stacking"
    active_models: list[str] = Field(
        default_factory=lambda: ["catboost", "prophet", "tft"]
    )
    weights: EnsembleWeightsConfig = Field(default_factory=EnsembleWeightsConfig)
    optimization: EnsembleOptimizationConfig = Field(
        default_factory=EnsembleOptimizationConfig,
    )
    stacking: StackingConfig = Field(default_factory=StackingConfig)
    fallback: EnsembleFallbackConfig = Field(default_factory=EnsembleFallbackConfig)

    @field_validator("active_models")
    @classmethod
    def _valid_model_names(cls, v: list[str]) -> list[str]:
        valid = {"catboost", "prophet", "tft"}
        for m in v:
            if m not in valid:
                msg = f"Unknown ensemble model: {m}. Valid: {valid}"
                raise ValueError(msg)
        if len(v) < 1:
            msg = "At least one model required in active_models"
            raise ValueError(msg)
        return v


# -- Hyperparameters --


class SearchParamConfig(BaseModel, frozen=True):
    """Single Optuna search parameter definition.

    Dynamically loaded from YAML — adding a new parameter to YAML
    requires NO code change.

    type=int:         trial.suggest_int(name, low, high, step?, log?)
    type=float:       trial.suggest_float(name, low, high, step?, log?)
    type=categorical: trial.suggest_categorical(name, choices)
    """

    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _validate_range_or_choices(self) -> Self:
        if self.type in ("int", "float"):
            if self.low is None or self.high is None:
                msg = f"type={self.type} requires low and high"
                raise ValueError(msg)
            if self.low > self.high:
                msg = f"low ({self.low}) > high ({self.high})"
                raise ValueError(msg)
            if self.log and self.step is not None:
                msg = "log=true and step are mutually exclusive"
                raise ValueError(msg)
        elif self.type == "categorical":
            if not self.choices:
                msg = "type=categorical requires non-empty choices"
                raise ValueError(msg)
        return self


class ModelSearchConfig(BaseModel, frozen=True):
    """Per-model Optuna search space and trial count.

    ``search_space`` is a dynamic dict — any parameter can be added
    via YAML without code changes.
    """

    n_trials: int = Field(default=50, ge=1)
    search_space: dict[str, SearchParamConfig] = Field(default_factory=dict)


class CrossValidationConfig(BaseModel, frozen=True):
    """Calendar-month aligned TSCV settings.

    ``val_months`` and ``test_months`` are counted as calendar months,
    NOT fixed day counts.  Each split aligns to month boundaries
    (e.g. Oct→train end, Nov→val, Dec→test).
    """

    n_splits: int = Field(default=12, ge=2)
    val_months: int = Field(default=1, ge=1)
    test_months: int = Field(default=1, ge=1)
    gap_hours: int = Field(default=0, ge=0)
    shuffle: bool = False


class HyperparameterConfig(BaseModel, frozen=True):
    """Hyperparameter tuning configuration for all models."""

    catboost: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    prophet: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    tft: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    cross_validation: CrossValidationConfig = Field(
        default_factory=CrossValidationConfig,
    )
    target_col: str = "consumption"
    skip_validation_after_optuna: bool = False


# ---------------------------------------------------------------------------
# Environment config (secrets from .env)
# ---------------------------------------------------------------------------


class EnvConfig(BaseSettings):
    """Environment variables loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    epias_username: str = ""
    epias_password: str = ""
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    sender_email: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
    aws_s3_bucket: str = ""
    aws_region: str = "eu-west-1"
    database_url: str = ""


# ---------------------------------------------------------------------------
# API Config
# ---------------------------------------------------------------------------


class ApiFilesConfig(BaseModel, frozen=True):
    """API file handling configuration."""

    upload_dir: str = "data/uploads"
    output_dir: str = "data/outputs"
    allowed_extensions: list[str] = Field(default_factory=lambda: [".xlsx", ".xls"])
    max_file_size_mb: int = Field(default=50, ge=1)
    cleanup_after_hours: int = Field(default=24, ge=1)


class ApiEmailConfig(BaseModel, frozen=True):
    """API email template configuration."""

    sender_name: str = "Energy Forecast"
    subject_template: str = "Tahmin Sonuçları - {job_id}"
    body_template: str = Field(
        default="""Merhaba,

Talep ettiğiniz 48 saatlik elektrik tüketimi tahmini ekte sunulmuştur.

İş No: {job_id}
Oluşturulma: {created_at}

İyi çalışmalar,
Energy Forecast Sistemi"""
    )


class ApiConfig(BaseModel, frozen=True):
    """API serving configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    rate_limit: str = "10/minute"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    files: ApiFilesConfig = Field(default_factory=ApiFilesConfig)
    email: ApiEmailConfig = Field(default_factory=ApiEmailConfig)


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


def _build_settings_dict(config_dir: Path) -> dict[str, Any]:
    """Read all YAML files from config_dir and build a merged dict."""
    # Top-level YAML files
    settings_data = _load_yaml(config_dir / "settings.yaml")
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    data_loader_data = _load_yaml(config_dir / "data_loader.yaml")
    openmeteo_data = _load_yaml(config_dir / "openmeteo.yaml")

    # API config (optional, use defaults if not found)
    api_yaml_path = config_dir / "api.yaml"
    if api_yaml_path.exists():
        api_data = _load_yaml(api_yaml_path)
    else:
        api_data = {}

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
