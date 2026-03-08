"""Configuration management.

Public API:
    - ``Settings``: Root config aggregating all YAML configs + env vars.
    - ``EnvConfig``: Environment variables loaded from ``.env``.
    - ``load_config()``: Load and validate all YAML configs.
    - ``get_default_config()``: Get default config (no YAML files required).

All config classes are re-exported here for convenient access via
``from energy_forecast.config import <ClassName>``.
"""

# -- general (helpers, project, pipeline, data-loader, openmeteo) -----------
# -- loader (root Settings, load functions) ---------------------------------
from energy_forecast.config._loader import (  # noqa: F401
    _DEFAULT_REGION,
    Settings,
    get_default_config,
    load_config,
)

# -- api (env, database, monitoring, API serving) ---------------------------
from energy_forecast.config.api import (  # noqa: F401
    ApiConfig,
    ApiEmailConfig,
    ApiFilesConfig,
    DatabaseConfig,
    DriftDetectionConfig,
    EnvConfig,
    MonitoringConfig,
)

# -- features (calendar, consumption, weather, solar, EPIAS) ----------------
from energy_forecast.config.features import (  # noqa: F401
    AnticipationConfig,
    BusinessHoursConfig,
    CalendarConfig,
    ComfortIndexConfig,
    ConsumptionConfig,
    ConsumptionLagConfig,
    CyclicalPeriod,
    EpiasConfig,
    EpiasExpandingConfig,
    EpiasLagConfig,
    EpiasRollingConfig,
    EpiasSourceConfig,
    EwmaConfig,
    ExpandingConfig,
    ExtremeFlagsConfig,
    FeaturesConfig,
    GenerationCompositesConfig,
    GenerationConfig,
    GenerationExpandingConfig,
    GenerationLagConfig,
    GenerationRollingConfig,
    HeatIndexConfig,
    HolidaysConfig,
    MomentumConfig,
    QuadraticTemperatureConfig,
    QuantileConfig,
    RollingConfig,
    SolarConfig,
    SolarLagRangeConfig,
    SolarLeadConfig,
    SolarLocationConfig,
    SolarPanelConfig,
    SplineSeasonalityConfig,
    TargetEncodingConfig,
    TempDeviationConfig,
    TrendRatioConfig,
    TrendRatioPairConfig,
    WeatherFeaturesConfig,
    WeatherInteractionsConfig,
    WeatherLagsConfig,
    WeatherRollingConfig,
    WeatherSeverityConfig,
    WeatherThresholdsConfig,
)
from energy_forecast.config.general import (  # noqa: F401
    CityConfig,
    DataLoaderConfig,
    DataValidationConfig,
    EpiasApiConfig,
    ExcelColumnsConfig,
    ForecastConfig,
    GeocodingConfig,
    LoggingConfig,
    OpenMeteoApiConfig,
    OpenMeteoConfig,
    PathsConfig,
    PipelineConfig,
    ProjectConfig,
    RegionConfig,
    TrainingPathsConfig,
    WeatherCacheConfig,
    _load_yaml,
)

# -- models (catboost, prophet, tft, ensemble, hyperparameters) -------------
from energy_forecast.config.models import (  # noqa: F401
    CatBoostConfig,
    CatBoostNanHandling,
    CatBoostTrainingConfig,
    CrossValidationConfig,
    EnsembleConfig,
    EnsembleFallbackConfig,
    EnsembleOptimizationConfig,
    EnsembleWeightBoundsConfig,
    EnsembleWeightsConfig,
    HyperparameterConfig,
    ModelSearchConfig,
    ProphetChangepointConfig,
    ProphetConfig,
    ProphetHolidaysConfig,
    ProphetOptimizationConfig,
    ProphetRegressorConfig,
    ProphetSeasonalityConfig,
    ProphetUncertaintyConfig,
    SearchParamConfig,
    SeasonalityPeriodConfig,
    StackingConfig,
    StackingMetaLearnerConfig,
    TFTArchitectureConfig,
    TFTConfig,
    TFTCovariatesConfig,
    TFTOptimizationConfig,
    TFTTrainingConfig,
)

__all__ = ["EnvConfig", "Settings", "get_default_config", "load_config"]
