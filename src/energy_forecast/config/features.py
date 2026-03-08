"""Feature engineering configuration models (calendar, consumption, weather, solar, EPIAS)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AnticipationConfig",
    "BusinessHoursConfig",
    "CalendarConfig",
    # Weather
    "ComfortIndexConfig",
    "ConsumptionConfig",
    # Consumption
    "ConsumptionLagConfig",
    # Calendar
    "CyclicalPeriod",
    "EpiasConfig",
    "EpiasExpandingConfig",
    "EpiasLagConfig",
    "EpiasRollingConfig",
    # EPIAS
    "EpiasSourceConfig",
    "EwmaConfig",
    "ExpandingConfig",
    "ExtremeFlagsConfig",
    # Combined
    "FeaturesConfig",
    "GenerationCompositesConfig",
    "GenerationConfig",
    "GenerationExpandingConfig",
    "GenerationLagConfig",
    "GenerationRollingConfig",
    "HeatIndexConfig",
    "HolidaysConfig",
    "MomentumConfig",
    "QuadraticTemperatureConfig",
    "QuantileConfig",
    "RollingConfig",
    "SolarConfig",
    "SolarLagRangeConfig",
    "SolarLeadConfig",
    # Solar
    "SolarLocationConfig",
    "SolarPanelConfig",
    "SplineSeasonalityConfig",
    "TargetEncodingConfig",
    "TempDeviationConfig",
    "TrendRatioConfig",
    "TrendRatioPairConfig",
    "WeatherFeaturesConfig",
    "WeatherInteractionsConfig",
    "WeatherLagsConfig",
    "WeatherRollingConfig",
    "WeatherSeverityConfig",
    "WeatherThresholdsConfig",
]


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Consumption
# ---------------------------------------------------------------------------


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
    """Hour x DayOfWeek target encoding feature parameters."""

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


# ---------------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# EPIAS
# ---------------------------------------------------------------------------


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
            "Real_Time_Consumption",
            "DAM_Purchase",
            "Load_Forecast",
        ]
    )
    lags: EpiasLagConfig = Field(default_factory=EpiasLagConfig)
    rolling: EpiasRollingConfig = Field(default_factory=EpiasRollingConfig)
    expanding: EpiasExpandingConfig = Field(default_factory=EpiasExpandingConfig)
    drop_raw: bool = True
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# ---------------------------------------------------------------------------
# Combined features
# ---------------------------------------------------------------------------


class FeaturesConfig(BaseModel, frozen=True):
    """All feature engineering configs combined."""

    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    consumption: ConsumptionConfig = Field(default_factory=ConsumptionConfig)
    weather: WeatherFeaturesConfig = Field(default_factory=WeatherFeaturesConfig)
    solar: SolarConfig = Field(default_factory=SolarConfig)
    epias: EpiasConfig = Field(default_factory=EpiasConfig)
