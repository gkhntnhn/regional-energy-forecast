"""Model configuration: CatBoost, Prophet, TFT, Ensemble, and hyperparameters."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "CatBoostConfig",
    "CatBoostNanHandling",
    # CatBoost
    "CatBoostTrainingConfig",
    "CrossValidationConfig",
    "EnsembleConfig",
    "EnsembleFallbackConfig",
    "EnsembleOptimizationConfig",
    "EnsembleWeightBoundsConfig",
    # Ensemble
    "EnsembleWeightsConfig",
    "HyperparameterConfig",
    "ModelSearchConfig",
    "ProphetChangepointConfig",
    "ProphetConfig",
    "ProphetHolidaysConfig",
    "ProphetOptimizationConfig",
    "ProphetRegressorConfig",
    "ProphetSeasonalityConfig",
    "ProphetUncertaintyConfig",
    # Hyperparameters
    "SearchParamConfig",
    # Prophet
    "SeasonalityPeriodConfig",
    "StackingConfig",
    "StackingMetaLearnerConfig",
    # TFT
    "TFTArchitectureConfig",
    "TFTConfig",
    "TFTCovariatesConfig",
    "TFTOptimizationConfig",
    "TFTTrainingConfig",
]


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


class CatBoostTrainingConfig(BaseModel, frozen=True):
    """CatBoost training parameters."""

    task_type: Literal["CPU", "GPU"] = "CPU"
    iterations: int = Field(default=5000, ge=100)
    learning_rate: float = Field(default=0.05, gt=0.0, lt=1.0)
    depth: int = Field(default=6, ge=1, le=16)
    loss_function: str = "RMSE"
    eval_metric: str = "MAPE"
    early_stopping_rounds: int = Field(default=100, ge=1)
    has_time: bool = True
    random_seed: int = 42
    verbose: int = 500


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
            "holiday_duration",
            # Interaction (flag x hour)
            "is_holiday_x_hour",
            "is_ramadan_x_hour",
            "is_weekend_x_hour",
            # Time-period flags
            "is_business_hours",
            "is_peak",
            "is_ramp_morning",
            "is_friday",
            "is_monday",
            "is_sunday",
            # Weather
            "weather_code",
            "weather_group",
            "wth_extreme_cold",
            # Season / solar
            "is_cooling_season",
        ]
    )
    nan_handling: CatBoostNanHandling = Field(default_factory=CatBoostNanHandling)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------


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
            ProphetRegressorConfig(name="apparent_temperature", mode="multiplicative"),
            ProphetRegressorConfig(name="relative_humidity_2m", mode="additive"),
            ProphetRegressorConfig(name="shortwave_radiation", mode="multiplicative"),
            ProphetRegressorConfig(name="wth_cdd", mode="multiplicative"),
            ProphetRegressorConfig(name="wth_hdd", mode="multiplicative"),
            # Deterministic (calendar/solar)
            ProphetRegressorConfig(name="is_weekend", mode="multiplicative"),
            ProphetRegressorConfig(name="is_sunday", mode="multiplicative"),
            ProphetRegressorConfig(name="is_holiday", mode="multiplicative"),
            ProphetRegressorConfig(name="is_business_hours", mode="multiplicative"),
            ProphetRegressorConfig(name="sol_elevation", mode="multiplicative"),
        ]
    )
    changepoint: ProphetChangepointConfig = Field(
        default_factory=ProphetChangepointConfig,
    )
    uncertainty: ProphetUncertaintyConfig = Field(
        default_factory=ProphetUncertaintyConfig,
    )


# ---------------------------------------------------------------------------
# TFT
# ---------------------------------------------------------------------------


class TFTArchitectureConfig(BaseModel, frozen=True):
    """TFT network architecture (NeuralForecast API)."""

    hidden_size: int = Field(default=64, ge=1)
    n_head: int = Field(default=2, ge=1)
    n_rnn_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class TFTTrainingConfig(BaseModel, frozen=True):
    """TFT training parameters (NeuralForecast API)."""

    encoder_length: int = Field(default=168, ge=1)
    prediction_length: int = Field(default=48, ge=1)
    max_steps: int = Field(default=2000, ge=1)
    windows_batch_size: int = Field(default=1024, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    early_stop_patience_steps: int = Field(default=200, ge=-1)  # -1 disables
    val_check_steps: int = Field(default=50, ge=1)
    gradient_clip_val: float = Field(default=0.1, gt=0.0)
    random_seed: int = 42
    accelerator: Literal["cpu", "gpu", "auto"] = "auto"
    num_workers: int = Field(default=4, ge=0)
    enable_progress_bar: bool = True
    precision: str = "bf16-mixed"
    scaler_type: str = "robust"
    rnn_type: str = "lstm"


class TFTCovariatesConfig(BaseModel, frozen=True):
    """TFT covariate specification (futr_exog_list / hist_exog_list)."""

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
            "wth_cdd",
            "wth_hdd",
            "hdd_x_hour",
            "temp_x_hour",
            "cdd_x_hour",
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
            "temperature_2m_window_24_max",
            "temperature_2m_window_12_max",
            "temperature_2m_window_6_mean",
        ]
    )


class TFTOptimizationConfig(BaseModel, frozen=True):
    """TFT optimization settings."""

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


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


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
    (e.g. Oct->train end, Nov->val, Dec->test).
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
