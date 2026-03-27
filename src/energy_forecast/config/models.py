"""Model configuration: CatBoost, TFT, TSMixerx, Ensemble, and hyperparameters."""

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
    # Hyperparameters
    "SearchParamConfig",
    "StackingConfig",
    "StackingMetaLearnerConfig",
    # TFT
    "TFTArchitectureConfig",
    "TFTConfig",
    "TFTCovariatesConfig",
    "TFTOptimizationConfig",
    "TFTTrainingConfig",
    # TSMixerx
    "TSMixerxArchitectureConfig",
    "TSMixerxConfig",
    "TSMixerxCovariatesConfig",
    "TSMixerxOptimizationConfig",
    "TSMixerxTrainingConfig",
]


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


class CatBoostTrainingConfig(BaseModel, frozen=True):
    """CatBoost training parameters."""

    task_type: Literal["CPU", "GPU"] = "CPU"
    iterations: int = Field(default=10000, ge=100)
    learning_rate: float = Field(default=0.05, gt=0.0, lt=1.0)
    depth: int = Field(default=6, ge=1, le=16)
    loss_function: str = "RMSE"
    eval_metric: str = "MAPE"
    early_stopping_rounds: int = Field(default=100, ge=1)
    bootstrap_type: str | None = "MVS"
    has_time: bool = True
    random_seed: int = 42
    verbose: int = 500


class CatBoostNanHandling(BaseModel, frozen=True):
    """CatBoost NaN handling strategy."""

    categorical: str = "missing"


class CatBoostConfig(BaseModel, frozen=True):
    """CatBoost model configuration."""

    training: CatBoostTrainingConfig = Field(default_factory=CatBoostTrainingConfig)
    selected_features_path: str | None = Field(
        default=None,
        description="Path to JSON file with selected feature names. None = use all.",
    )
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
            # Season / solar
            "is_cooling_season",
            "is_heating_season",
            "sol_is_daylight",
            "is_new_year",
            "dst_transition",
        ]
    )
    nan_handling: CatBoostNanHandling = Field(default_factory=CatBoostNanHandling)
    best_params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TFT
# NOTE: Pydantic defaults are synced with YAML production values (tft.yaml).
# Exception: accelerator="auto" (safe fallback — YAML overrides to "gpu" for RunPod).
# YAML always takes precedence at runtime via load_config().
# ---------------------------------------------------------------------------


class TFTArchitectureConfig(BaseModel, frozen=True):
    """TFT network architecture (NeuralForecast API)."""

    hidden_size: int = Field(default=64, ge=1)
    n_head: int = Field(default=4, ge=1)
    n_rnn_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class TFTTrainingConfig(BaseModel, frozen=True):
    """TFT training parameters (NeuralForecast API)."""

    encoder_length: int = Field(default=168, ge=1)
    prediction_length: int = Field(default=48, ge=1)
    max_steps: int = Field(default=3000, ge=1)
    windows_batch_size: int = Field(default=64, ge=1)
    step_size: int = Field(default=12, ge=1)
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
            "apparent_temperature",
            "holiday_duration",
            "tatil_tipi",
            "day_of_week_sin",
            "wth_hdd",
            "is_weekend",
            "is_new_year",
            "dst_transition",
        ]
    )
    time_varying_unknown: list[str] = Field(
        default_factory=lambda: [
            "consumption_lag_168",
            "consumption_lag_336",
            "consumption_week_ratio",
            "consumption_lag_48",
            "consumption_momentum_168",
            "temperature_2m_window_24_max",
            "consumption_pct_change_168",
        ]
    )


class TFTOptimizationConfig(BaseModel, frozen=True):
    """TFT optimization settings."""

    optuna_splits: int = Field(default=12, ge=1)
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
    best_params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TSMixerx
# ---------------------------------------------------------------------------


class TSMixerxArchitectureConfig(BaseModel, frozen=True):
    """TSMixerx architecture parameters."""

    n_block: int = Field(default=2, ge=1)
    ff_dim: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    input_size: int = Field(default=168, ge=1)
    revin: bool = True


class TSMixerxTrainingConfig(BaseModel, frozen=True):
    """TSMixerx training parameters (NeuralForecast API)."""

    prediction_length: int = Field(default=48, ge=1)
    max_steps: int = Field(default=3000, ge=1)
    windows_batch_size: int = Field(default=64, ge=1)
    step_size: int = Field(default=12, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    early_stop_patience_steps: int = Field(default=200, ge=-1)  # -1 disables
    val_check_steps: int = Field(default=50, ge=1)
    random_seed: int = 42
    accelerator: Literal["cpu", "gpu", "auto"] = "auto"
    num_workers: int = Field(default=4, ge=0)
    enable_progress_bar: bool = True
    scaler_type: str = "robust"


class TSMixerxCovariatesConfig(BaseModel, frozen=True):
    """TSMixerx covariate specification (futr_exog_list / hist_exog_list)."""

    futr_exog: list[str] = Field(
        default_factory=lambda: [
            "apparent_temperature",
            "holiday_duration",
            "tatil_tipi",
            "day_of_week_sin",
            "wth_hdd",
            "is_weekend",
            "is_new_year",
            "dst_transition",
        ]
    )
    hist_exog: list[str] = Field(
        default_factory=lambda: [
            "consumption_lag_168",
            "consumption_lag_336",
            "consumption_week_ratio",
            "consumption_lag_48",
            "consumption_momentum_168",
            "temperature_2m_window_24_max",
            "consumption_pct_change_168",
        ]
    )


class TSMixerxOptimizationConfig(BaseModel, frozen=True):
    """TSMixerx optimization settings."""

    optuna_splits: int = Field(default=12, ge=1)
    n_jobs: int = Field(default=1, ge=1)
    val_size_hours: int = Field(default=720, ge=24)


class TSMixerxConfig(BaseModel, frozen=True):
    """TSMixerx model configuration (point forecast, MAE loss)."""

    architecture: TSMixerxArchitectureConfig = Field(
        default_factory=TSMixerxArchitectureConfig,
    )
    training: TSMixerxTrainingConfig = Field(default_factory=TSMixerxTrainingConfig)
    covariates: TSMixerxCovariatesConfig = Field(
        default_factory=TSMixerxCovariatesConfig,
    )
    optimization: TSMixerxOptimizationConfig = Field(
        default_factory=TSMixerxOptimizationConfig,
    )
    best_params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class EnsembleWeightsConfig(BaseModel, frozen=True):
    """Default weights for ensemble models.

    Weights are auto-normalized to sum=1 based on active models at runtime.
    """

    catboost: float = Field(default=0.33, ge=0.0, le=1.0)
    tft: float = Field(default=0.34, ge=0.0, le=1.0)
    tsmixerx: float = Field(default=0.33, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_valid(self) -> Self:
        total = self.catboost + self.tft + self.tsmixerx
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
            "tft": self.tft,
            "tsmixerx": self.tsmixerx,
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

    catboost: tuple[float, float] = (0.2, 0.7)
    tft: tuple[float, float] = (0.2, 0.7)
    tsmixerx: tuple[float, float] = (0.1, 0.5)


class EnsembleOptimizationConfig(BaseModel, frozen=True):
    """Weight optimization settings."""

    enabled: bool = True
    metric: str = "mape"
    bounds: EnsembleWeightBoundsConfig = Field(default_factory=EnsembleWeightBoundsConfig)


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
    verbose: int = 50


class StackingConfig(BaseModel, frozen=True):
    """Stacking ensemble configuration."""

    meta_learner: StackingMetaLearnerConfig = Field(default_factory=StackingMetaLearnerConfig)
    val_ratio: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Fraction of OOF data used for meta-learner validation (temporal split).",
    )
    context_features: list[str] = Field(
        default_factory=lambda: [
            "hour",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "month",
        ]
    )


class EnsembleConfig(BaseModel, frozen=True):
    """Ensemble model configuration."""

    mode: str = "stacking"
    active_models: list[str] = Field(default_factory=lambda: ["catboost", "tft", "tsmixerx"])
    weights: EnsembleWeightsConfig = Field(default_factory=EnsembleWeightsConfig)
    optimization: EnsembleOptimizationConfig = Field(
        default_factory=EnsembleOptimizationConfig,
    )
    stacking: StackingConfig = Field(default_factory=StackingConfig)
    fallback: EnsembleFallbackConfig = Field(default_factory=EnsembleFallbackConfig)

    @field_validator("active_models")
    @classmethod
    def _valid_model_names(cls, v: list[str]) -> list[str]:
        valid = {"catboost", "tft", "tsmixerx"}
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
    tft: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    tsmixerx: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    cross_validation: CrossValidationConfig = Field(
        default_factory=CrossValidationConfig,
    )
    target_col: str = "consumption"
    skip_validation_after_optuna: bool = False
