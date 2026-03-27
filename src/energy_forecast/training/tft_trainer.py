"""TFT training pipeline: TSCV + Optuna + MLflow.

Thin subclass of NeuralForecastTrainer that provides TFT-specific
configuration building and forecaster construction.  All orchestration
logic (TSCV, Optuna, MLflow, OOF cache) lives in the base class.
"""

from __future__ import annotations

from typing import Any

from energy_forecast.config import Settings
from energy_forecast.models.tft import TFTForecaster
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.nf_base_trainer import (
    NeuralForecastTrainer,
    NFPipelineResult,
    NFTrainingResult,
)
from energy_forecast.training.results import SplitResult

# Backward compatibility aliases — ensemble_trainer and tests import these.
TFTTrainingResult = NFTrainingResult
TFTPipelineResult = NFPipelineResult
TFTSplitResult = SplitResult


# ---------------------------------------------------------------------------
# TFTTrainer
# ---------------------------------------------------------------------------


class TFTTrainer(NeuralForecastTrainer):
    """TFT training pipeline with TSCV, Optuna, and MLflow.

    Inherits all orchestration from ``NeuralForecastTrainer`` and implements
    TFT-specific config building and forecaster creation.

    Args:
        settings: Full application settings.
        tracker: MLflow experiment tracker (disabled by default).
        force_hpo: Force Optuna HPO even when best_params exist.
    """

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
        *,
        force_hpo: bool = False,
    ) -> None:
        super().__init__(settings, tracker, force_hpo=force_hpo)
        self._tft_config = settings.tft
        self._search_config = settings.hyperparameters.tft

    # -- Abstract property implementations -----------------------------------

    @property
    def _model_name(self) -> str:
        return "tft"

    @property
    def _model_config(self) -> Any:
        return self._tft_config

    @property
    def _hp_search_config(self) -> Any:
        return self._search_config

    # -- Abstract method implementations -------------------------------------

    def _build_nf_config(self, params: dict[str, Any]) -> Any:
        """Build TFT config with Optuna-suggested parameters."""
        from energy_forecast.config import (
            TFTArchitectureConfig,
            TFTConfig,
            TFTCovariatesConfig,
            TFTTrainingConfig,
        )

        base = self._tft_config

        arch_params = {
            "hidden_size": params.get("hidden_size", base.architecture.hidden_size),
            "n_head": params.get("n_head", base.architecture.n_head),
            "n_rnn_layers": params.get("n_rnn_layers", base.architecture.n_rnn_layers),
            "dropout": params.get("dropout", base.architecture.dropout),
        }

        train_params = {
            "encoder_length": base.training.encoder_length,
            "prediction_length": base.training.prediction_length,
            "windows_batch_size": params.get(
                "windows_batch_size", base.training.windows_batch_size
            ),
            "max_steps": base.training.max_steps,
            "step_size": base.training.step_size,
            "learning_rate": params.get("learning_rate", base.training.learning_rate),
            "early_stop_patience_steps": base.training.early_stop_patience_steps,
            "val_check_steps": base.training.val_check_steps,
            "gradient_clip_val": base.training.gradient_clip_val,
            "random_seed": base.training.random_seed,
            "accelerator": base.training.accelerator,
            "num_workers": base.training.num_workers,
            "enable_progress_bar": base.training.enable_progress_bar,
            "precision": base.training.precision,
            "scaler_type": base.training.scaler_type,
            "rnn_type": base.training.rnn_type,
        }

        return TFTConfig(
            architecture=TFTArchitectureConfig(**arch_params),
            training=TFTTrainingConfig(**train_params),
            covariates=TFTCovariatesConfig(
                time_varying_known=list(base.covariates.time_varying_known),
                time_varying_unknown=list(base.covariates.time_varying_unknown),
            ),
            quantiles=list(base.quantiles),
            loss=base.loss,
        )

    def _create_forecaster(self, config: Any) -> Any:
        """Create TFTForecaster instance."""
        return TFTForecaster(config)

    def _get_futr_exog_list(self) -> list[str]:
        return list(self._tft_config.covariates.time_varying_known)

    def _get_hist_exog_list(self) -> list[str]:
        return list(self._tft_config.covariates.time_varying_unknown)

    # -- Optional hook override ----------------------------------------------

    def _log_model_artifact(self, model: Any) -> None:
        """Log TFT model artifact to MLflow."""
        self._tracker.log_tft_model(model, "tft_model")
