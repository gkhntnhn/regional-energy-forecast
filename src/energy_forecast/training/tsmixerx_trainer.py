"""TSMixerx training pipeline: TSCV + Optuna + MLflow.

Thin subclass of NeuralForecastTrainer that provides TSMixerx-specific
configuration building and forecaster construction.  All orchestration
logic (TSCV, Optuna, MLflow, OOF cache) lives in the base class.
"""

from __future__ import annotations

from typing import Any

from energy_forecast.config import Settings
from energy_forecast.models.tsmixerx import TSMixerxForecaster
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.nf_base_trainer import (
    NeuralForecastTrainer,
    NFPipelineResult,
    NFTrainingResult,
)
from energy_forecast.training.results import SplitResult

# Backward compatibility aliases — ensemble_trainer and tests import these.
TSMixerxTrainingResult = NFTrainingResult
TSMixerxPipelineResult = NFPipelineResult
TSMixerxSplitResult = SplitResult


# ---------------------------------------------------------------------------
# TSMixerxTrainer
# ---------------------------------------------------------------------------


class TSMixerxTrainer(NeuralForecastTrainer):
    """TSMixerx training pipeline with TSCV, Optuna, and MLflow.

    Inherits all orchestration from ``NeuralForecastTrainer`` and implements
    TSMixerx-specific config building and forecaster creation.

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
        self._tsmixerx_config = settings.tsmixerx
        self._search_config = settings.hyperparameters.tsmixerx

    # -- Abstract property implementations -----------------------------------

    @property
    def _model_name(self) -> str:
        return "tsmixerx"

    @property
    def _model_config(self) -> Any:
        return self._tsmixerx_config

    @property
    def _hp_search_config(self) -> Any:
        return self._search_config

    # -- Abstract method implementations -------------------------------------

    def _build_nf_config(self, params: dict[str, Any]) -> Any:
        """Build TSMixerx config with Optuna-suggested parameters."""
        from energy_forecast.config import (
            TSMixerxArchitectureConfig,
            TSMixerxConfig,
            TSMixerxCovariatesConfig,
            TSMixerxTrainingConfig,
        )

        base = self._tsmixerx_config

        arch_params = {
            "n_block": params.get("n_block", base.architecture.n_block),
            "ff_dim": params.get("ff_dim", base.architecture.ff_dim),
            "dropout": params.get("dropout", base.architecture.dropout),
            "input_size": base.architecture.input_size,
            "revin": base.architecture.revin,
        }

        train_params = {
            "prediction_length": base.training.prediction_length,
            "max_steps": params.get("max_steps", base.training.max_steps),
            "windows_batch_size": params.get(
                "windows_batch_size", base.training.windows_batch_size
            ),
            "step_size": base.training.step_size,
            "learning_rate": params.get("learning_rate", base.training.learning_rate),
            "early_stop_patience_steps": base.training.early_stop_patience_steps,
            "val_check_steps": base.training.val_check_steps,
            "random_seed": base.training.random_seed,
            "accelerator": base.training.accelerator,
            "num_workers": base.training.num_workers,
            "enable_progress_bar": base.training.enable_progress_bar,
            "scaler_type": base.training.scaler_type,
        }

        return TSMixerxConfig(
            architecture=TSMixerxArchitectureConfig(**arch_params),
            training=TSMixerxTrainingConfig(**train_params),
            covariates=TSMixerxCovariatesConfig(
                futr_exog=list(base.covariates.futr_exog),
                hist_exog=list(base.covariates.hist_exog),
            ),
        )

    def _create_forecaster(self, config: Any) -> Any:
        """Create TSMixerxForecaster instance."""
        return TSMixerxForecaster(config)

    def _get_futr_exog_list(self) -> list[str]:
        return list(self._tsmixerx_config.covariates.futr_exog)

    def _get_hist_exog_list(self) -> list[str]:
        return list(self._tsmixerx_config.covariates.hist_exog)
