"""Abstract base class for NeuralForecast-based trainers (TFT, TSMixerx).

Captures ~400 lines of shared logic: TSCV + Optuna HPO + MLflow logging +
OOF cache + final model training.  Subclasses only implement model-specific
configuration and forecaster construction.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from loguru import logger
from optuna import Study, Trial, TrialPruned, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Reduce GPU memory fragmentation (safe no-op if no CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from energy_forecast.config import Settings
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import compute_all
from energy_forecast.training.results import SplitResult
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter
from energy_forecast.training.utils import optuna_storage

# ---------------------------------------------------------------------------
# Shared result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NFTrainingResult:
    """Aggregated result across all CV splits (shared by TFT and TSMixerx)."""

    split_results: list[SplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float


@dataclass(frozen=True)
class NFPipelineResult:
    """Full training pipeline result (shared by TFT and TSMixerx)."""

    study: Study | None
    best_params: dict[str, Any]
    training_result: NFTrainingResult
    final_model: Any  # TFTForecaster or TSMixerxForecaster
    training_time_seconds: float


# ---------------------------------------------------------------------------
# Abstract base trainer
# ---------------------------------------------------------------------------


class NeuralForecastTrainer(ABC):
    """Abstract base for NeuralForecast trainers (TFT, TSMixerx).

    Subclasses implement six abstract methods/properties that supply
    model-specific configuration and forecaster construction.  All
    orchestration (TSCV, Optuna, MLflow, OOF cache) lives here.

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
        self._settings = settings
        self._hp_config = settings.hyperparameters
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(settings.hyperparameters.cross_validation)
        self._target_col = settings.hyperparameters.target_col
        self._skip_validation = settings.hyperparameters.skip_validation_after_optuna
        self._force_hpo = force_hpo

    # -- Abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def _model_name(self) -> str:
        """Return model name: ``'tft'`` or ``'tsmixerx'``."""

    @property
    @abstractmethod
    def _model_config(self) -> Any:
        """Return model-specific config (e.g. ``settings.tft``)."""

    @property
    @abstractmethod
    def _hp_search_config(self) -> Any:
        """Return HP search config (e.g. ``settings.hyperparameters.tft``)."""

    @abstractmethod
    def _build_nf_config(self, params: dict[str, Any]) -> Any:
        """Build model config with Optuna-suggested parameters."""

    @abstractmethod
    def _create_forecaster(self, config: Any) -> Any:
        """Create a model-specific forecaster instance."""

    @abstractmethod
    def _get_futr_exog_list(self) -> list[str]:
        """Return future exogenous covariate names for logging."""

    @abstractmethod
    def _get_hist_exog_list(self) -> list[str]:
        """Return historical exogenous covariate names for logging."""

    # -- Optional hooks (overridable) ----------------------------------------

    def _log_model_artifact(self, model: Any) -> None:  # noqa: B027
        """Log model artifact to MLflow.  Override for model-specific logging."""

    # -- Optuna storage ------------------------------------------------------

    def _optuna_storage(self, model_name: str) -> optuna.storages.RDBStorage | str | None:
        """Return Optuna storage (delegates to shared ``optuna_storage``)."""
        return optuna_storage(
            self._hp_search_config.n_trials,
            model_name,
            self._settings.paths.models_dir,
        )

    # -- GPU cleanup helper --------------------------------------------------

    def _maybe_empty_cuda_cache(self) -> None:
        """Empty CUDA cache when safe (single-job mode)."""
        if torch.cuda.is_available() and self._model_config.optimization.n_jobs <= 1:
            torch.cuda.empty_cache()

    # -- Single split training -----------------------------------------------

    def _train_split(
        self,
        split_info: SplitInfo,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
        max_steps: int | None = None,
        trial: Trial | None = None,
    ) -> SplitResult:
        """Train on a single CV split.

        Args:
            split_info: Split boundary information.
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            params: Hyperparameters.
            max_steps: Override max training steps.
            trial: Optuna trial for step-level pruning callback.

        Returns:
            SplitResult with metrics.

        Raises:
            TrialPruned: When step-level pruning determines the trial is unpromising.
        """
        nf_config = self._build_nf_config(params)

        # Create pruning callback for step-level Optuna integration
        callbacks: list[Any] = []
        if trial is not None:
            from optuna.integration import PyTorchLightningPruningCallback

            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="valid_loss"))

        model = self._create_forecaster(nf_config)
        try:
            model.train(
                train_df,
                val_df,
                target_col=self._target_col,
                max_steps=max_steps,
                callbacks=callbacks or None,
            )

            # Train: last 48h (quick sanity metric)
            train_pred = model.predict(train_df, target_col=self._target_col)

            # Val: rolling prediction — covers full validation period
            train_val_df = pd.concat([train_df, val_df])
            val_pred = model.rolling_predict(
                train_val_df,
                eval_start=split_info.val_start,
                eval_end=split_info.val_end,
                target_col=self._target_col,
            )

            # Test: rolling prediction — covers full test period
            full_df = pd.concat([train_df, val_df, test_df])
            test_pred = model.rolling_predict(
                full_df,
                eval_start=split_info.test_start,
                eval_end=split_info.test_end,
                target_col=self._target_col,
            )
        finally:
            del model
            gc.collect()
            self._maybe_empty_cuda_cache()

        # Align predictions with actuals
        from energy_forecast.models.base import PREDICTION_COL

        y_train = np.asarray(
            train_df[self._target_col].values[-len(train_pred) :], dtype=np.float64
        )
        train_pred_arr = np.asarray(train_pred[PREDICTION_COL].values, dtype=np.float64)

        val_common_idx = val_pred.index.intersection(val_df.index)
        y_val = np.asarray(val_df.loc[val_common_idx, self._target_col].values, dtype=np.float64)
        val_pred_arr = np.asarray(
            val_pred.loc[val_common_idx, PREDICTION_COL].values, dtype=np.float64
        )

        test_common_idx = test_pred.index.intersection(test_df.index)
        y_test = np.asarray(
            test_df.loc[test_common_idx, self._target_col].values, dtype=np.float64
        )
        test_pred_arr = np.asarray(
            test_pred.loc[test_common_idx, PREDICTION_COL].values, dtype=np.float64
        )

        # Inverse Box-Cox: metrics always in MWh space
        if self._settings.boxcox.enabled:
            from energy_forecast.transform import inv_boxcox

            lam = self._settings.boxcox.lambda_param
            y_train = inv_boxcox(y_train, lam)
            y_val = inv_boxcox(y_val, lam)
            y_test = inv_boxcox(y_test, lam)
            train_pred_arr = inv_boxcox(train_pred_arr, lam)
            val_pred_arr = inv_boxcox(val_pred_arr, lam)
            test_pred_arr = inv_boxcox(test_pred_arr, lam)

        return SplitResult(
            split_idx=split_info.split_idx,
            train_metrics=compute_all(y_train, train_pred_arr),
            val_metrics=compute_all(y_val, val_pred_arr),
            test_metrics=compute_all(y_test, test_pred_arr),
            val_month=split_info.val_start.strftime("%Y-%m"),
            test_month=split_info.test_start.strftime("%Y-%m"),
            val_predictions=val_pred_arr,
            val_actuals=y_val,
            test_predictions=test_pred_arr,
            test_actuals=y_test,
        )

    # -- All splits training -------------------------------------------------

    def _train_all_splits(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        max_steps: int | None = None,
    ) -> NFTrainingResult:
        """Train on all TSCV splits and aggregate results.

        Args:
            df: Full feature-engineered DataFrame.
            params: Hyperparameters.
            max_steps: Override max training steps.

        Returns:
            NFTrainingResult with aggregated metrics.
        """
        results: list[SplitResult] = []

        for info, train_df, val_df, test_df in self._splitter.iter_splits(df):
            result = self._train_split(info, train_df, val_df, test_df, params, max_steps)
            results.append(result)
            logger.info(
                "Split {} | val={} MAPE={:.2f}% | test={} MAPE={:.2f}%",
                result.split_idx,
                result.val_month,
                result.val_metrics.mape,
                result.test_month,
                result.test_metrics.mape,
            )

        val_mapes = [r.val_metrics.mape for r in results]
        test_mapes = [r.test_metrics.mape for r in results]

        return NFTrainingResult(
            split_results=results,
            avg_val_mape=float(np.mean(val_mapes)),
            avg_test_mape=float(np.mean(test_mapes)),
            std_val_mape=float(np.std(val_mapes)),
        )

    # -- Optuna objective (dynamic from YAML) --------------------------------

    def _create_objective(
        self,
        df: pd.DataFrame,
    ) -> tuple[Callable[[Trial], float], dict[int, list[SplitResult]]]:
        """Create Optuna objective using dynamic YAML search space.

        Uses ``optuna_splits`` CV splits with step-level pruning via
        ``PyTorchLightningPruningCallback``.

        Returns:
            Tuple of (objective function, trial split results cache).
        """
        n_optuna_splits: int = self._model_config.optimization.optuna_splits
        search_space = self._hp_search_config.search_space

        all_splits = list(self._splitter.iter_splits(df))
        if not all_splits:
            msg = "No CV splits available"
            raise ValueError(msg)

        if n_optuna_splits >= len(all_splits):
            selected_splits = all_splits
        else:
            indices = np.linspace(0, len(all_splits) - 1, n_optuna_splits, dtype=int)
            selected_splits = [all_splits[i] for i in indices]

        name = self._model_name.upper()
        logger.info(
            "{} Optuna: using {}/{} CV splits, step-level pruning active",
            name,
            len(selected_splits),
            len(all_splits),
        )

        trial_results: dict[int, list[SplitResult]] = {}
        cache_lock = threading.Lock()

        def objective(trial: Trial) -> float:
            suggested = suggest_params(trial, search_space)

            # Loss is a config param, not a model constructor param.
            # Pop it from suggested and update config via model_copy.
            loss_name = suggested.pop("loss", None)
            if loss_name is not None:
                original_config = self._model_config
                updated_config = original_config.model_copy(update={"loss": loss_name})
                # Swap instance config for this trial
                config_attr = f"_{self._model_name}_config"
                object.__setattr__(self, config_attr, updated_config)

            val_mapes: list[float] = []
            test_mapes: list[float] = []
            split_results: list[SplitResult] = []

            for _fold_idx, (info, train_df, val_df, test_df) in enumerate(selected_splits):
                try:
                    result = self._train_split(
                        info,
                        train_df,
                        val_df,
                        test_df,
                        suggested,
                        trial=trial,
                    )
                    val_mapes.append(result.val_metrics.mape)
                    test_mapes.append(result.test_metrics.mape)
                    split_results.append(result)
                except TrialPruned:
                    raise
                except Exception as e:
                    logger.warning("Trial split {} failed: {}", info.split_idx, e)
                    return float("inf")

            avg_mape = float(np.mean(val_mapes))
            trial.set_user_attr("val_mapes", val_mapes)
            trial.set_user_attr("avg_test_mape", float(np.mean(test_mapes)))
            with cache_lock:
                trial_results[trial.number] = split_results
            return avg_mape

        return objective, trial_results

    # -- Optimize ------------------------------------------------------------

    def optimize(
        self,
        df: pd.DataFrame,
    ) -> tuple[Study, NFTrainingResult]:
        """Run Optuna hyperparameter optimization.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Tuple of (study, best_trial_result trained on all splits).
        """
        name = self._model_name
        storage = self._optuna_storage(name)
        study = create_study(
            study_name=name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(
                multivariate=True,
                n_startup_trials=5,
                seed=self._model_config.training.random_seed,
            ),
            pruner=MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=3,
            ),
        )

        objective, trial_results = self._create_objective(df)

        n_jobs: int = self._model_config.optimization.n_jobs
        logger.info(
            "{} Optuna: {} trials, {} parallel job(s)",
            name.upper(),
            self._hp_search_config.n_trials,
            n_jobs,
        )
        study.optimize(objective, n_trials=self._hp_search_config.n_trials, n_jobs=n_jobs)

        logger.info(
            "Optimization done — best val MAPE: {:.2f}%, params: {}",
            study.best_value,
            study.best_params,
        )

        best_trial_num = study.best_trial.number

        if best_trial_num in trial_results:
            cached_splits = trial_results[best_trial_num]
            best_result = NFTrainingResult(
                split_results=cached_splits,
                avg_val_mape=study.best_value,
                avg_test_mape=float(
                    study.best_trial.user_attrs.get("avg_test_mape", float("nan"))
                ),
                std_val_mape=float(np.std([sr.val_metrics.mape for sr in cached_splits])),
            )
            logger.info("Using cached predictions from trial {}", best_trial_num)
        elif self._skip_validation:
            logger.info("Skipping post-Optuna validation (skip_validation_after_optuna=true)")
            best_result = NFTrainingResult(
                split_results=[],
                avg_val_mape=study.best_value,
                avg_test_mape=float(
                    study.best_trial.user_attrs.get("avg_test_mape", float("nan"))
                ),
                std_val_mape=0.0,
            )
        else:
            logger.info("Cache miss for best trial — retraining on all splits")
            best_result = self._train_all_splits(df, study.best_params)

        return study, best_result

    # -- Final model ---------------------------------------------------------

    def train_final(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
    ) -> Any:
        """Train final model on all data with best params.

        Uses last portion of data as validation for early stopping.

        Args:
            df: Full dataset.
            params: Best hyperparameters from optimization.

        Returns:
            Trained forecaster.
        """
        val_size: int = self._model_config.optimization.val_size_hours
        if len(df) > val_size * 2:
            train_df = df.iloc[:-val_size]
            val_df: pd.DataFrame | None = df.iloc[-val_size:]
        else:
            train_df = df
            val_df = None

        nf_config = self._build_nf_config(params)
        model = self._create_forecaster(nf_config)
        model.train(train_df, val_df, target_col=self._target_col)

        logger.info("Final {} model trained on {} samples", self._model_name.upper(), len(df))
        return model

    # -- MLflow logging helpers ----------------------------------------------

    def _log_cv_metrics(self, best_result: NFTrainingResult) -> None:
        """Log CV metrics and split details to MLflow."""
        test_mapes = [sr.test_metrics.mape for sr in best_result.split_results]
        std_test_mape = float(np.std(test_mapes)) if test_mapes else 0.0

        self._tracker.log_metrics(
            {
                "avg_val_mape": best_result.avg_val_mape,
                "avg_test_mape": best_result.avg_test_mape,
                "std_val_mape": best_result.std_val_mape,
                "std_test_mape": std_test_mape,
            }
        )
        for sr in best_result.split_results:
            self._tracker.log_split_metrics(
                sr.split_idx, sr.train_metrics, sr.val_metrics, sr.test_metrics
            )

    def _log_common_meta(self, df: pd.DataFrame, extra: dict[str, Any] | None = None) -> None:
        """Log training metadata, config snapshot, and covariate lists."""
        meta: dict[str, Any] = {
            "data_rows": len(df),
            "data_cols": len(df.columns),
            "n_splits": self._hp_config.cross_validation.n_splits,
            "python_version": sys.version,
            "platform": sys.platform,
        }
        if extra:
            meta.update(extra)
        self._tracker.log_training_meta(meta)

        self._tracker.log_config_snapshot(
            self._model_config.model_dump(),
            f"{self._model_name}_config.yaml",
        )
        self._tracker.log_params(
            {
                "futr_exog_list": ",".join(self._get_futr_exog_list()),
                "hist_exog_list": ",".join(self._get_hist_exog_list()),
            }
        )

    def _save_oof_cache(self, best_result: NFTrainingResult) -> None:
        """Save OOF cache for ensemble (non-fatal)."""
        from energy_forecast.training.oof_cache import compute_config_hash, save_oof_cache

        try:
            config_hash = compute_config_hash(self._settings, self._model_name)
            save_oof_cache(
                self._model_name,
                best_result.split_results,
                self._settings.paths.models_dir,
                config_hash,
            )
        except Exception as e:
            logger.warning("Failed to save OOF cache (non-fatal): {}", e)

    # -- Full pipeline -------------------------------------------------------

    def run(self, df: pd.DataFrame) -> NFPipelineResult:
        """Execute training pipeline — auto-selects fixed or Optuna mode.

        If ``best_params`` is populated in the model YAML and ``--force-hpo``
        is not set, skips Optuna and uses stored params directly.
        """
        name = self._model_name
        has_best = bool(self._model_config.best_params)
        if has_best and not self._force_hpo:
            logger.info("best_params found in {}.yaml — using fixed mode", name)
            return self._run_fixed(df)
        if has_best and self._force_hpo:
            logger.info("--force-hpo: ignoring best_params, running Optuna")
        return self._run_optuna(df)

    def _run_fixed(self, df: pd.DataFrame) -> NFPipelineResult:
        """Train with stored best_params — skip Optuna entirely."""
        start = time.monotonic()
        name = self._model_name
        best_params = dict(self._model_config.best_params)
        logger.info("Fixed params: {}", best_params)

        with self._tracker.start_run(f"{name}_fixed"):
            self._tracker.log_params(best_params)
            best_result = self._train_all_splits(df, best_params)
            self._log_cv_metrics(best_result)
            self._log_common_meta(df, extra={"mode": "fixed"})

        with self._tracker.start_run(f"{name}_final"):
            final_model = self.train_final(df, best_params)

            model_dir = Path(self._settings.paths.models_dir) / name
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model.save(model_dir)
            logger.info("Model saved to {}", model_dir)

            self._log_model_artifact(final_model)

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {"training_time_seconds": elapsed},
            )

        logger.info("{} fixed-params pipeline complete in {:.1f}s", name.upper(), elapsed)

        self._save_oof_cache(best_result)

        return NFPipelineResult(
            study=None,
            best_params=best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )

    def _run_optuna(self, df: pd.DataFrame) -> NFPipelineResult:
        """Execute full Optuna HPO pipeline."""
        start = time.monotonic()
        name = self._model_name

        with self._tracker.start_run(f"{name}_optimization"):
            study, best_result = self.optimize(df)
            self._tracker.log_params(study.best_params)
            self._log_cv_metrics(best_result)
            self._log_common_meta(
                df,
                extra={
                    "n_trials": self._hp_search_config.n_trials,
                    "best_trial_number": study.best_trial.number,
                },
            )

        with self._tracker.start_run(f"{name}_final"):
            final_model = self.train_final(df, study.best_params)

            model_dir = Path(self._settings.paths.models_dir) / name
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model.save(model_dir)
            logger.info("Model saved to {}", model_dir)

            self._log_model_artifact(final_model)

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {"training_time_seconds": elapsed},
            )

        logger.info("{} pipeline complete in {:.1f}s", name.upper(), elapsed)

        self._save_oof_cache(best_result)

        return NFPipelineResult(
            study=study,
            best_params=study.best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )
