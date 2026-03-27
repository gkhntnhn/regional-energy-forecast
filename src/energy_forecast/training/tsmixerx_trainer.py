"""TSMixerx training pipeline: TSCV + Optuna + MLflow.

Orchestrates hyperparameter optimization via Optuna, cross-validated
training on calendar-month splits, and final model training on all data.

Follows the same pattern as TFTTrainer using shared M5 infrastructure.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
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
from energy_forecast.models.tsmixerx import TSMixerxForecaster
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import compute_all
from energy_forecast.training.results import SplitResult as TSMixerxSplitResult
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter
from energy_forecast.training.utils import optuna_storage


@dataclass(frozen=True)
class TSMixerxTrainingResult:
    """Aggregated result across all CV splits."""

    split_results: list[TSMixerxSplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float


@dataclass(frozen=True)
class TSMixerxPipelineResult:
    """Full training pipeline result."""

    study: Study | None
    best_params: dict[str, Any]
    training_result: TSMixerxTrainingResult
    final_model: TSMixerxForecaster
    training_time_seconds: float


# ---------------------------------------------------------------------------
# TSMixerxTrainer
# ---------------------------------------------------------------------------


class TSMixerxTrainer:
    """TSMixerx training pipeline with TSCV, Optuna, and MLflow.

    Args:
        settings: Full application settings.
        tracker: MLflow experiment tracker (disabled by default).
    """

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
        *,
        force_hpo: bool = False,
    ) -> None:
        self._settings = settings
        self._tsmixerx_config = settings.tsmixerx
        self._hp_config = settings.hyperparameters
        self._search_config = settings.hyperparameters.tsmixerx
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(settings.hyperparameters.cross_validation)
        self._target_col = settings.hyperparameters.target_col
        self._skip_validation = settings.hyperparameters.skip_validation_after_optuna
        self._force_hpo = force_hpo

    # -- Optuna storage --

    def _optuna_storage(self, model_name: str) -> optuna.storages.RDBStorage | str | None:
        """Return Optuna storage (delegates to shared ``optuna_storage``)."""
        return optuna_storage(
            self._search_config.n_trials,
            model_name,
            self._settings.paths.models_dir,
        )

    # -- Build TSMixerx config with overrides --

    def _build_tsmixerx_config(self, params: dict[str, Any]) -> Any:
        """Build TSMixerx config with Optuna-suggested parameters.

        Args:
            params: Suggested hyperparameters from Optuna.

        Returns:
            Updated TSMixerxConfig.
        """
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

    # -- Single split training --

    def _train_split(
        self,
        split_info: SplitInfo,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
        max_steps: int | None = None,
        trial: Trial | None = None,
    ) -> TSMixerxSplitResult:
        """Train TSMixerx on a single CV split.

        Args:
            split_info: Split boundary information.
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            params: Hyperparameters.
            max_steps: Override max training steps.
            trial: Optuna trial for step-level pruning callback.

        Returns:
            TSMixerxSplitResult with metrics.
        """
        config = self._build_tsmixerx_config(params)

        callbacks: list[Any] = []
        if trial is not None:
            from optuna.integration import PyTorchLightningPruningCallback

            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="valid_loss"))

        model = TSMixerxForecaster(config)
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

            # Val: rolling prediction
            train_val_df = pd.concat([train_df, val_df])
            val_pred = model.rolling_predict(
                train_val_df,
                eval_start=split_info.val_start,
                eval_end=split_info.val_end,
                target_col=self._target_col,
            )

            # Test: rolling prediction
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
            if torch.cuda.is_available() and self._tsmixerx_config.optimization.n_jobs <= 1:
                torch.cuda.empty_cache()

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
        y_test = np.asarray(test_df.loc[test_common_idx, self._target_col].values, dtype=np.float64)
        test_pred_arr = np.asarray(
            test_pred.loc[test_common_idx, PREDICTION_COL].values, dtype=np.float64
        )

        return TSMixerxSplitResult(
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

    # -- All splits training --

    def _train_all_splits(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        max_steps: int | None = None,
    ) -> TSMixerxTrainingResult:
        """Train on all TSCV splits and aggregate results.

        Args:
            df: Full feature-engineered DataFrame.
            params: Hyperparameters.
            max_steps: Override max training steps.

        Returns:
            TSMixerxTrainingResult with aggregated metrics.
        """
        results: list[TSMixerxSplitResult] = []

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

        return TSMixerxTrainingResult(
            split_results=results,
            avg_val_mape=float(np.mean(val_mapes)),
            avg_test_mape=float(np.mean(test_mapes)),
            std_val_mape=float(np.std(val_mapes)),
        )

    # -- Optuna objective --

    def _create_objective(
        self,
        df: pd.DataFrame,
    ) -> tuple[Callable[[Trial], float], dict[int, list[TSMixerxSplitResult]]]:
        """Create Optuna objective using dynamic YAML search space.

        Returns:
            Tuple of (objective function, trial split results cache).
        """
        n_optuna_splits = self._tsmixerx_config.optimization.optuna_splits
        search_space = self._search_config.search_space

        all_splits = list(self._splitter.iter_splits(df))
        if not all_splits:
            msg = "No CV splits available"
            raise ValueError(msg)

        if n_optuna_splits >= len(all_splits):
            selected_splits = all_splits
        else:
            indices = np.linspace(0, len(all_splits) - 1, n_optuna_splits, dtype=int)
            selected_splits = [all_splits[i] for i in indices]

        logger.info(
            "TSMixerx Optuna: using {}/{} CV splits, step-level pruning active",
            len(selected_splits),
            len(all_splits),
        )

        trial_results: dict[int, list[TSMixerxSplitResult]] = {}
        cache_lock = threading.Lock()

        def objective(trial: Trial) -> float:
            suggested = suggest_params(trial, search_space)
            val_mapes: list[float] = []
            test_mapes: list[float] = []
            split_results: list[TSMixerxSplitResult] = []

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

    # -- Optimize --

    def optimize(
        self,
        df: pd.DataFrame,
    ) -> tuple[Study, TSMixerxTrainingResult]:
        """Run Optuna hyperparameter optimization.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Tuple of (study, best_trial_result).
        """
        storage = self._optuna_storage("tsmixerx")
        study = create_study(
            study_name="tsmixerx",
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=self._tsmixerx_config.training.random_seed),
            pruner=MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=3,
            ),
        )

        objective, trial_results = self._create_objective(df)

        n_jobs = self._tsmixerx_config.optimization.n_jobs
        logger.info(
            "TSMixerx Optuna: {} trials, {} parallel job(s)",
            self._search_config.n_trials,
            n_jobs,
        )
        study.optimize(objective, n_trials=self._search_config.n_trials, n_jobs=n_jobs)

        logger.info(
            "Optimization done — best val MAPE: {:.2f}%, params: {}",
            study.best_value,
            study.best_params,
        )

        best_trial_num = study.best_trial.number

        if best_trial_num in trial_results:
            cached_splits = trial_results[best_trial_num]
            best_result = TSMixerxTrainingResult(
                split_results=cached_splits,
                avg_val_mape=study.best_value,
                avg_test_mape=float(study.best_trial.user_attrs.get("avg_test_mape", float("nan"))),
                std_val_mape=float(np.std([sr.val_metrics.mape for sr in cached_splits])),
            )
            logger.info("Using cached predictions from trial {}", best_trial_num)
        elif self._skip_validation:
            logger.info("Skipping post-Optuna validation (skip_validation_after_optuna=true)")
            best_result = TSMixerxTrainingResult(
                split_results=[],
                avg_val_mape=study.best_value,
                avg_test_mape=float(study.best_trial.user_attrs.get("avg_test_mape", float("nan"))),
                std_val_mape=0.0,
            )
        else:
            logger.info("Cache miss for best trial — retraining on all splits")
            best_result = self._train_all_splits(df, study.best_params)

        return study, best_result

    # -- Final model --

    def train_final(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
    ) -> TSMixerxForecaster:
        """Train final model on all data with best params.

        Args:
            df: Full dataset.
            params: Best hyperparameters from optimization.

        Returns:
            Trained TSMixerxForecaster.
        """
        val_size = self._tsmixerx_config.optimization.val_size_hours
        if len(df) > val_size * 2:
            train_df = df.iloc[:-val_size]
            val_df = df.iloc[-val_size:]
        else:
            train_df = df
            val_df = None

        config = self._build_tsmixerx_config(params)
        model = TSMixerxForecaster(config)
        model.train(train_df, val_df, target_col=self._target_col)

        logger.info("Final TSMixerx model trained on {} samples", len(df))
        return model

    # -- Full pipeline --

    def run(self, df: pd.DataFrame) -> TSMixerxPipelineResult:
        """Execute training pipeline — auto-selects fixed or Optuna mode.

        If ``best_params`` is populated in tsmixerx.yaml and ``--force-hpo``
        is not set, skips Optuna and uses stored params directly.
        """
        has_best = bool(self._tsmixerx_config.best_params)
        if has_best and not self._force_hpo:
            logger.info("best_params found in tsmixerx.yaml — using fixed mode")
            return self._run_fixed(df)
        if has_best and self._force_hpo:
            logger.info("--force-hpo: ignoring best_params, running Optuna")
        return self._run_optuna(df)

    def _run_fixed(self, df: pd.DataFrame) -> TSMixerxPipelineResult:
        """Train with stored best_params — skip Optuna entirely."""
        start = time.monotonic()
        best_params = dict(self._tsmixerx_config.best_params)
        logger.info("Fixed params: {}", best_params)

        with self._tracker.start_run("tsmixerx_fixed"):
            self._tracker.log_params(best_params)

            # Train all splits (_train_split calls _build_tsmixerx_config internally)
            best_result = self._train_all_splits(df, best_params)

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
            self._tracker.log_params({"mode": "fixed"})

        with self._tracker.start_run("tsmixerx_final"):
            final_model = self.train_final(df, best_params)

            model_dir = Path(self._settings.paths.models_dir) / "tsmixerx"
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model.save(model_dir)
            logger.info("Model saved to {}", model_dir)

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {"training_time_seconds": elapsed},
            )

        logger.info("TSMixerx fixed-params pipeline complete in {:.1f}s", elapsed)

        from energy_forecast.training.oof_cache import (
            compute_config_hash,
            save_oof_cache,
        )

        try:
            config_hash = compute_config_hash(self._settings, "tsmixerx")
            save_oof_cache(
                "tsmixerx",
                best_result.split_results,
                self._settings.paths.models_dir,
                config_hash,
            )
        except Exception as e:
            logger.warning("Failed to save OOF cache (non-fatal): {}", e)

        return TSMixerxPipelineResult(
            study=None,
            best_params=best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )

    def _run_optuna(self, df: pd.DataFrame) -> TSMixerxPipelineResult:
        """Execute full Optuna HPO pipeline (original run() logic)."""
        start = time.monotonic()

        with self._tracker.start_run("tsmixerx_optimization"):
            study, best_result = self.optimize(df)
            self._tracker.log_params(study.best_params)

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
                    sr.split_idx,
                    sr.train_metrics,
                    sr.val_metrics,
                    sr.test_metrics,
                )

            self._tracker.log_training_meta(
                {
                    "data_rows": len(df),
                    "data_cols": len(df.columns),
                    "n_splits": self._hp_config.cross_validation.n_splits,
                    "n_trials": self._search_config.n_trials,
                    "best_trial_number": study.best_trial.number,
                    "python_version": sys.version,
                    "platform": sys.platform,
                }
            )
            self._tracker.log_config_snapshot(
                self._tsmixerx_config.model_dump(),
                "tsmixerx_config.yaml",
            )
            self._tracker.log_params(
                {
                    "futr_exog_list": ",".join(self._tsmixerx_config.covariates.futr_exog),
                    "hist_exog_list": ",".join(self._tsmixerx_config.covariates.hist_exog),
                }
            )

        with self._tracker.start_run("tsmixerx_final"):
            final_model = self.train_final(df, study.best_params)

            model_dir = Path(self._settings.paths.models_dir) / "tsmixerx"
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model.save(model_dir)
            logger.info("Model saved to {}", model_dir)

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {"training_time_seconds": elapsed},
            )

        logger.info("TSMixerx pipeline complete in {:.1f}s", elapsed)

        # Save OOF cache for ensemble
        from energy_forecast.training.oof_cache import (
            compute_config_hash,
            save_oof_cache,
        )

        try:
            config_hash = compute_config_hash(self._settings, "tsmixerx")
            save_oof_cache(
                "tsmixerx",
                best_result.split_results,
                self._settings.paths.models_dir,
                config_hash,
            )
        except Exception as e:
            logger.warning("Failed to save OOF cache (non-fatal): {}", e)

        return TSMixerxPipelineResult(
            study=study,
            best_params=study.best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )
