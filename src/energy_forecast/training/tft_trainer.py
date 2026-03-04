"""TFT training pipeline: TSCV + Optuna + MLflow.

Orchestrates hyperparameter optimization via Optuna, cross-validated
training on calendar-month splits, and final model training on all data.

Uses the same shared infrastructure as CatBoostTrainer and ProphetTrainer:
- TimeSeriesSplitter for calendar-month TSCV
- suggest_params for dynamic Optuna search space
- compute_all / MetricsResult for metrics
- ExperimentTracker for MLflow logging
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from optuna import Study, Trial, TrialPruned, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from energy_forecast.config.settings import Settings
from energy_forecast.models.tft import TFTForecaster
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult, compute_all
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TFTSplitResult:
    """Result from a single CV split."""

    split_idx: int
    train_metrics: MetricsResult
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    val_month: str
    test_month: str
    val_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    val_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None


@dataclass(frozen=True)
class TFTTrainingResult:
    """Aggregated result across all CV splits."""

    split_results: list[TFTSplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float


@dataclass(frozen=True)
class TFTPipelineResult:
    """Full training pipeline result."""

    study: Study
    best_params: dict[str, Any]
    training_result: TFTTrainingResult
    final_model: TFTForecaster
    training_time_seconds: float


# ---------------------------------------------------------------------------
# TFTTrainer
# ---------------------------------------------------------------------------


class TFTTrainer:
    """TFT training pipeline with TSCV, Optuna, and MLflow.

    Follows the same pattern as CatBoostTrainer and ProphetTrainer,
    using shared M5 infrastructure.

    Args:
        settings: Full application settings.
        tracker: MLflow experiment tracker (disabled by default).
    """

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self._settings = settings
        self._tft_config = settings.tft
        self._hp_config = settings.hyperparameters
        self._search_config = settings.hyperparameters.tft
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(settings.hyperparameters.cross_validation)
        self._target_col = settings.hyperparameters.target_col
        self._skip_validation = settings.hyperparameters.skip_validation_after_optuna

    # -- Optuna storage --

    def _optuna_storage(self, model_name: str) -> str | None:
        """Return SQLite storage URL for Optuna study persistence."""
        if self._search_config.n_trials <= 3:
            return None
        studies_dir = Path(self._settings.paths.models_dir) / "optuna_studies"
        studies_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{studies_dir / model_name}.db"

    # -- Build TFT config with overrides --

    def _build_tft_config(self, params: dict[str, Any]) -> Any:
        """Build TFT config with Optuna-suggested parameters.

        Args:
            params: Suggested hyperparameters from Optuna.

        Returns:
            Updated TFTConfig.
        """
        from energy_forecast.config.settings import (
            TFTArchitectureConfig,
            TFTConfig,
            TFTCovariatesConfig,
            TFTTrainingConfig,
        )

        base = self._tft_config

        # Override architecture params
        arch_params = {
            "hidden_size": params.get("hidden_size", base.architecture.hidden_size),
            "attention_head_size": params.get(
                "attention_head_size", base.architecture.attention_head_size
            ),
            "lstm_layers": params.get("lstm_layers", base.architecture.lstm_layers),
            "dropout": params.get("dropout", base.architecture.dropout),
            "hidden_continuous_size": params.get(
                "hidden_continuous_size", base.architecture.hidden_continuous_size
            ),
        }

        # Override training params (carry ALL fields from base config)
        train_params = {
            "encoder_length": base.training.encoder_length,
            "prediction_length": base.training.prediction_length,
            "batch_size": params.get("batch_size", base.training.batch_size),
            "max_epochs": base.training.max_epochs,
            "learning_rate": params.get("learning_rate", base.training.learning_rate),
            "early_stop_patience": base.training.early_stop_patience,
            "gradient_clip_val": base.training.gradient_clip_val,
            "random_seed": base.training.random_seed,
            "accelerator": base.training.accelerator,
            "num_workers": base.training.num_workers,
            "enable_progress_bar": base.training.enable_progress_bar,
            "enable_model_summary": base.training.enable_model_summary,
            "precision": base.training.precision,
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

    # -- Single split training --

    def _train_split(
        self,
        split_info: SplitInfo,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
        max_epochs: int | None = None,
        trial: Trial | None = None,
    ) -> TFTSplitResult:
        """Train TFT on a single CV split.

        Args:
            split_info: Split boundary information.
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            params: Hyperparameters.
            max_epochs: Override max epochs (for faster optimization).
            trial: Optuna trial for epoch-level pruning callback.

        Returns:
            TFTSplitResult with metrics.

        Raises:
            TrialPruned: When epoch-level pruning determines the trial is unpromising.
        """
        # Build config with suggested params
        tft_config = self._build_tft_config(params)

        # Create pruning callback for epoch-level Optuna integration
        callbacks: list[Any] = []
        if trial is not None:
            from optuna.integration import PyTorchLightningPruningCallback

            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor="val_loss")
            )

        # Create and train model (try/finally ensures GPU memory cleanup on pruning)
        model = TFTForecaster(tft_config)
        try:
            model.train(
                train_df,
                val_df,
                target_col=self._target_col,
                max_epochs=max_epochs,
                callbacks=callbacks or None,
            )

            # Predictions
            train_pred = model.predict(train_df, target_col=self._target_col)
            val_pred = model.predict(val_df, target_col=self._target_col)
            test_pred = model.predict(test_df, target_col=self._target_col)
        finally:
            # Free GPU/CPU memory from this fold's model
            del model
            gc.collect()
            # empty_cache evicts ALL GPU cache — unsafe when parallel threads share
            # the same CUDA context (n_jobs > 1).  Let PyTorch manage memory instead.
            if (
                torch.cuda.is_available()
                and self._tft_config.optimization.n_jobs <= 1
            ):
                torch.cuda.empty_cache()

        # Align predictions with actuals
        y_train = np.asarray(train_df[self._target_col].values[-len(train_pred):], dtype=np.float64)
        y_val = np.asarray(val_df[self._target_col].values[-len(val_pred):], dtype=np.float64)
        y_test = np.asarray(test_df[self._target_col].values[-len(test_pred):], dtype=np.float64)

        from energy_forecast.models.base import PREDICTION_COL

        train_pred_arr = np.asarray(train_pred[PREDICTION_COL].values, dtype=np.float64)
        val_pred_arr = np.asarray(val_pred[PREDICTION_COL].values, dtype=np.float64)
        test_pred_arr = np.asarray(test_pred[PREDICTION_COL].values, dtype=np.float64)

        return TFTSplitResult(
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
        max_epochs: int | None = None,
    ) -> TFTTrainingResult:
        """Train on all TSCV splits and aggregate results.

        Args:
            df: Full feature-engineered DataFrame.
            params: Hyperparameters.
            max_epochs: Override max epochs (for faster optimization).

        Returns:
            TFTTrainingResult with aggregated metrics.
        """
        results: list[TFTSplitResult] = []

        for info, train_df, val_df, test_df in self._splitter.iter_splits(df):
            result = self._train_split(
                info, train_df, val_df, test_df, params, max_epochs
            )
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

        return TFTTrainingResult(
            split_results=results,
            avg_val_mape=float(np.mean(val_mapes)),
            avg_test_mape=float(np.mean(test_mapes)),
            std_val_mape=float(np.std(val_mapes)),
        )

    # -- Optuna objective (dynamic from YAML) --

    def _create_objective(
        self,
        df: pd.DataFrame,
    ) -> tuple[Callable[[Trial], float], dict[int, list[TFTSplitResult]]]:
        """Create Optuna objective using dynamic YAML search space.

        Uses ``optuna_splits`` CV splits with epoch-level pruning via
        ``PyTorchLightningPruningCallback``.  All trials train at full
        ``max_epochs``; bad trials are pruned early by the MedianPruner
        based on ``val_loss`` reported each epoch.

        Returns:
            Tuple of (objective function, trial split results cache).
        """
        n_optuna_splits = self._tft_config.optimization.optuna_splits
        search_space = self._search_config.search_space

        all_splits = list(self._splitter.iter_splits(df))
        if not all_splits:
            msg = "No CV splits available"
            raise ValueError(msg)

        # Use up to n_optuna_splits, evenly spaced across available splits
        if n_optuna_splits >= len(all_splits):
            selected_splits = all_splits
        else:
            indices = np.linspace(0, len(all_splits) - 1, n_optuna_splits, dtype=int)
            selected_splits = [all_splits[i] for i in indices]

        logger.info(
            "TFT Optuna: using {}/{} CV splits, epoch-level pruning active",
            len(selected_splits),
            len(all_splits),
        )

        trial_results: dict[int, list[TFTSplitResult]] = {}
        cache_lock = threading.Lock()

        def objective(trial: Trial) -> float:
            suggested = suggest_params(trial, search_space)
            val_mapes: list[float] = []
            test_mapes: list[float] = []
            split_results: list[TFTSplitResult] = []

            for _fold_idx, (info, train_df, val_df, test_df) in enumerate(
                selected_splits
            ):
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
    ) -> tuple[Study, TFTTrainingResult]:
        """Run Optuna hyperparameter optimization.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Tuple of (study, best_trial_result trained on all splits).
        """
        storage = self._optuna_storage("tft")
        study = create_study(
            study_name="tft",
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=self._tft_config.training.random_seed),
            pruner=MedianPruner(
                n_startup_trials=2,  # First 2 trials run uninterrupted (reference)
                n_warmup_steps=3,    # First 3 epochs per trial safe (model stabilization)
            ),
        )

        objective, trial_results = self._create_objective(df)

        n_jobs = self._tft_config.optimization.n_jobs
        logger.info(
            "TFT Optuna: {} trials, {} parallel job(s)",
            self._search_config.n_trials,
            n_jobs,
        )
        study.optimize(
            objective, n_trials=self._search_config.n_trials, n_jobs=n_jobs
        )

        logger.info(
            "Optimization done — best val MAPE: {:.2f}%, params: {}",
            study.best_value,
            study.best_params,
        )

        # Epoch-level pruning means all trials run at max_epochs, so the best
        # trial's cached results are production-quality — no retrain needed.
        best_trial_num = study.best_trial.number

        if best_trial_num in trial_results:
            cached_splits = trial_results[best_trial_num]
            best_result = TFTTrainingResult(
                split_results=cached_splits,
                avg_val_mape=study.best_value,
                avg_test_mape=float(
                    study.best_trial.user_attrs.get("avg_test_mape", float("nan"))
                ),
                std_val_mape=float(
                    np.std([sr.val_metrics.mape for sr in cached_splits])
                ),
            )
            logger.info("Using cached predictions from trial {}", best_trial_num)
        elif self._skip_validation:
            logger.info("Skipping post-Optuna validation (skip_validation_after_optuna=true)")
            best_result = TFTTrainingResult(
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

    # -- Final model --

    def train_final(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
    ) -> TFTForecaster:
        """Train final model on all data with best params.

        Uses last portion of data as validation for early stopping.

        Args:
            df: Full dataset.
            params: Best hyperparameters from optimization.

        Returns:
            Trained TFTForecaster.
        """
        # Use configured validation size
        val_size = self._tft_config.optimization.val_size_hours
        if len(df) > val_size * 2:
            train_df = df.iloc[:-val_size]
            val_df = df.iloc[-val_size:]
        else:
            train_df = df
            val_df = None

        tft_config = self._build_tft_config(params)
        model = TFTForecaster(tft_config)
        model.train(train_df, val_df, target_col=self._target_col)

        logger.info("Final TFT model trained on {} samples", len(df))
        return model

    # -- Full pipeline --

    def run(
        self,
        df: pd.DataFrame,
    ) -> TFTPipelineResult:
        """Execute full training pipeline: optimize + final model + MLflow.

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            TFTPipelineResult with study, final model, and metrics.
        """
        start = time.monotonic()

        with self._tracker.start_run("tft_optimization"):
            study, best_result = self.optimize(df)
            self._tracker.log_params(study.best_params)
            self._tracker.log_metrics(
                {
                    "avg_val_mape": best_result.avg_val_mape,
                    "avg_test_mape": best_result.avg_test_mape,
                    "std_val_mape": best_result.std_val_mape,
                }
            )
            for sr in best_result.split_results:
                self._tracker.log_split_metrics(
                    sr.split_idx, sr.train_metrics, sr.val_metrics, sr.test_metrics
                )

        with self._tracker.start_run("tft_final"):
            final_model = self.train_final(df, study.best_params)

            # Save model to timestamped subdirectory (consistent with CatBoost/Prophet)
            from datetime import datetime

            from energy_forecast.utils import TZ_ISTANBUL

            run_ts = datetime.now(tz=TZ_ISTANBUL).strftime("%Y-%m-%d_%H-%M")
            model_dir = Path(self._settings.paths.models_dir) / "tft" / f"tft_{run_ts}"
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model.save(model_dir)
            logger.info("Model saved to {}", model_dir)

            self._tracker.log_tft_model(final_model, "tft_model")

        elapsed = time.monotonic() - start
        logger.info("TFT pipeline complete in {:.1f}s", elapsed)

        return TFTPipelineResult(
            study=study,
            best_params=study.best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )
