"""CatBoost training pipeline: TSCV + Optuna + MLflow.

Orchestrates hyperparameter optimization via Optuna, cross-validated
training on calendar-month splits, and final model training on all data.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from loguru import logger
from optuna import Study, Trial, TrialPruned, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from energy_forecast.config.settings import Settings
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult, compute_all
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitResult:
    """Result from a single CV split."""

    split_idx: int
    train_metrics: MetricsResult
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    best_iteration: int
    val_month: str
    test_month: str
    val_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    val_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Aggregated result across all CV splits."""

    split_results: list[SplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float
    avg_best_iteration: int
    feature_names: list[str]


@dataclass(frozen=True)
class PipelineResult:
    """Full training pipeline result."""

    study: Study
    best_params: dict[str, Any]
    training_result: TrainingResult
    final_model: CatBoostRegressor
    training_time_seconds: float


# ---------------------------------------------------------------------------
# CatBoostTrainer
# ---------------------------------------------------------------------------


class CatBoostTrainer:
    """CatBoost training pipeline with TSCV, Optuna, and MLflow.

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
        self._cb_config = settings.catboost
        self._hp_config = settings.hyperparameters
        self._search_config = settings.hyperparameters.catboost
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(settings.hyperparameters.cross_validation)
        self._target_col = settings.hyperparameters.target_col
        self._skip_validation = settings.hyperparameters.skip_validation_after_optuna

    # -- Optuna storage --

    def _optuna_storage(self, model_name: str) -> optuna.storages.RDBStorage | str | None:
        """Return Optuna storage: PostgreSQL if available, else SQLite.

        Returns None for very short runs (n_trials <= 3) to avoid overhead.
        """
        if self._search_config.n_trials <= 3:
            return None
        db_url = os.environ.get("DATABASE_URL_SYNC", "")
        if db_url:
            return optuna.storages.RDBStorage(
                url=db_url,
                engine_kwargs={"pool_size": 1, "max_overflow": 0},
            )
        studies_dir = Path(self._settings.paths.models_dir) / "optuna_studies"
        studies_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{studies_dir / model_name}.db"

    # -- X/y split (resolves M4 leakage audit warning) --

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series[Any]]:
        """Separate target column from features."""
        y: pd.Series[Any] = df[self._target_col]
        x = df.drop(columns=[self._target_col])
        return x, y

    # -- Categorical preparation --

    def _prepare_categoricals(
        self, x: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[int]]:
        """Convert categoricals to str, fill NaN, return (df_copy, column indices).

        Returns a defensive copy to avoid mutating the caller's DataFrame.
        """
        x = x.copy()
        cat_cols = [c for c in self._cb_config.categorical_features if c in x.columns]
        fill_val = self._cb_config.nan_handling.categorical
        for col in cat_cols:
            x[col] = x[col].fillna(fill_val).astype(str)
        return x, [x.columns.get_loc(c) for c in cat_cols]  # type: ignore[misc]

    # -- Single split training --

    def _train_split(
        self,
        split_info: SplitInfo,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> SplitResult:
        """Train CatBoost on a single CV split."""
        x_train, y_train = self._split_xy(train_df)
        x_val, y_val = self._split_xy(val_df)
        x_test, y_test = self._split_xy(test_df)

        x_train, cat_idx = self._prepare_categoricals(x_train)
        x_val, _ = self._prepare_categoricals(x_val)
        x_test, _ = self._prepare_categoricals(x_test)

        train_pool = Pool(x_train, label=y_train, cat_features=cat_idx)
        val_pool = Pool(x_val, label=y_val, cat_features=cat_idx)

        model = CatBoostRegressor(**params, allow_writing_files=False)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self._cb_config.training.early_stopping_rounds,
            verbose=self._cb_config.training.verbose,
        )

        train_pred: np.ndarray[Any, np.dtype[np.floating[Any]]] = model.predict(x_train)
        val_pred: np.ndarray[Any, np.dtype[np.floating[Any]]] = model.predict(x_val)
        test_pred: np.ndarray[Any, np.dtype[np.floating[Any]]] = model.predict(x_test)

        return SplitResult(
            split_idx=split_info.split_idx,
            train_metrics=compute_all(y_train.to_numpy(), train_pred),
            val_metrics=compute_all(y_val.to_numpy(), val_pred),
            test_metrics=compute_all(y_test.to_numpy(), test_pred),
            best_iteration=int(model.best_iteration_),
            val_month=split_info.val_start.strftime("%Y-%m"),
            test_month=split_info.test_start.strftime("%Y-%m"),
            val_predictions=val_pred,
            val_actuals=y_val.to_numpy(),
            test_predictions=test_pred,
            test_actuals=y_test.to_numpy(),
        )

    # -- All splits training --

    def _train_all_splits(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        trial: Trial | None = None,
    ) -> TrainingResult:
        """Train on all TSCV splits and aggregate results."""
        x_sample, _ = self._split_xy(df.iloc[:1])
        results: list[SplitResult] = []

        for fold_idx, (info, train_df, val_df, test_df) in enumerate(
            self._splitter.iter_splits(df)
        ):
            result = self._train_split(info, train_df, val_df, test_df, params)
            results.append(result)
            logger.info(
                "Split {} | val={} MAPE={:.2f}% | test={} MAPE={:.2f}%",
                result.split_idx,
                result.val_month,
                result.val_metrics.mape,
                result.test_month,
                result.test_metrics.mape,
            )

            if trial is not None:
                trial.report(result.val_metrics.mape, fold_idx)
                if trial.should_prune():
                    raise TrialPruned()

        val_mapes = [r.val_metrics.mape for r in results]
        test_mapes = [r.test_metrics.mape for r in results]
        best_iters = [r.best_iteration for r in results]

        return TrainingResult(
            split_results=results,
            avg_val_mape=float(np.mean(val_mapes)),
            avg_test_mape=float(np.mean(test_mapes)),
            std_val_mape=float(np.std(val_mapes)),
            avg_best_iteration=int(np.mean(best_iters)),
            feature_names=list(x_sample.columns),
        )

    def _get_fixed_params(self) -> dict[str, Any]:
        """Return CatBoost parameters that stay constant across all trials."""
        return {
            "task_type": self._cb_config.training.task_type,
            "iterations": self._cb_config.training.iterations,
            "eval_metric": self._cb_config.training.eval_metric,
            "random_seed": self._cb_config.training.random_seed,
            "has_time": self._cb_config.training.has_time,
            "use_best_model": True,
        }

    # -- Optuna objective (dynamic from YAML) --

    def _create_objective(
        self, df: pd.DataFrame
    ) -> tuple[Callable[[Trial], float], dict[int, TrainingResult]]:
        """Create Optuna objective that uses dynamic YAML search space.

        Returns:
            Tuple of (objective function, trial results cache).
        """
        search_space = self._search_config.search_space
        fixed_params = self._get_fixed_params()
        trial_results: dict[int, TrainingResult] = {}

        def objective(trial: Trial) -> float:
            suggested = suggest_params(trial, search_space)
            params = {**fixed_params, **suggested}
            result = self._train_all_splits(df, params, trial=trial)
            trial.set_user_attr("avg_best_iteration", result.avg_best_iteration)
            trial.set_user_attr("avg_test_mape", result.avg_test_mape)
            trial_results[trial.number] = result
            return result.avg_val_mape

        return objective, trial_results

    # -- Optimize --

    def optimize(self, df: pd.DataFrame) -> tuple[Study, TrainingResult]:
        """Run Optuna hyperparameter optimization.

        Returns:
            Tuple of (study, best_trial_result).
        """
        storage = self._optuna_storage("catboost")
        study = create_study(
            study_name="catboost",
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(
                multivariate=True,
                seed=self._cb_config.training.random_seed,
            ),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        )

        objective, trial_results = self._create_objective(df)
        study.optimize(objective, n_trials=self._search_config.n_trials)

        logger.info(
            "Optimization done — best val MAPE: {:.2f}%, params: {}",
            study.best_value,
            study.best_params,
        )

        best_params = {**self._get_fixed_params(), **study.best_params}
        best_trial_num = study.best_trial.number

        if best_trial_num in trial_results:
            best_result = trial_results[best_trial_num]
            logger.info("Using cached predictions from trial {}", best_trial_num)
        elif self._skip_validation:
            logger.info("Skipping post-Optuna validation (skip_validation_after_optuna=true)")
            x_sample, _ = self._split_xy(df.iloc[:1])
            best_result = TrainingResult(
                split_results=[],
                avg_val_mape=study.best_value,
                avg_test_mape=float(
                    study.best_trial.user_attrs.get("avg_test_mape", float("nan"))
                ),
                std_val_mape=0.0,
                avg_best_iteration=int(
                    study.best_trial.user_attrs.get("avg_best_iteration", 500)
                ),
                feature_names=list(x_sample.columns),
            )
        else:
            logger.warning("Cache miss for trial {}, retraining", best_trial_num)
            best_result = self._train_all_splits(df, best_params)

        return study, best_result

    # -- Final model --

    def train_final(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        n_iterations: int,
    ) -> CatBoostRegressor:
        """Train final model on all data with best params.

        Args:
            df: Full dataset.
            params: Best hyperparameters from optimization.
            n_iterations: Average best iteration from CV splits.

        Returns:
            Trained CatBoostRegressor.
        """
        x, y = self._split_xy(df)
        x, cat_idx = self._prepare_categoricals(x)

        final_params = {
            **self._get_fixed_params(),
            **params,
            "iterations": n_iterations,
            "use_best_model": False,  # no eval_set in final training
        }
        model = CatBoostRegressor(**final_params, allow_writing_files=False)
        train_pool = Pool(x, label=y, cat_features=cat_idx)
        model.fit(train_pool, verbose=self._cb_config.training.verbose)

        logger.info("Final model trained — iterations: {}", n_iterations)
        return model

    # -- Full pipeline --

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """Execute full training pipeline: optimize + final model + MLflow.

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            PipelineResult with study, final model, and metrics.
        """
        start = time.monotonic()

        with self._tracker.start_run("catboost_optimization"):
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

        with self._tracker.start_run("catboost_final"):
            final_model = self.train_final(df, study.best_params, best_result.avg_best_iteration)

            # Save model to timestamped subdirectory
            from datetime import datetime

            from energy_forecast.utils import TZ_ISTANBUL

            run_ts = datetime.now(tz=TZ_ISTANBUL).strftime("%Y-%m-%d_%H-%M")
            model_dir = (
                Path(self._settings.paths.models_dir)
                / "catboost"
                / f"catboost_{run_ts}"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.cbm"
            final_model.save_model(str(model_path))
            logger.info("Model saved to {}", model_path)

            self._tracker.log_model(final_model, artifact_path="catboost_model")

            importance = dict(
                zip(
                    best_result.feature_names,
                    [float(v) for v in final_model.get_feature_importance()],
                    strict=True,
                )
            )
            self._tracker.log_feature_importance(importance)

        elapsed = time.monotonic() - start
        logger.info("Pipeline complete in {:.1f}s", elapsed)

        return PipelineResult(
            study=study,
            best_params=study.best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )
