"""CatBoost training pipeline: TSCV + Optuna + MLflow.

Orchestrates hyperparameter optimization via Optuna, cross-validated
training on calendar-month splits, and final model training on all data.
"""

from __future__ import annotations

import json
import sys
import tempfile
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

from energy_forecast.config import Settings
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import compute_all
from energy_forecast.training.results import SplitResult
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter
from energy_forecast.training.utils import optuna_storage


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

    study: Study | None
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
        *,
        force_hpo: bool = False,
    ) -> None:
        self._settings = settings
        self._cb_config = settings.catboost
        self._hp_config = settings.hyperparameters
        self._search_config = settings.hyperparameters.catboost
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(settings.hyperparameters.cross_validation)
        self._target_col = settings.hyperparameters.target_col
        self._skip_validation = settings.hyperparameters.skip_validation_after_optuna
        self._selected_features = self._load_selected_features()
        self._force_hpo = force_hpo

    # -- Feature selection --

    def _load_selected_features(self) -> list[str] | None:
        """Load selected feature list from JSON if configured."""
        path_str = self._cb_config.selected_features_path
        if path_str is None:
            return None
        path = Path(path_str)
        if not path.exists():
            logger.warning("Selected features file not found: {}", path)
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        features: list[str] = data["features"]
        logger.info("Loaded {} selected features from {}", len(features), path)
        return features

    # -- Optuna storage --

    def _optuna_storage(self, model_name: str) -> optuna.storages.RDBStorage | str | None:
        """Return Optuna storage (delegates to shared ``optuna_storage``)."""
        return optuna_storage(
            self._search_config.n_trials,
            model_name,
            self._settings.paths.models_dir,
        )

    # -- X/y split (resolves M4 leakage audit warning) --

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series[Any]]:
        """Separate target column from features, applying selection if configured."""
        y: pd.Series[Any] = df[self._target_col]
        x = df.drop(columns=[self._target_col])
        if self._selected_features is not None:
            keep = [c for c in self._selected_features if c in x.columns]
            x = x[keep]
        return x, y

    # -- Categorical preparation --

    def _prepare_categoricals(self, x: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
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

        # Inverse Box-Cox: metrics always in MWh space
        if self._settings.boxcox.enabled:
            from energy_forecast.transform import inv_boxcox

            lam = self._settings.boxcox.lambda_param
            y_train_mwh = inv_boxcox(y_train.to_numpy(), lam)
            y_val_mwh = inv_boxcox(y_val.to_numpy(), lam)
            y_test_mwh = inv_boxcox(y_test.to_numpy(), lam)
            train_pred_mwh = inv_boxcox(train_pred, lam)
            val_pred_mwh = inv_boxcox(val_pred, lam)
            test_pred_mwh = inv_boxcox(test_pred, lam)
        else:
            y_train_mwh = y_train.to_numpy()
            y_val_mwh = y_val.to_numpy()
            y_test_mwh = y_test.to_numpy()
            train_pred_mwh = train_pred
            val_pred_mwh = val_pred
            test_pred_mwh = test_pred

        return SplitResult(
            split_idx=split_info.split_idx,
            train_metrics=compute_all(y_train_mwh, train_pred_mwh),
            val_metrics=compute_all(y_val_mwh, val_pred_mwh),
            test_metrics=compute_all(y_test_mwh, test_pred_mwh),
            best_iteration=int(model.best_iteration_),
            val_month=split_info.val_start.strftime("%Y-%m"),
            test_month=split_info.test_start.strftime("%Y-%m"),
            val_predictions=val_pred_mwh,
            val_actuals=y_val_mwh,
            test_predictions=test_pred_mwh,
            test_actuals=y_test_mwh,
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
                avg_test_mape=float(study.best_trial.user_attrs.get("avg_test_mape", float("nan"))),
                std_val_mape=0.0,
                avg_best_iteration=int(study.best_trial.user_attrs.get("avg_best_iteration", 500)),
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
        """Execute full training pipeline: fixed params or Optuna HPO.

        Automatically selects mode based on ``best_params`` in catboost.yaml:
        - best_params populated and ``--force-hpo`` not set -> fixed mode
        - best_params empty or ``--force-hpo`` set -> Optuna HPO

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            PipelineResult with study, final model, and metrics.
        """
        has_best = bool(self._cb_config.best_params)
        if has_best and not self._force_hpo:
            logger.info("best_params found in catboost.yaml — using fixed mode")
            return self._run_fixed(df)
        if has_best and self._force_hpo:
            logger.info("--force-hpo: ignoring best_params, running Optuna")
        return self._run_optuna(df)

    def _run_fixed(self, df: pd.DataFrame) -> PipelineResult:
        """Skip Optuna, use best_params from YAML config.

        Runs 12-fold CV for OOF predictions, then trains final model.

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            PipelineResult with study=None.
        """
        start = time.monotonic()

        best = dict(self._cb_config.best_params)
        avg_iter = best.pop("avg_best_iteration", None)
        params = {**self._get_fixed_params(), **best}

        logger.info("Fixed params: {}", params)

        with self._tracker.start_run("catboost_fixed"):
            # 12-fold CV (OOF predictions for ensemble)
            best_result = self._train_all_splits(df, params)

            # Compute std_test_mape from split results
            test_mapes = [sr.test_metrics.mape for sr in best_result.split_results]
            std_test_mape = float(np.std(test_mapes)) if test_mapes else 0.0

            self._tracker.log_params(best)
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

            self._tracker.log_training_meta(
                {
                    "data_rows": len(df),
                    "data_cols": len(df.columns),
                    "n_splits": self._hp_config.cross_validation.n_splits,
                    "mode": "fixed",
                    "python_version": sys.version,
                    "platform": sys.platform,
                }
            )
            self._tracker.log_config_snapshot(
                self._settings.catboost.model_dump(),
                "catboost_config.yaml",
            )

        with self._tracker.start_run("catboost_final"):
            # Final model — use avg_best_iteration from YAML or from CV results
            n_iter = int(avg_iter) if avg_iter is not None else best_result.avg_best_iteration
            final_model = self.train_final(df, best, n_iter)

            # Save model to fixed directory (overwrite previous)
            model_dir = Path(self._settings.paths.models_dir) / "catboost"
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

            # Log ALL feature importances as artifact (JSON)
            with tempfile.TemporaryDirectory() as tmpdir:
                fi_path = Path(tmpdir) / "feature_importance_all.json"
                sorted_fi = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                fi_path.write_text(
                    json.dumps(sorted_fi, indent=2),
                    encoding="utf-8",
                )
                self._tracker.log_artifact(str(fi_path))

            # Log predictions summary from last split
            if best_result.split_results:
                last_sr = best_result.split_results[-1]
                if last_sr.val_predictions is not None and last_sr.val_actuals is not None:
                    self._tracker.log_predictions_summary(
                        last_sr.val_actuals,
                        last_sr.val_predictions,
                        prefix="final_val",
                    )
                if last_sr.test_predictions is not None and last_sr.test_actuals is not None:
                    self._tracker.log_predictions_summary(
                        last_sr.test_actuals,
                        last_sr.test_predictions,
                        prefix="final_test",
                    )

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {"training_time_seconds": elapsed},
            )

        logger.info("Pipeline complete (fixed mode) in {:.1f}s", elapsed)

        # Save OOF cache for ensemble
        from energy_forecast.training.oof_cache import compute_config_hash, save_oof_cache

        try:
            config_hash = compute_config_hash(self._settings, "catboost")
            save_oof_cache(
                "catboost", best_result.split_results, self._settings.paths.models_dir, config_hash
            )
        except Exception as e:
            logger.warning("Failed to save OOF cache (non-fatal): {}", e)

        return PipelineResult(
            study=None,
            best_params=best,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )

    def _run_optuna(self, df: pd.DataFrame) -> PipelineResult:
        """Execute Optuna HPO training pipeline: optimize + final model + MLflow.

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            PipelineResult with study, final model, and metrics.
        """
        start = time.monotonic()

        with self._tracker.start_run("catboost_optimization"):
            study, best_result = self.optimize(df)
            self._tracker.log_params(study.best_params)

            # Compute std_test_mape from split results
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
                self._settings.catboost.model_dump(),
                "catboost_config.yaml",
            )

        with self._tracker.start_run("catboost_final"):
            final_model = self.train_final(df, study.best_params, best_result.avg_best_iteration)

            # Save model to fixed directory (overwrite previous)
            model_dir = Path(self._settings.paths.models_dir) / "catboost"
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

            # Log ALL feature importances as artifact (JSON)
            with tempfile.TemporaryDirectory() as tmpdir:
                fi_path = Path(tmpdir) / "feature_importance_all.json"
                sorted_fi = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                fi_path.write_text(
                    json.dumps(sorted_fi, indent=2),
                    encoding="utf-8",
                )
                self._tracker.log_artifact(str(fi_path))

            # Log predictions summary from last split
            if best_result.split_results:
                last_sr = best_result.split_results[-1]
                if last_sr.val_predictions is not None and last_sr.val_actuals is not None:
                    self._tracker.log_predictions_summary(
                        last_sr.val_actuals,
                        last_sr.val_predictions,
                        prefix="final_val",
                    )
                if last_sr.test_predictions is not None and last_sr.test_actuals is not None:
                    self._tracker.log_predictions_summary(
                        last_sr.test_actuals,
                        last_sr.test_predictions,
                        prefix="final_test",
                    )

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {"training_time_seconds": elapsed},
            )

        logger.info("Pipeline complete in {:.1f}s", elapsed)

        # Save OOF cache for ensemble
        from energy_forecast.training.oof_cache import compute_config_hash, save_oof_cache

        try:
            config_hash = compute_config_hash(self._settings, "catboost")
            save_oof_cache(
                "catboost", best_result.split_results, self._settings.paths.models_dir, config_hash
            )
        except Exception as e:
            logger.warning("Failed to save OOF cache (non-fatal): {}", e)

        return PipelineResult(
            study=study,
            best_params=study.best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )
