"""Ensemble training pipeline: CatBoost + Prophet + TFT.

Supports two modes:
- stacking: CatBoost meta-learner with hour/weekday context (default)
- weighted_average: Global SLSQP weight optimization (fallback)

Orchestrates training of active models, then combines them via
the selected ensemble method.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from scipy.optimize import minimize

from energy_forecast.config import Settings
from energy_forecast.training.catboost_trainer import (
    CatBoostTrainer,
)
from energy_forecast.training.catboost_trainer import (
    PipelineResult as CatBoostPipelineResult,
)
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult, compute_all
from energy_forecast.training.metrics import mape as mape_fn
from energy_forecast.training.prophet_trainer import (
    ProphetPipelineResult,
    ProphetTrainer,
)
from energy_forecast.training.splitter import TimeSeriesSplitter
from energy_forecast.training.tft_trainer import (
    TFTPipelineResult,
    TFTTrainer,
)

# Type alias for any model pipeline result
ModelResult = CatBoostPipelineResult | ProphetPipelineResult | TFTPipelineResult

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleSplitResult:
    """Result from a single CV split for ensemble."""

    split_idx: int
    model_metrics: dict[str, MetricsResult]
    ensemble_metrics: MetricsResult
    model_predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]]
    ensemble_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]]
    y_true: np.ndarray[Any, np.dtype[np.floating[Any]]]
    weights: dict[str, float]


@dataclass(frozen=True)
class EnsembleTrainingResult:
    """Aggregated result from ensemble training."""

    split_results: list[EnsembleSplitResult]
    avg_val_mape: float
    std_val_mape: float
    model_avg_val_mapes: dict[str, float]
    optimized_weights: dict[str, float]
    mode: str = "weighted_average"
    avg_test_mape: float = 0.0
    std_test_mape: float = 0.0


@dataclass(frozen=True)
class EnsemblePipelineResult:
    """Full ensemble training pipeline result."""

    model_results: dict[str, ModelResult]
    training_result: EnsembleTrainingResult
    comparison_df: pd.DataFrame
    training_time_seconds: float
    meta_model: CatBoostRegressor | None = None


# ---------------------------------------------------------------------------
# EnsembleTrainer
# ---------------------------------------------------------------------------


class EnsembleTrainer:
    """Ensemble training pipeline combining CatBoost, Prophet, and TFT.

    Supports stacking (CatBoost meta-learner) and weighted_average modes.

    Args:
        settings: Full application settings.
        tracker: MLflow experiment tracker (disabled by default).
        active_models_override: Override active models from config.
    """

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
        active_models_override: list[str] | None = None,
    ) -> None:
        self._settings = settings
        self._ensemble_config = settings.ensemble
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._meta_model: CatBoostRegressor | None = None
        self._mode = self._ensemble_config.mode

        # Determine active models
        if active_models_override is not None:
            self._active_models = list(active_models_override)
        else:
            self._active_models = list(settings.ensemble.active_models)

        # Validate active models
        valid_models = {"catboost", "prophet", "tft"}
        for m in self._active_models:
            if m not in valid_models:
                msg = f"Unknown model: {m}. Valid: {valid_models}"
                raise ValueError(msg)

        # Stacking requires >= 2 models; fall back to weighted_average
        if self._mode == "stacking" and len(self._active_models) < 2:
            logger.warning(
                "Stacking requires >= 2 models, falling back to weighted_average"
            )
            self._mode = "weighted_average"

        logger.info(
            "Ensemble active models: {} | mode: {}", self._active_models, self._mode
        )

        # Create sub-trainers in active_models order (first model trains first)
        _factory: dict[
            str, Callable[[], CatBoostTrainer | ProphetTrainer | TFTTrainer]
        ] = {
            "catboost": lambda: CatBoostTrainer(settings, tracker),
            "prophet": lambda: ProphetTrainer(settings, tracker),
            "tft": lambda: TFTTrainer(settings, tracker),
        }
        self._trainers: dict[str, CatBoostTrainer | ProphetTrainer | TFTTrainer] = {}
        for model_name in self._active_models:
            if model_name in _factory:
                self._trainers[model_name] = _factory[model_name]()

    def run(self, df: pd.DataFrame) -> EnsemblePipelineResult:
        """Execute full ensemble training pipeline.

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            EnsemblePipelineResult with models, weights, and comparison.
        """
        start = time.monotonic()
        logger.info(
            "Starting ensemble training pipeline with {} models (mode={})",
            len(self._active_models),
            self._mode,
        )

        # Train all active models
        model_results, failed_models = self._train_models(df)

        # Update active models based on successful training
        successful_models = [m for m in self._active_models if m not in failed_models]
        if not successful_models:
            msg = f"All models failed. Errors: {failed_models}"
            raise RuntimeError(msg)

        if failed_models:
            logger.warning(
                "Some models failed, continuing with: {}. Failed: {}",
                successful_models,
                list(failed_models.keys()),
            )
            self._active_models = successful_models
            if self._mode == "stacking" and len(self._active_models) < 2:
                logger.warning("Too few models for stacking, falling back to weighted_average")
                self._mode = "weighted_average"

        # Collect predictions and compute ensemble metrics
        training_result = self._compute_ensemble(model_results, df)

        # Generate comparison DataFrame
        comparison_df = self._generate_comparison_df(model_results, training_result)

        # Log to MLflow
        with self._tracker.start_run("ensemble_final"):
            self._tracker.log_params(
                {
                    "active_models": ",".join(self._active_models),
                    "ensemble_mode": self._mode,
                    **{
                        f"weight_{m}": w
                        for m, w in training_result.optimized_weights.items()
                    },
                }
            )
            self._tracker.log_metrics(
                {
                    "ensemble_avg_val_mape": training_result.avg_val_mape,
                    "ensemble_std_val_mape": training_result.std_val_mape,
                    "ensemble_avg_test_mape": training_result.avg_test_mape,
                    **{
                        f"{m}_avg_val_mape": v
                        for m, v in training_result.model_avg_val_mapes.items()
                    },
                }
            )

        elapsed = time.monotonic() - start
        logger.info("Ensemble pipeline complete in {:.1f}s", elapsed)

        # Print comparison summary
        self._print_summary(comparison_df, training_result)

        return EnsemblePipelineResult(
            model_results=model_results,
            training_result=training_result,
            comparison_df=comparison_df,
            training_time_seconds=elapsed,
            meta_model=self._meta_model,
        )

    def _train_models(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, ModelResult], dict[str, Exception]]:
        """Train all active models.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Tuple of (successful results dict, failed models dict with errors).
        """
        results: dict[str, ModelResult] = {}
        errors: dict[str, Exception] = {}

        for model_name, trainer in self._trainers.items():
            try:
                logger.info("Training {} model...", model_name)
                result = trainer.run(df)
                results[model_name] = result
                logger.info(
                    "{} training complete — val MAPE: {:.2f}%",
                    model_name,
                    result.training_result.avg_val_mape,
                )
            except Exception as e:
                errors[model_name] = e
                logger.warning("{} training failed: {}", model_name, e)

                if not self._ensemble_config.fallback.enabled:
                    raise RuntimeError(
                        f"{model_name} failed and fallback disabled: {e}"
                    ) from e

        return results, errors

    # -- Ensemble computation (mode branching) --

    def _compute_ensemble(
        self,
        model_results: dict[str, ModelResult],
        df: pd.DataFrame,
    ) -> EnsembleTrainingResult:
        """Compute ensemble predictions and optimize weights or train meta-learner.

        Args:
            model_results: Dict of model name -> pipeline result.
            df: Original feature DataFrame (for OOF timestamp reconstruction).

        Returns:
            EnsembleTrainingResult with optimized weights/meta-learner and metrics.
        """
        # Collect val/test metrics from each split (shared by both modes)
        split_results = self._collect_split_metrics(model_results)

        default_weights = self._ensemble_config.weights.get_normalized(
            self._active_models
        )

        if self._mode == "stacking":
            return self._compute_stacking_ensemble(
                model_results, split_results, default_weights, df
            )
        return self._compute_weighted_average_ensemble(
            model_results, split_results, default_weights
        )

    def _compute_stacking_ensemble(
        self,
        model_results: dict[str, ModelResult],
        split_results: list[EnsembleSplitResult],
        default_weights: dict[str, float],
        df: pd.DataFrame,
    ) -> EnsembleTrainingResult:
        """Stacking mode: train CatBoost meta-learner on OOF predictions."""
        logger.info("Building OOF prediction matrix for stacking...")
        oof_df = self._build_oof_dataframe(model_results, df)
        logger.info("OOF matrix: {} rows x {} cols", len(oof_df), len(oof_df.columns))

        # Train meta-learner
        meta_model, meta_val_mape = self._train_meta_learner(oof_df)
        self._meta_model = meta_model

        # Compute real test MAPE via meta-learner on test predictions
        test_mapes = self._compute_stacking_test_mape(model_results, df)

        # Ensemble val MAPE = meta-learner hold-out MAPE
        # Per-model val MAPEs from split results
        model_avg_mapes = {
            m: float(np.mean([sr.model_metrics[m].mape for sr in split_results]))
            for m in self._active_models
        }

        return EnsembleTrainingResult(
            split_results=split_results,
            avg_val_mape=meta_val_mape,
            std_val_mape=0.0,  # Single hold-out, no per-fold std
            model_avg_val_mapes=model_avg_mapes,
            optimized_weights=default_weights,  # Unused in stacking, kept for compat
            mode="stacking",
            avg_test_mape=float(np.mean(test_mapes)) if test_mapes else 0.0,
            std_test_mape=float(np.std(test_mapes)) if test_mapes else 0.0,
        )

    def _compute_weighted_average_ensemble(
        self,
        model_results: dict[str, ModelResult],
        split_results: list[EnsembleSplitResult],
        default_weights: dict[str, float],
    ) -> EnsembleTrainingResult:
        """Weighted average mode: SLSQP weight optimization + real test MAPE."""
        # Optimize weights
        if self._ensemble_config.optimization.enabled and len(self._active_models) > 1:
            optimized_weights = self._optimize_weights(split_results, default_weights)
        else:
            optimized_weights = default_weights

        # Recompute val metrics with optimized weights
        final_split_results = self._compute_weighted_ensemble(
            split_results, optimized_weights
        )

        ensemble_mapes = [sr.ensemble_metrics.mape for sr in final_split_results]
        model_avg_mapes = {
            m: float(np.mean([sr.model_metrics[m].mape for sr in final_split_results]))
            for m in self._active_models
        }

        # Real blended test MAPE across all splits
        test_mapes = self._compute_weighted_test_mape(
            model_results, optimized_weights
        )

        return EnsembleTrainingResult(
            split_results=final_split_results,
            avg_val_mape=float(np.mean(ensemble_mapes)),
            std_val_mape=float(np.std(ensemble_mapes)),
            model_avg_val_mapes=model_avg_mapes,
            optimized_weights=optimized_weights,
            mode="weighted_average",
            avg_test_mape=float(np.mean(test_mapes)) if test_mapes else 0.0,
            std_test_mape=float(np.std(test_mapes)) if test_mapes else 0.0,
        )

    # -- OOF builder (stacking) --

    def _build_oof_dataframe(
        self,
        model_results: dict[str, ModelResult],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build OOF prediction matrix from all CV splits with context features.

        Reconstructs timestamps by re-running the splitter to get val indices,
        then joins base model val_predictions with hour/weekday/holiday context.

        Args:
            model_results: Base model training results with val_predictions.
            df: Original feature DataFrame with DatetimeIndex.

        Returns:
            DataFrame: [pred_catboost, pred_prophet, ..., hour, dow, ..., y_true]
        """
        cv_config = self._settings.hyperparameters.cross_validation
        splitter = TimeSeriesSplitter.from_config(cv_config)
        context_features = list(self._ensemble_config.stacking.context_features)

        oof_parts: list[pd.DataFrame] = []
        for split_info, _train_df, val_slice, _test_df in splitter.iter_splits(df):
            split_idx = split_info.split_idx

            # Collect each model's val predictions for this split
            preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
            y_true: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

            for model_name in self._active_models:
                sr = model_results[model_name].training_result.split_results[split_idx]
                if sr.val_predictions is not None:
                    preds[model_name] = sr.val_predictions
                    if y_true is None and sr.val_actuals is not None:
                        y_true = sr.val_actuals

            if not preds or y_true is None:
                continue

            # Truncate to common length (TFT may produce fewer rows)
            min_len = min(len(p) for p in preds.values())
            min_len = min(min_len, len(y_true))
            val_slice = val_slice.iloc[:min_len]

            row_df = pd.DataFrame(index=val_slice.index)
            for model_name, pred_arr in preds.items():
                row_df[f"pred_{model_name}"] = pred_arr[:min_len]

            # Context features from DatetimeIndex
            dt_idx = pd.DatetimeIndex(val_slice.index)
            if "hour" in context_features:
                row_df["hour"] = dt_idx.hour
            if "day_of_week" in context_features:
                row_df["day_of_week"] = dt_idx.dayofweek
            if "is_weekend" in context_features:
                row_df["is_weekend"] = (dt_idx.dayofweek >= 5).astype(int)
            if "month" in context_features:
                row_df["month"] = dt_idx.month
            if "is_holiday" in context_features:
                if "is_holiday" in val_slice.columns:
                    row_df["is_holiday"] = val_slice["is_holiday"].values[:min_len]
                else:
                    row_df["is_holiday"] = 0

            row_df["y_true"] = y_true[:min_len]
            oof_parts.append(row_df)

        return pd.concat(oof_parts, axis=0).sort_index()

    def _train_meta_learner(
        self, oof_df: pd.DataFrame
    ) -> tuple[CatBoostRegressor, float]:
        """Train CatBoost meta-learner on OOF predictions.

        Uses temporal 80/20 split for validation (no shuffle).

        Args:
            oof_df: OOF DataFrame with pred_*, context, and y_true columns.

        Returns:
            Tuple of (trained meta-learner, validation MAPE).
        """
        cfg = self._ensemble_config.stacking.meta_learner
        feature_cols = [c for c in oof_df.columns if c != "y_true"]

        x_meta = oof_df[feature_cols].copy()
        y_meta = oof_df["y_true"]

        # Categorical features for CatBoost
        cat_cols = [c for c in ["hour", "day_of_week", "month"] if c in feature_cols]
        for col in cat_cols:
            x_meta[col] = x_meta[col].astype(str)
        cat_indices = [feature_cols.index(c) for c in cat_cols]

        # Temporal 80/20 split (no shuffle — time series)
        split_point = int(len(x_meta) * 0.8)
        x_train, x_val = x_meta.iloc[:split_point], x_meta.iloc[split_point:]
        y_train, y_val = y_meta.iloc[:split_point], y_meta.iloc[split_point:]

        logger.info(
            "Training meta-learner: {} features, {} train / {} val rows",
            len(feature_cols), len(x_train), len(x_val),
        )

        meta_model = CatBoostRegressor(
            depth=cfg.depth,
            iterations=cfg.iterations,
            learning_rate=cfg.learning_rate,
            loss_function=cfg.loss_function,
            l2_leaf_reg=cfg.l2_leaf_reg,
            early_stopping_rounds=cfg.early_stopping_rounds,
            task_type=cfg.task_type,
            cat_features=cat_indices,
            verbose=50,
        )
        meta_model.fit(x_train, y_train, eval_set=(x_val, y_val))

        # Validation MAPE
        val_pred = meta_model.predict(x_val)
        meta_val_mape = float(mape_fn(y_val.to_numpy(), val_pred))

        # Log feature importance
        importances = dict(
            zip(feature_cols, meta_model.get_feature_importance(), strict=True)
        )
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("Meta-learner val MAPE: {:.3f}%", meta_val_mape)
        logger.info("Meta-learner feature importance (top 5):")
        for name, imp in sorted_imp[:5]:
            logger.info("  {}: {:.1f}", name, imp)

        return meta_model, meta_val_mape

    # -- Test MAPE computation (both modes) --

    def _compute_stacking_test_mape(
        self,
        model_results: dict[str, ModelResult],
        df: pd.DataFrame,
    ) -> list[float]:
        """Compute real test MAPE by applying meta-learner to test predictions.

        Args:
            model_results: Base model results with test_predictions per split.
            df: Original feature DataFrame for timestamp reconstruction.

        Returns:
            List of test MAPEs, one per CV split.
        """
        if self._meta_model is None:
            return []

        cv_config = self._settings.hyperparameters.cross_validation
        splitter = TimeSeriesSplitter.from_config(cv_config)
        context_features = list(self._ensemble_config.stacking.context_features)
        test_mapes: list[float] = []

        for split_info, _train_df, _val_df, test_slice in splitter.iter_splits(df):
            split_idx = split_info.split_idx

            # Collect test predictions
            preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
            y_test: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

            for model_name in self._active_models:
                sr = model_results[model_name].training_result.split_results[split_idx]
                if sr.test_predictions is not None:
                    preds[model_name] = sr.test_predictions
                    if y_test is None and sr.test_actuals is not None:
                        y_test = sr.test_actuals

            if not preds or y_test is None:
                continue

            min_len = min(len(p) for p in preds.values())
            min_len = min(min_len, len(y_test))
            test_slice = test_slice.iloc[:min_len]

            # Build meta-learner input
            meta_df = pd.DataFrame(index=test_slice.index)
            for model_name, pred_arr in preds.items():
                meta_df[f"pred_{model_name}"] = pred_arr[:min_len]

            dt_idx = pd.DatetimeIndex(test_slice.index)
            if "hour" in context_features:
                meta_df["hour"] = dt_idx.hour
            if "day_of_week" in context_features:
                meta_df["day_of_week"] = dt_idx.dayofweek
            if "is_weekend" in context_features:
                meta_df["is_weekend"] = (dt_idx.dayofweek >= 5).astype(int)
            if "month" in context_features:
                meta_df["month"] = dt_idx.month
            if "is_holiday" in context_features:
                if "is_holiday" in test_slice.columns:
                    meta_df["is_holiday"] = test_slice["is_holiday"].values[:min_len]
                else:
                    meta_df["is_holiday"] = 0

            # Categorical conversion
            for col in ["hour", "day_of_week", "month"]:
                if col in meta_df.columns:
                    meta_df[col] = meta_df[col].astype(str)

            ensemble_pred = self._meta_model.predict(meta_df)
            test_mapes.append(float(mape_fn(y_test[:min_len], ensemble_pred)))

        return test_mapes

    def _compute_weighted_test_mape(
        self,
        model_results: dict[str, ModelResult],
        weights: dict[str, float],
    ) -> list[float]:
        """Compute real blended test MAPE across all CV splits.

        Args:
            model_results: Base model results with test_predictions per split.
            weights: Optimized ensemble weights.

        Returns:
            List of test MAPEs, one per CV split.
        """
        first_model = next(iter(model_results.values()))
        n_splits = len(first_model.training_result.split_results)
        test_mapes: list[float] = []

        for split_idx in range(n_splits):
            preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
            y_test: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

            for model_name in self._active_models:
                sr = model_results[model_name].training_result.split_results[split_idx]
                if sr.test_predictions is not None:
                    preds[model_name] = sr.test_predictions
                    if y_test is None and sr.test_actuals is not None:
                        y_test = sr.test_actuals

            if not preds or y_test is None:
                continue

            min_len = min(len(p) for p in preds.values())
            min_len = min(min_len, len(y_test))
            preds = {m: p[:min_len] for m, p in preds.items()}

            blended = np.zeros(min_len, dtype=np.float64)
            for m in preds:
                blended += weights[m] * preds[m]
            test_mapes.append(float(mape_fn(y_test[:min_len], blended)))

        return test_mapes

    # -- Existing methods (val prediction collection, weight optimization) --

    def _collect_split_metrics(
        self,
        model_results: dict[str, ModelResult],
    ) -> list[EnsembleSplitResult]:
        """Collect validation metrics and raw predictions from each model's splits.

        Uses real val_predictions/val_actuals from each trainer's SplitResult
        for proper blended-prediction weight optimization.
        """
        # Get split count from first model
        first_model = next(iter(model_results.values()))
        n_splits = len(first_model.training_result.split_results)

        # Verify all models have same split count
        for model_name, result in model_results.items():
            if len(result.training_result.split_results) != n_splits:
                msg = f"Split count mismatch for {model_name}"
                raise ValueError(msg)

        default_weights = self._ensemble_config.weights.get_normalized(
            self._active_models
        )

        split_results: list[EnsembleSplitResult] = []
        for split_idx in range(n_splits):
            model_metrics: dict[str, MetricsResult] = {}
            model_predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
            y_true: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

            for model_name in self._active_models:
                split_result = model_results[model_name].training_result.split_results[
                    split_idx
                ]
                model_metrics[model_name] = split_result.val_metrics

                # Extract real predictions (available after trainer fix)
                if split_result.val_predictions is not None:
                    model_predictions[model_name] = split_result.val_predictions
                    if y_true is None and split_result.val_actuals is not None:
                        y_true = split_result.val_actuals

            # Truncate to common length (models may predict different lengths)
            if model_predictions:
                min_len = min(len(p) for p in model_predictions.values())
                model_predictions = {
                    m: p[:min_len] for m, p in model_predictions.items()
                }
                if y_true is not None:
                    y_true = y_true[:min_len]

            # Compute blended ensemble prediction with default weights
            if model_predictions and y_true is not None:
                blended = np.zeros(min_len, dtype=np.float64)
                for m in model_predictions:
                    blended += default_weights[m] * model_predictions[m]
                ensemble_metrics = compute_all(y_true, blended)
                ensemble_predictions = blended
            else:
                # Fallback: metric-level approximation (no raw predictions)
                logger.warning(
                    "Split {}: no raw predictions, using metric approximation",
                    split_idx,
                )
                ensemble_mape = sum(
                    default_weights[m] * model_metrics[m].mape
                    for m in self._active_models
                )
                dw = default_weights
                am = self._active_models
                mm = model_metrics
                ensemble_metrics = MetricsResult(
                    mape=ensemble_mape,
                    mae=sum(dw[m] * mm[m].mae for m in am),
                    rmse=sum(dw[m] * mm[m].rmse for m in am),
                    r2=sum(dw[m] * mm[m].r2 for m in am),
                    smape=sum(dw[m] * mm[m].smape for m in am),
                    wmape=sum(dw[m] * mm[m].wmape for m in am),
                    mbe=sum(dw[m] * mm[m].mbe for m in am),
                )
                ensemble_predictions = np.zeros(1, dtype=np.float64)
                y_true = np.ones(1, dtype=np.float64)

            split_results.append(
                EnsembleSplitResult(
                    split_idx=split_idx,
                    model_metrics=model_metrics,
                    ensemble_metrics=ensemble_metrics,
                    model_predictions=model_predictions,
                    ensemble_predictions=ensemble_predictions,
                    y_true=y_true,
                    weights=default_weights.copy(),
                )
            )

        return split_results

    def _optimize_weights(
        self,
        split_results: list[EnsembleSplitResult],
        initial_weights: dict[str, float],
    ) -> dict[str, float]:
        """Optimize weights using scipy.optimize.minimize.

        Uses SLSQP with constraint sum(weights)=1 and per-model bounds.

        Args:
            split_results: List of split results with model metrics.
            initial_weights: Initial weight values.

        Returns:
            Dictionary with optimized weights.
        """
        logger.info("Optimizing ensemble weights for {} models...", len(self._active_models))

        bounds_cfg = self._ensemble_config.optimization.bounds

        # Check if we have real predictions for blended optimization
        has_real_predictions = all(
            len(sr.model_predictions) == len(self._active_models)
            and all(len(p) > 1 for p in sr.model_predictions.values())
            for sr in split_results
        )

        def objective(weights: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
            """Calculate mean validation MAPE on blended predictions."""
            w_dict = {m: weights[i] for i, m in enumerate(self._active_models)}
            mapes: list[float] = []

            for sr in split_results:
                if has_real_predictions:
                    # Correct: MAPE of blended prediction vs actuals
                    blended = np.zeros_like(sr.y_true)
                    for m in self._active_models:
                        blended += w_dict[m] * sr.model_predictions[m]
                    mapes.append(mape_fn(sr.y_true, blended))
                else:
                    # Fallback: weighted average of MAPEs (approximation)
                    ensemble_mape = sum(
                        w_dict[m] * sr.model_metrics[m].mape
                        for m in self._active_models
                    )
                    mapes.append(ensemble_mape)

            return float(np.mean(mapes))

        # Constraint: sum(weights) = 1
        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

        # Per-model bounds
        bounds: list[tuple[float, float]] = []
        for model_name in self._active_models:
            model_bounds = getattr(bounds_cfg, model_name, (0.1, 0.9))
            bounds.append(model_bounds)

        # Initial guess from normalized default weights
        x0 = np.array([initial_weights[m] for m in self._active_models], dtype=np.float64)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 100},
        )

        optimized_weights = {
            m: float(result.x[i]) for i, m in enumerate(self._active_models)
        }
        optimized_mape = float(result.fun)

        logger.info(
            "Weight optimization complete: {} | MAPE={:.3f}%",
            {m: f"{w:.3f}" for m, w in optimized_weights.items()},
            optimized_mape,
        )

        return optimized_weights

    def _compute_weighted_ensemble(
        self,
        split_results: list[EnsembleSplitResult],
        weights: dict[str, float],
    ) -> list[EnsembleSplitResult]:
        """Recompute ensemble metrics with given weights using real predictions.

        Args:
            split_results: Original split results with real model predictions.
            weights: Optimized weights.

        Returns:
            Updated split results with new ensemble metrics.
        """
        updated_results: list[EnsembleSplitResult] = []

        for sr in split_results:
            has_preds = (
                len(sr.model_predictions) == len(self._active_models)
                and all(len(p) > 1 for p in sr.model_predictions.values())
            )

            if has_preds:
                # Compute blended prediction and real metrics
                blended = np.zeros_like(sr.y_true)
                for m in self._active_models:
                    blended += weights[m] * sr.model_predictions[m]
                ensemble_metrics = compute_all(sr.y_true, blended)
                ensemble_predictions = blended
            else:
                # Fallback: metric-level approximation
                ensemble_metrics = MetricsResult(
                    mape=sum(weights[m] * sr.model_metrics[m].mape for m in self._active_models),
                    mae=sum(weights[m] * sr.model_metrics[m].mae for m in self._active_models),
                    rmse=sum(weights[m] * sr.model_metrics[m].rmse for m in self._active_models),
                    r2=sum(weights[m] * sr.model_metrics[m].r2 for m in self._active_models),
                    smape=sum(weights[m] * sr.model_metrics[m].smape for m in self._active_models),
                    wmape=sum(weights[m] * sr.model_metrics[m].wmape for m in self._active_models),
                    mbe=sum(weights[m] * sr.model_metrics[m].mbe for m in self._active_models),
                )
                ensemble_predictions = sr.ensemble_predictions

            updated_results.append(
                EnsembleSplitResult(
                    split_idx=sr.split_idx,
                    model_metrics=sr.model_metrics,
                    ensemble_metrics=ensemble_metrics,
                    model_predictions=sr.model_predictions,
                    ensemble_predictions=ensemble_predictions,
                    y_true=sr.y_true,
                    weights=weights.copy(),
                )
            )

        return updated_results

    # -- Reporting --

    def _generate_comparison_df(
        self,
        model_results: dict[str, ModelResult],
        training_result: EnsembleTrainingResult,
    ) -> pd.DataFrame:
        """Generate comparison DataFrame for all models vs ensemble.

        Args:
            model_results: Dict of model pipeline results.
            training_result: Ensemble training result.

        Returns:
            DataFrame with comparison metrics.
        """
        data: list[dict[str, Any]] = []

        # Individual models
        for model_name in self._active_models:
            result = model_results[model_name]
            val_mape = result.training_result.avg_val_mape
            test_mape = result.training_result.avg_test_mape

            data.append(
                {
                    "Model": model_name.capitalize(),
                    "Val MAPE (%)": val_mape,
                    "Test MAPE (%)": test_mape,
                    "Weight": training_result.optimized_weights.get(model_name, 0.0),
                }
            )

        # Ensemble
        best_single_val = min(d["Val MAPE (%)"] for d in data)
        ens_val = training_result.avg_val_mape
        ens_test = training_result.avg_test_mape

        improvement = ((best_single_val - ens_val) / best_single_val) * 100

        data.append(
            {
                "Model": f"Ensemble ({training_result.mode})",
                "Val MAPE (%)": ens_val,
                "Test MAPE (%)": ens_test,
                "Weight": 1.0,
                "Improvement": f"{improvement:+.1f}%",
            }
        )

        return pd.DataFrame(data)

    def _print_summary(
        self,
        comparison_df: pd.DataFrame,
        training_result: EnsembleTrainingResult,
    ) -> None:
        """Print formatted comparison summary to terminal.

        Args:
            comparison_df: Comparison DataFrame.
            training_result: Ensemble training result.
        """
        logger.info("")
        logger.info("=" * 75)
        logger.info("                    ENSEMBLE TRAINING REPORT")
        logger.info("=" * 75)
        logger.info("  Mode: {}", training_result.mode)
        logger.info("")
        logger.info(
            "{:<20} {:<12} {:<12} {:<10} {:<12}",
            "Model", "Val MAPE", "Test MAPE", "Weight", "Improvement"
        )
        logger.info("-" * 66)

        for _, row in comparison_df.iterrows():
            improvement = row.get("Improvement", "")
            logger.info(
                "{:<20} {:>8.2f}%    {:>8.2f}%    {:>6.3f}    {:<12}",
                row['Model'], row['Val MAPE (%)'], row['Test MAPE (%)'],
                row['Weight'], improvement
            )

        logger.info("-" * 66)
        if training_result.mode == "weighted_average":
            weights_str = ", ".join(
                f"{m}={w:.3f}" for m, w in training_result.optimized_weights.items()
            )
            logger.info("Optimized Weights: {}", weights_str)
        else:
            ml_depth = self._ensemble_config.stacking.meta_learner.depth
            logger.info("Meta-learner: CatBoost depth={}", ml_depth)
        logger.info("Active Models: {}", ", ".join(self._active_models))
        logger.info("=" * 75)
        logger.info("")


def save_ensemble_weights(weights: dict[str, float], path: Path) -> None:
    """Save ensemble weights to JSON file.

    Args:
        weights: Dictionary with model weights.
        path: Path to save weights JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    logger.info("Saved ensemble weights to {}", path)


def load_ensemble_weights(path: Path) -> dict[str, float]:
    """Load ensemble weights from JSON file.

    Args:
        path: Path to weights JSON.

    Returns:
        Dictionary with model weights.
    """
    with open(path, encoding="utf-8") as f:
        weights: dict[str, float] = json.load(f)
    return weights
