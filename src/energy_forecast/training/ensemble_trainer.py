"""Ensemble training pipeline: CatBoost + Prophet + TFT.

Supports two modes:
- stacking: CatBoost meta-learner with hour/weekday context (default)
- weighted_average: Global SLSQP weight optimization (fallback)

Orchestrates training of active models, then combines them via
the selected ensemble method.

Heavy-lifting functions live in:
- ``ensemble_stacking`` — OOF builder, meta-learner, stacking test eval
- ``ensemble_weights``  — SLSQP optimizer, split metrics, weighted eval
- ``ensemble_report``   — comparison DataFrame, terminal summary
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from energy_forecast.config import Settings
from energy_forecast.training.catboost_trainer import (
    CatBoostTrainer,
)
from energy_forecast.training.catboost_trainer import (
    PipelineResult as CatBoostPipelineResult,
)
from energy_forecast.training.ensemble_report import (
    generate_comparison_df,
    print_summary,
)
from energy_forecast.training.ensemble_stacking import (
    build_oof_dataframe,
    compute_stacking_test_mape,
    train_meta_learner,
)
from energy_forecast.training.ensemble_weights import (
    collect_split_metrics,
    compute_weighted_ensemble,
    compute_weighted_test_mape,
    optimize_weights,
)
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult
from energy_forecast.training.prophet_trainer import (
    ProphetPipelineResult,
    ProphetTrainer,
)
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
                    "ensemble_std_test_mape": training_result.std_test_mape,
                    **{
                        f"{m}_avg_val_mape": v
                        for m, v in training_result.model_avg_val_mapes.items()
                    },
                }
            )

            # Log ensemble weights (existing method, was never called)
            self._tracker.log_ensemble_weights(training_result.optimized_weights)

            # Per-split ensemble + per-model metrics
            for sr in training_result.split_results:
                split_batch: dict[str, float] = {
                    f"ensemble_split_{sr.split_idx:02d}_mape": sr.ensemble_metrics.mape,
                    f"ensemble_split_{sr.split_idx:02d}_mae": sr.ensemble_metrics.mae,
                    f"ensemble_split_{sr.split_idx:02d}_rmse": sr.ensemble_metrics.rmse,
                    f"ensemble_split_{sr.split_idx:02d}_r2": sr.ensemble_metrics.r2,
                }
                # Per-model per-split MAPE
                for model_name, model_metric in sr.model_metrics.items():
                    split_batch[
                        f"{model_name}_split_{sr.split_idx:02d}_mape"
                    ] = model_metric.mape
                self._tracker.log_metrics(split_batch)

            # Stacking: log meta-model artifact + feature importance
            if self._mode == "stacking" and self._meta_model is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    meta_path = Path(tmpdir) / "meta_model.cbm"
                    self._meta_model.save_model(str(meta_path))
                    self._tracker.log_artifact(str(meta_path), "meta_model")

                # Meta-model feature importance
                meta_fi = dict(
                    zip(
                        self._meta_model.feature_names_,
                        [float(v) for v in self._meta_model.get_feature_importance()],
                        strict=True,
                    )
                )
                self._tracker.log_feature_importance(meta_fi, top_n=len(meta_fi))

            elapsed = time.monotonic() - start
            self._tracker.log_training_meta(
                {
                    "training_time_seconds": elapsed,
                    "active_model_count": len(self._active_models),
                    "ensemble_mode": self._mode,
                }
            )

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
        """Compute ensemble predictions and optimize weights or train meta-learner."""
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
            optimized_w = self._optimize_weights(split_results, default_weights)
        else:
            optimized_w = default_weights

        # Recompute val metrics with optimized weights
        final_split_results = self._compute_weighted_ensemble(
            split_results, optimized_w,
        )

        ensemble_mapes = [sr.ensemble_metrics.mape for sr in final_split_results]
        model_avg_mapes = {
            m: float(np.mean([sr.model_metrics[m].mape for sr in final_split_results]))
            for m in self._active_models
        }

        # Real blended test MAPE across all splits
        test_mapes = self._compute_weighted_test_mape(model_results, optimized_w)

        return EnsembleTrainingResult(
            split_results=final_split_results,
            avg_val_mape=float(np.mean(ensemble_mapes)),
            std_val_mape=float(np.std(ensemble_mapes)),
            model_avg_val_mapes=model_avg_mapes,
            optimized_weights=optimized_w,
            mode="weighted_average",
            avg_test_mape=float(np.mean(test_mapes)) if test_mapes else 0.0,
            std_test_mape=float(np.std(test_mapes)) if test_mapes else 0.0,
        )

    # -- Delegate methods (backward-compat for tests calling private API) --

    def _collect_split_metrics(
        self, model_results: dict[str, ModelResult],
    ) -> list[EnsembleSplitResult]:
        """Delegate to ``ensemble_weights.collect_split_metrics``."""
        default_weights = self._ensemble_config.weights.get_normalized(
            self._active_models
        )
        return collect_split_metrics(
            model_results, self._active_models, default_weights,
        )

    def _optimize_weights(
        self,
        split_results: list[EnsembleSplitResult],
        initial_weights: dict[str, float],
    ) -> dict[str, float]:
        """Delegate to ``ensemble_weights.optimize_weights``."""
        return optimize_weights(
            split_results, initial_weights,
            self._active_models, self._ensemble_config.optimization.bounds,
        )

    def _compute_weighted_ensemble(
        self,
        split_results: list[EnsembleSplitResult],
        weights: dict[str, float],
    ) -> list[EnsembleSplitResult]:
        """Delegate to ``ensemble_weights.compute_weighted_ensemble``."""
        return compute_weighted_ensemble(
            split_results, weights, self._active_models,
        )

    def _compute_weighted_test_mape(
        self,
        model_results: dict[str, ModelResult],
        weights: dict[str, float],
    ) -> list[float]:
        """Delegate to ``ensemble_weights.compute_weighted_test_mape``."""
        return compute_weighted_test_mape(
            model_results, weights, self._active_models,
        )

    def _build_oof_dataframe(
        self,
        model_results: dict[str, ModelResult],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Delegate to ``ensemble_stacking.build_oof_dataframe``."""
        cv_config = self._settings.hyperparameters.cross_validation
        context_features = list(self._ensemble_config.stacking.context_features)
        return build_oof_dataframe(
            model_results, df, self._active_models, cv_config, context_features,
        )

    def _train_meta_learner(
        self, oof_df: pd.DataFrame,
    ) -> tuple[CatBoostRegressor, float]:
        """Delegate to ``ensemble_stacking.train_meta_learner``."""
        meta_config = self._ensemble_config.stacking.meta_learner
        return train_meta_learner(oof_df, meta_config)

    def _compute_stacking_test_mape(
        self,
        model_results: dict[str, ModelResult],
        df: pd.DataFrame,
    ) -> list[float]:
        """Delegate to ``ensemble_stacking.compute_stacking_test_mape``."""
        if self._meta_model is None:
            return []
        cv_config = self._settings.hyperparameters.cross_validation
        context_features = list(self._ensemble_config.stacking.context_features)
        return compute_stacking_test_mape(
            self._meta_model, model_results, df,
            self._active_models, cv_config, context_features,
        )

    def _generate_comparison_df(
        self,
        model_results: dict[str, ModelResult],
        training_result: EnsembleTrainingResult,
    ) -> pd.DataFrame:
        """Delegate to ``ensemble_report.generate_comparison_df``."""
        return generate_comparison_df(
            model_results, training_result, self._active_models,
        )

    def _print_summary(
        self,
        comparison_df: pd.DataFrame,
        training_result: EnsembleTrainingResult,
    ) -> None:
        """Delegate to ``ensemble_report.print_summary``."""
        print_summary(
            comparison_df, training_result,
            self._active_models, self._ensemble_config,
        )


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
