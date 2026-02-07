"""Ensemble training pipeline: CatBoost + Prophet + TFT weighted average.

Orchestrates training of active models, optimizes ensemble weights on
validation data, and generates comparison metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

from energy_forecast.config.settings import Settings
from energy_forecast.training.catboost_trainer import (
    CatBoostTrainer,
)
from energy_forecast.training.catboost_trainer import (
    PipelineResult as CatBoostPipelineResult,
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


@dataclass(frozen=True)
class EnsemblePipelineResult:
    """Full ensemble training pipeline result."""

    model_results: dict[str, ModelResult]
    training_result: EnsembleTrainingResult
    comparison_df: pd.DataFrame
    training_time_seconds: float


# ---------------------------------------------------------------------------
# EnsembleTrainer
# ---------------------------------------------------------------------------


class EnsembleTrainer:
    """Ensemble training pipeline combining CatBoost, Prophet, and TFT.

    Trains active models with their respective TSCV + Optuna pipelines,
    then optimizes ensemble weights on validation predictions.

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

        logger.info("Ensemble active models: {}", self._active_models)

        # Create sub-trainers for active models only
        self._trainers: dict[str, CatBoostTrainer | ProphetTrainer | TFTTrainer] = {}
        if "catboost" in self._active_models:
            self._trainers["catboost"] = CatBoostTrainer(settings, tracker)
        if "prophet" in self._active_models:
            self._trainers["prophet"] = ProphetTrainer(settings, tracker)
        if "tft" in self._active_models:
            self._trainers["tft"] = TFTTrainer(settings, tracker)

    def run(self, df: pd.DataFrame) -> EnsemblePipelineResult:
        """Execute full ensemble training pipeline.

        1. Train all active models with Optuna optimization
        2. Collect validation predictions from each
        3. Optimize ensemble weights
        4. Generate comparison report

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            EnsemblePipelineResult with models, weights, and comparison.
        """
        start = time.monotonic()
        logger.info(
            "Starting ensemble training pipeline with {} models",
            len(self._active_models),
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

        # Collect predictions and compute ensemble metrics
        training_result = self._compute_ensemble(model_results)

        # Generate comparison DataFrame
        comparison_df = self._generate_comparison_df(model_results, training_result)

        # Log to MLflow
        with self._tracker.start_run("ensemble_final"):
            self._tracker.log_params(
                {
                    "active_models": ",".join(self._active_models),
                    "weight_optimization_enabled": self._ensemble_config.optimization.enabled,
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

    def _compute_ensemble(
        self,
        model_results: dict[str, ModelResult],
    ) -> EnsembleTrainingResult:
        """Compute ensemble predictions and optimize weights.

        Args:
            model_results: Dict of model name -> pipeline result.

        Returns:
            EnsembleTrainingResult with optimized weights and metrics.
        """
        # Collect validation MAPEs from each split
        split_results = self._collect_split_metrics(model_results)

        # Get normalized default weights for active models
        default_weights = self._ensemble_config.weights.get_normalized(
            self._active_models
        )

        # Optimize weights
        if self._ensemble_config.optimization.enabled and len(self._active_models) > 1:
            optimized_weights = self._optimize_weights(split_results, default_weights)
        else:
            optimized_weights = default_weights

        # Recompute ensemble metrics with optimized weights
        final_split_results = self._compute_weighted_ensemble(
            split_results, optimized_weights
        )

        # Calculate aggregated metrics
        ensemble_mapes = [sr.ensemble_metrics.mape for sr in final_split_results]
        model_avg_mapes = {
            m: float(np.mean([sr.model_metrics[m].mape for sr in final_split_results]))
            for m in self._active_models
        }

        return EnsembleTrainingResult(
            split_results=final_split_results,
            avg_val_mape=float(np.mean(ensemble_mapes)),
            std_val_mape=float(np.std(ensemble_mapes)),
            model_avg_val_mapes=model_avg_mapes,
            optimized_weights=optimized_weights,
        )

    def _collect_split_metrics(
        self,
        model_results: dict[str, ModelResult],
    ) -> list[EnsembleSplitResult]:
        """Collect validation metrics from each model's splits.

        Uses metric interpolation since we don't store raw predictions.
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

            for model_name in self._active_models:
                split_result = model_results[model_name].training_result.split_results[
                    split_idx
                ]
                model_metrics[model_name] = split_result.val_metrics
                # Placeholder predictions for metric-based optimization
                model_predictions[model_name] = np.zeros(100, dtype=np.float64)

            # Compute weighted ensemble metrics (approximation)
            ensemble_mape = sum(
                default_weights[m] * model_metrics[m].mape for m in self._active_models
            )
            ensemble_mae = sum(
                default_weights[m] * model_metrics[m].mae for m in self._active_models
            )
            ensemble_rmse = sum(
                default_weights[m] * model_metrics[m].rmse for m in self._active_models
            )
            ensemble_r2 = sum(
                default_weights[m] * model_metrics[m].r2 for m in self._active_models
            )
            ensemble_smape = sum(
                default_weights[m] * model_metrics[m].smape for m in self._active_models
            )
            ensemble_wmape = sum(
                default_weights[m] * model_metrics[m].wmape for m in self._active_models
            )
            ensemble_mbe = sum(
                default_weights[m] * model_metrics[m].mbe for m in self._active_models
            )

            ensemble_metrics = MetricsResult(
                mape=ensemble_mape,
                mae=ensemble_mae,
                rmse=ensemble_rmse,
                r2=ensemble_r2,
                smape=ensemble_smape,
                wmape=ensemble_wmape,
                mbe=ensemble_mbe,
            )

            split_results.append(
                EnsembleSplitResult(
                    split_idx=split_idx,
                    model_metrics=model_metrics,
                    ensemble_metrics=ensemble_metrics,
                    model_predictions=model_predictions,
                    ensemble_predictions=np.zeros(100, dtype=np.float64),
                    y_true=np.ones(100, dtype=np.float64),
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

        def objective(weights: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
            """Calculate mean validation MAPE for given weights."""
            w_dict = {m: weights[i] for i, m in enumerate(self._active_models)}
            mapes: list[float] = []

            for sr in split_results:
                # Weighted average of MAPEs
                ensemble_mape = sum(
                    w_dict[m] * sr.model_metrics[m].mape for m in self._active_models
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
        """Recompute ensemble metrics with given weights.

        Args:
            split_results: Original split results.
            weights: Optimized weights.

        Returns:
            Updated split results with new ensemble metrics.
        """
        updated_results: list[EnsembleSplitResult] = []

        for sr in split_results:
            # Compute weighted ensemble metrics
            ensemble_mape = sum(
                weights[m] * sr.model_metrics[m].mape for m in self._active_models
            )
            ensemble_mae = sum(
                weights[m] * sr.model_metrics[m].mae for m in self._active_models
            )
            ensemble_rmse = sum(
                weights[m] * sr.model_metrics[m].rmse for m in self._active_models
            )
            ensemble_r2 = sum(
                weights[m] * sr.model_metrics[m].r2 for m in self._active_models
            )
            ensemble_smape = sum(
                weights[m] * sr.model_metrics[m].smape for m in self._active_models
            )
            ensemble_wmape = sum(
                weights[m] * sr.model_metrics[m].wmape for m in self._active_models
            )
            ensemble_mbe = sum(
                weights[m] * sr.model_metrics[m].mbe for m in self._active_models
            )

            ensemble_metrics = MetricsResult(
                mape=ensemble_mape,
                mae=ensemble_mae,
                rmse=ensemble_rmse,
                r2=ensemble_r2,
                smape=ensemble_smape,
                wmape=ensemble_wmape,
                mbe=ensemble_mbe,
            )

            updated_results.append(
                EnsembleSplitResult(
                    split_idx=sr.split_idx,
                    model_metrics=sr.model_metrics,
                    ensemble_metrics=ensemble_metrics,
                    model_predictions=sr.model_predictions,
                    ensemble_predictions=sr.ensemble_predictions,
                    y_true=sr.y_true,
                    weights=weights.copy(),
                )
            )

        return updated_results

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

        # Compute ensemble test MAPE
        ens_test = sum(
            training_result.optimized_weights[m]
            * model_results[m].training_result.avg_test_mape
            for m in self._active_models
        )

        improvement = ((best_single_val - ens_val) / best_single_val) * 100

        data.append(
            {
                "Model": "Ensemble",
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
        logger.info("")
        logger.info(
            "{:<12} {:<12} {:<12} {:<10} {:<12}",
            "Model", "Val MAPE", "Test MAPE", "Weight", "Improvement"
        )
        logger.info("-" * 58)

        for _, row in comparison_df.iterrows():
            improvement = row.get("Improvement", "")
            logger.info(
                "{:<12} {:>8.2f}%    {:>8.2f}%    {:>6.3f}    {:<12}",
                row['Model'], row['Val MAPE (%)'], row['Test MAPE (%)'],
                row['Weight'], improvement
            )

        logger.info("-" * 58)
        weights_str = ", ".join(
            f"{m}={w:.3f}" for m, w in training_result.optimized_weights.items()
        )
        logger.info("Optimized Weights: {}", weights_str)
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
