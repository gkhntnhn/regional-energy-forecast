"""Ensemble weight optimization and weighted-average evaluation.

Extracted from ``ensemble_trainer.py``. Contains SLSQP weight optimization,
split metric collection, and blended prediction recomputation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from energy_forecast.training.metrics import MetricsResult, compute_all
from energy_forecast.training.metrics import mape as mape_fn

if TYPE_CHECKING:
    from energy_forecast.training.ensemble_trainer import EnsembleSplitResult


def collect_split_metrics(
    model_results: dict[str, Any],
    active_models: list[str],
    default_weights: dict[str, float],
) -> list[EnsembleSplitResult]:
    """Collect validation metrics and raw predictions from each model's splits.

    Uses real val_predictions/val_actuals from each trainer's SplitResult
    for proper blended-prediction weight optimization.

    Args:
        model_results: Dict of model name -> pipeline result.
        active_models: List of active model names.
        default_weights: Default ensemble weights.

    Returns:
        List of EnsembleSplitResult for each CV split.
    """
    from energy_forecast.training.ensemble_trainer import EnsembleSplitResult

    # Get split count from first model
    first_model = next(iter(model_results.values()))
    n_splits = len(first_model.training_result.split_results)

    # Verify all models have same split count
    for model_name, result in model_results.items():
        if len(result.training_result.split_results) != n_splits:
            msg = f"Split count mismatch for {model_name}"
            raise ValueError(msg)

    split_results: list[EnsembleSplitResult] = []
    for split_idx in range(n_splits):
        model_metrics: dict[str, MetricsResult] = {}
        model_predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
        y_true: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

        for model_name in active_models:
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
                for m in active_models
            )
            dw = default_weights
            am = active_models
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


def optimize_weights(
    split_results: list[EnsembleSplitResult],
    initial_weights: dict[str, float],
    active_models: list[str],
    bounds_config: Any,
) -> dict[str, float]:
    """Optimize ensemble weights via scipy SLSQP.

    Minimizes mean validation MAPE on blended predictions subject to
    ``sum(weights) = 1`` and per-model bounds from config.

    Args:
        split_results: List of split results with model predictions.
        initial_weights: Initial weight values.
        active_models: List of active model names.
        bounds_config: Optimization bounds configuration object.

    Returns:
        Dictionary with optimized weights.
    """
    logger.info("Optimizing ensemble weights for {} models...", len(active_models))

    # Check if we have real predictions for blended optimization
    has_real_predictions = all(
        len(sr.model_predictions) == len(active_models)
        and all(len(p) > 1 for p in sr.model_predictions.values())
        for sr in split_results
    )

    def objective(weights: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
        """Calculate mean validation MAPE on blended predictions."""
        w_dict = {m: weights[i] for i, m in enumerate(active_models)}
        mapes: list[float] = []

        for sr in split_results:
            if has_real_predictions:
                # Correct: MAPE of blended prediction vs actuals
                blended = np.zeros_like(sr.y_true)
                for m in active_models:
                    blended += w_dict[m] * sr.model_predictions[m]
                mapes.append(mape_fn(sr.y_true, blended))
            else:
                # Fallback: weighted average of MAPEs (approximation)
                ensemble_mape = sum(
                    w_dict[m] * sr.model_metrics[m].mape
                    for m in active_models
                )
                mapes.append(ensemble_mape)

        return float(np.mean(mapes))

    # Constraint: sum(weights) = 1
    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

    # Per-model bounds
    bounds: list[tuple[float, float]] = []
    for model_name in active_models:
        model_bounds = getattr(bounds_config, model_name, (0.1, 0.9))
        bounds.append(model_bounds)

    # Initial guess from normalized default weights
    x0 = np.array([initial_weights[m] for m in active_models], dtype=np.float64)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 100},
    )

    optimized_weights = {
        m: float(result.x[i]) for i, m in enumerate(active_models)
    }
    optimized_mape = float(result.fun)

    logger.info(
        "Weight optimization complete: {} | MAPE={:.3f}%",
        {m: f"{w:.3f}" for m, w in optimized_weights.items()},
        optimized_mape,
    )

    return optimized_weights


def compute_weighted_ensemble(
    split_results: list[EnsembleSplitResult],
    weights: dict[str, float],
    active_models: list[str],
) -> list[EnsembleSplitResult]:
    """Recompute ensemble metrics with given weights using real predictions.

    Args:
        split_results: Original split results with real model predictions.
        weights: Optimized weights.
        active_models: List of active model names.

    Returns:
        Updated split results with new ensemble metrics.
    """
    from energy_forecast.training.ensemble_trainer import EnsembleSplitResult

    updated_results: list[EnsembleSplitResult] = []

    for sr in split_results:
        has_preds = (
            len(sr.model_predictions) == len(active_models)
            and all(len(p) > 1 for p in sr.model_predictions.values())
        )

        if has_preds:
            # Compute blended prediction and real metrics
            blended = np.zeros_like(sr.y_true)
            for m in active_models:
                blended += weights[m] * sr.model_predictions[m]
            ensemble_metrics = compute_all(sr.y_true, blended)
            ensemble_predictions = blended
        else:
            # Fallback: metric-level approximation
            ensemble_metrics = MetricsResult(
                mape=sum(weights[m] * sr.model_metrics[m].mape for m in active_models),
                mae=sum(weights[m] * sr.model_metrics[m].mae for m in active_models),
                rmse=sum(weights[m] * sr.model_metrics[m].rmse for m in active_models),
                r2=sum(weights[m] * sr.model_metrics[m].r2 for m in active_models),
                smape=sum(weights[m] * sr.model_metrics[m].smape for m in active_models),
                wmape=sum(weights[m] * sr.model_metrics[m].wmape for m in active_models),
                mbe=sum(weights[m] * sr.model_metrics[m].mbe for m in active_models),
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


def compute_weighted_test_mape(
    model_results: dict[str, Any],
    weights: dict[str, float],
    active_models: list[str],
) -> list[float]:
    """Compute real blended test MAPE across all CV splits.

    Args:
        model_results: Base model results with test_predictions per split.
        weights: Optimized ensemble weights.
        active_models: List of active model names.

    Returns:
        List of test MAPEs, one per CV split.
    """
    first_model = next(iter(model_results.values()))
    n_splits = len(first_model.training_result.split_results)
    test_mapes: list[float] = []

    for split_idx in range(n_splits):
        preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
        y_test: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

        for model_name in active_models:
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
