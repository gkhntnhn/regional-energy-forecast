"""Ensemble training report generation.

Extracted from ``ensemble_trainer.py``. Builds comparison DataFrames
and prints formatted summaries to the terminal via loguru.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from energy_forecast.config import EnsembleConfig
    from energy_forecast.training.ensemble_trainer import (
        EnsembleTrainingResult,
        ModelResult,
    )


def generate_comparison_df(
    model_results: dict[str, ModelResult],
    training_result: EnsembleTrainingResult,
    active_models: list[str],
) -> pd.DataFrame:
    """Generate comparison DataFrame for all models vs ensemble.

    Args:
        model_results: Dict of model pipeline results.
        training_result: Ensemble training result.
        active_models: List of active model names.

    Returns:
        DataFrame with comparison metrics.
    """
    data: list[dict[str, Any]] = []

    # Individual models
    for model_name in active_models:
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


def print_summary(
    comparison_df: pd.DataFrame,
    training_result: EnsembleTrainingResult,
    active_models: list[str],
    ensemble_config: EnsembleConfig,
) -> None:
    """Print formatted comparison summary to terminal.

    Args:
        comparison_df: Comparison DataFrame.
        training_result: Ensemble training result.
        active_models: List of active model names.
        ensemble_config: Ensemble configuration (for meta-learner depth).
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
        ml_depth = ensemble_config.stacking.meta_learner.depth
        logger.info("Meta-learner: CatBoost depth={}", ml_depth)
    logger.info("Active Models: {}", ", ".join(active_models))
    logger.info("=" * 75)
    logger.info("")
