"""CLI entry point for model training.

Usage:
    python -m energy_forecast.training.run --model catboost [--n-trials 5] [--data PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energy_forecast.config import Settings, load_config
from energy_forecast.training.catboost_trainer import CatBoostTrainer
from energy_forecast.training.experiment import ExperimentTracker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of arguments (defaults to sys.argv).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train energy forecast models.",
        prog="python -m energy_forecast.training.run",
    )
    parser.add_argument(
        "--model",
        choices=["catboost", "prophet", "tft", "ensemble"],
        required=True,
        help="Model to train.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Path to feature-engineered parquet file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override Optuna trial count.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking.",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="Path to configs directory.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Override active models for ensemble (comma-separated: catboost,prophet,tft).",
    )
    return parser.parse_args(argv)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load feature-engineered data from parquet.

    Args:
        data_path: Path to parquet file.

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not data_path.exists():
        msg = f"Data file not found: {data_path}"
        raise FileNotFoundError(msg)

    df: pd.DataFrame = pd.read_parquet(data_path)
    logger.info("Loaded data: {} rows, {} columns", len(df), len(df.columns))
    return df


def run_catboost(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> None:
    """Run CatBoost training pipeline.

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.
    """
    tracker = ExperimentTracker(
        experiment_name="energy-forecast-catboost",
        tracking_uri=settings.env.mlflow_tracking_uri,
        enabled=not no_mlflow,
    )
    trainer = CatBoostTrainer(settings, tracker)
    result = trainer.run(data)

    logger.info("Best val MAPE: {:.2f}%", result.training_result.avg_val_mape)
    logger.info("Best test MAPE: {:.2f}%", result.training_result.avg_test_mape)
    logger.info("Best params: {}", result.best_params)
    logger.info("Training time: {:.1f}s", result.training_time_seconds)


def run_prophet(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> None:
    """Run Prophet training pipeline.

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.
    """
    from energy_forecast.training.prophet_trainer import ProphetTrainer

    tracker = ExperimentTracker(
        experiment_name="energy-forecast-prophet",
        tracking_uri=settings.env.mlflow_tracking_uri,
        enabled=not no_mlflow,
    )
    trainer = ProphetTrainer(settings, tracker)
    result = trainer.run(data)

    logger.info("Best val MAPE: {:.2f}%", result.training_result.avg_val_mape)
    logger.info("Best test MAPE: {:.2f}%", result.training_result.avg_test_mape)
    logger.info("Best params: {}", result.best_params)
    logger.info("Training time: {:.1f}s", result.training_time_seconds)


def run_tft(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> None:
    """Run TFT training pipeline.

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.
    """
    from energy_forecast.training.tft_trainer import TFTTrainer

    tracker = ExperimentTracker(
        experiment_name="energy-forecast-tft",
        tracking_uri=settings.env.mlflow_tracking_uri,
        enabled=not no_mlflow,
    )
    trainer = TFTTrainer(settings, tracker)
    result = trainer.run(data)

    logger.info("Best val MAPE: {:.2f}%", result.training_result.avg_val_mape)
    logger.info("Best test MAPE: {:.2f}%", result.training_result.avg_test_mape)
    logger.info("Best params: {}", result.best_params)
    logger.info("Training time: {:.1f}s", result.training_time_seconds)


def run_ensemble(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
    active_models_override: list[str] | None = None,
) -> None:
    """Run Ensemble training pipeline (CatBoost + Prophet + TFT).

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.
        active_models_override: Override active models from config.
    """
    from energy_forecast.training.ensemble_trainer import (
        EnsembleTrainer,
        save_ensemble_weights,
    )

    tracker = ExperimentTracker(
        experiment_name="energy-forecast-ensemble",
        tracking_uri=settings.env.mlflow_tracking_uri,
        enabled=not no_mlflow,
    )
    trainer = EnsembleTrainer(
        settings, tracker, active_models_override=active_models_override
    )
    result = trainer.run(data)

    # Save weights
    weights_path = Path("models/ensemble_weights.json")
    save_ensemble_weights(result.training_result.optimized_weights, weights_path)

    logger.info("Ensemble val MAPE: {:.2f}%", result.training_result.avg_val_mape)
    for model_name, mape in result.training_result.model_avg_val_mapes.items():
        logger.info("{} val MAPE: {:.2f}%", model_name.capitalize(), mape)
    logger.info("Optimized weights: {}", result.training_result.optimized_weights)
    logger.info("Training time: {:.1f}s", result.training_time_seconds)


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Optional list of arguments for testing.
    """
    args = parse_args(argv)

    logger.info("Loading config from {}", args.configs)
    settings = load_config(args.configs)

    # Override n_trials if specified (applies to all models for ensemble)
    if args.n_trials is not None:
        catboost_config = settings.hyperparameters.catboost
        prophet_config = settings.hyperparameters.prophet
        tft_config = settings.hyperparameters.tft
        object.__setattr__(catboost_config, "n_trials", args.n_trials)
        object.__setattr__(prophet_config, "n_trials", args.n_trials)
        object.__setattr__(tft_config, "n_trials", args.n_trials)
        logger.info("Overriding n_trials to {}", args.n_trials)

    data = load_data(args.data)

    # Parse --models override for ensemble
    active_models_override: list[str] | None = None
    if args.models:
        active_models_override = [m.strip() for m in args.models.split(",")]
        logger.info("Active models override: {}", active_models_override)

    model_runners: dict[str, Any] = {
        "catboost": lambda: run_catboost(settings, data, no_mlflow=args.no_mlflow),
        "prophet": lambda: run_prophet(settings, data, no_mlflow=args.no_mlflow),
        "tft": lambda: run_tft(settings, data, no_mlflow=args.no_mlflow),
        "ensemble": lambda: run_ensemble(
            settings,
            data,
            no_mlflow=args.no_mlflow,
            active_models_override=active_models_override,
        ),
    }

    runner = model_runners.get(args.model)
    if runner is None:
        logger.error("Unknown model: {}", args.model)
        sys.exit(1)

    runner()


if __name__ == "__main__":
    main()
