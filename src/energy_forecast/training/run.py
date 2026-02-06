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
        choices=["catboost"],
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


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Optional list of arguments for testing.
    """
    args = parse_args(argv)

    logger.info("Loading config from {}", args.configs)
    settings = load_config(args.configs)

    # Override n_trials if specified
    if args.n_trials is not None:
        search_config = settings.hyperparameters.catboost
        object.__setattr__(search_config, "n_trials", args.n_trials)
        logger.info("Overriding n_trials to {}", args.n_trials)

    data = load_data(args.data)

    model_runners: dict[str, Any] = {
        "catboost": lambda: run_catboost(settings, data, no_mlflow=args.no_mlflow),
    }

    runner = model_runners.get(args.model)
    if runner is None:
        logger.error("Unknown model: {}", args.model)
        sys.exit(1)

    runner()


if __name__ == "__main__":
    main()
