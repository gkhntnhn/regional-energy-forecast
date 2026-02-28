"""CLI entry point for model training.

Usage:
    python -m energy_forecast.training.run --model catboost [--n-trials 5] [--data PATH]
    python -m energy_forecast.training.run --model catboost --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from energy_forecast.config import Settings, load_config
from energy_forecast.config.settings import SearchParamConfig
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
        default=None,
        help="Path to feature-engineered parquet file. Defaults to config path.",
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
        "--config",
        type=Path,
        default=None,
        help="Path to override config YAML.",
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


def apply_config_overrides(settings: Settings, config_path: Path) -> None:
    """Apply override config to settings.

    Modifies settings in-place by overriding hyperparameters and CV config.

    Args:
        settings: Settings object to modify.
        config_path: Path to override YAML file.
    """
    if not config_path.exists():
        msg = f"Override config not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        overrides = yaml.safe_load(f)

    logger.info("Applying config overrides from {}", config_path)

    hp = settings.hyperparameters

    # Override CatBoost
    if "catboost" in overrides:
        cb_override = overrides["catboost"]
        cb_config = hp.catboost
        if "n_trials" in cb_override:
            object.__setattr__(cb_config, "n_trials", cb_override["n_trials"])
        if "search_space" in cb_override:
            new_space = {
                k: SearchParamConfig(**v) for k, v in cb_override["search_space"].items()
            }
            object.__setattr__(cb_config, "search_space", new_space)
        logger.debug("CatBoost overrides applied")

    # Override Prophet
    if "prophet" in overrides:
        p_override = overrides["prophet"]
        p_config = hp.prophet
        if "n_trials" in p_override:
            object.__setattr__(p_config, "n_trials", p_override["n_trials"])
        if "search_space" in p_override:
            new_space = {
                k: SearchParamConfig(**v) for k, v in p_override["search_space"].items()
            }
            object.__setattr__(p_config, "search_space", new_space)
        logger.debug("Prophet overrides applied")

    # Override TFT
    if "tft" in overrides:
        tft_override = overrides["tft"]
        tft_config = hp.tft
        if "n_trials" in tft_override:
            object.__setattr__(tft_config, "n_trials", tft_override["n_trials"])
        if "search_space" in tft_override:
            new_space = {
                k: SearchParamConfig(**v) for k, v in tft_override["search_space"].items()
            }
            object.__setattr__(tft_config, "search_space", new_space)
        # Override TFT training params
        if "training" in tft_override:
            train_ovr = tft_override["training"]
            train_cfg = settings.tft.training
            for key, val in train_ovr.items():
                if hasattr(train_cfg, key):
                    object.__setattr__(train_cfg, key, val)
        # Override TFT optimization params (e.g., fast_epochs for Optuna trials)
        if "optimization" in tft_override:
            opt_ovr = tft_override["optimization"]
            opt_cfg = settings.tft.optimization
            for key, val in opt_ovr.items():
                if hasattr(opt_cfg, key):
                    object.__setattr__(opt_cfg, key, val)
        logger.debug("TFT overrides applied")

    # Override cross-validation
    if "cross_validation" in overrides:
        cv_override = overrides["cross_validation"]
        cv_config = hp.cross_validation
        for key, val in cv_override.items():
            if hasattr(cv_config, key):
                object.__setattr__(cv_config, key, val)
        logger.debug("Cross-validation overrides applied")

    # Override validation settings
    if "validation" in overrides:
        val_override = overrides["validation"]
        if val_override.get("skip_after_optuna", False):
            object.__setattr__(hp, "skip_validation_after_optuna", True)
            logger.debug("Post-Optuna validation skip enabled")


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
    weights_path = Path(settings.paths.ensemble_weights)
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

    # Apply override config if specified
    if args.config is not None:
        apply_config_overrides(settings, args.config)

    # Override n_trials if specified (applies to all models for ensemble)
    if args.n_trials is not None:
        catboost_config = settings.hyperparameters.catboost
        prophet_config = settings.hyperparameters.prophet
        tft_config = settings.hyperparameters.tft
        object.__setattr__(catboost_config, "n_trials", args.n_trials)
        object.__setattr__(prophet_config, "n_trials", args.n_trials)
        object.__setattr__(tft_config, "n_trials", args.n_trials)
        logger.info("Overriding n_trials to {}", args.n_trials)

    # Use CLI data path or config default
    data_path = args.data or Path(settings.paths.features_data)
    data = load_data(data_path)

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
