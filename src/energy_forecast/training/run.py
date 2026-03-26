"""CLI entry point for model training.

Usage:
    python -m energy_forecast.training.run --model catboost [--n-trials 5] [--data PATH]
    python -m energy_forecast.training.run --model catboost --no-mlflow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from energy_forecast.config import SearchParamConfig, Settings, load_config
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
        choices=["catboost", "tft", "tsmixerx", "ensemble"],
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
        help="Override active models for ensemble (comma-separated: catboost,tft,tsmixerx).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force retrain even if OOF cache exists (ensemble only).",
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

    def _apply_model_hp_override(
        model_name: str, model_config: Any, override_dict: dict[str, Any]
    ) -> None:
        """Apply n_trials and search_space overrides to a model's HP config."""
        if "n_trials" in override_dict:
            object.__setattr__(model_config, "n_trials", override_dict["n_trials"])
        if "search_space" in override_dict:
            new_space = {
                k: SearchParamConfig(**v) for k, v in override_dict["search_space"].items()
            }
            object.__setattr__(model_config, "search_space", new_space)
        logger.debug("{} overrides applied", model_name)

    # Override CatBoost / TFT / TSMixerx hyperparameters
    for model_name, model_config in [
        ("catboost", hp.catboost),
        ("tft", hp.tft),
        ("tsmixerx", hp.tsmixerx),
    ]:
        if model_name in overrides:
            _apply_model_hp_override(model_name, model_config, overrides[model_name])

    # TFT-specific: training + optimization param overrides
    if "tft" in overrides:
        tft_override = overrides["tft"]
        for section, cfg in [
            ("training", settings.tft.training),
            ("optimization", settings.tft.optimization),
        ]:
            if section in tft_override:
                for key, val in tft_override[section].items():
                    if hasattr(cfg, key):
                        object.__setattr__(cfg, key, val)
        logger.debug("TFT training/optimization overrides applied")

    # TSMixerx-specific: training + optimization param overrides
    if "tsmixerx" in overrides:
        tsmixerx_override = overrides["tsmixerx"]
        for section, cfg in [
            ("training", settings.tsmixerx.training),
            ("optimization", settings.tsmixerx.optimization),
        ]:
            if section in tsmixerx_override:
                for key, val in tsmixerx_override[section].items():
                    if hasattr(cfg, key):
                        object.__setattr__(cfg, key, val)
        logger.debug("TSMixerx training/optimization overrides applied")

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


def _run_model(
    model_name: str,
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> dict[str, Any]:
    """Run a single-model training pipeline (CatBoost, TFT, or TSMixerx).

    Shared logic for tracker creation, training, logging, and result formatting.

    Args:
        model_name: One of "catboost", "prophet", "tft".
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.

    Returns:
        Dict with metrics and model_path for DB recording.
    """
    # Lazy imports for each trainer
    trainer_factories: dict[str, tuple[str, str]] = {
        "catboost": (
            "energy_forecast.training.catboost_trainer",
            "CatBoostTrainer",
        ),
        "tft": (
            "energy_forecast.training.tft_trainer",
            "TFTTrainer",
        ),
        "tsmixerx": (
            "energy_forecast.training.tsmixerx_trainer",
            "TSMixerxTrainer",
        ),
    }

    import importlib

    module_path, class_name = trainer_factories[model_name]
    module = importlib.import_module(module_path)
    trainer_cls = getattr(module, class_name)

    tracker = ExperimentTracker(
        experiment_name=f"energy-forecast-{model_name}",
        tracking_uri=settings.env.mlflow_tracking_uri,
        enabled=not no_mlflow,
    )
    trainer = trainer_cls(settings, tracker)
    result = trainer.run(data)

    logger.info("Best val MAPE: {:.2f}%", result.training_result.avg_val_mape)
    logger.info("Best test MAPE: {:.2f}%", result.training_result.avg_test_mape)
    logger.info("Best params: {}", result.best_params)
    logger.info("Training time: {:.1f}s", result.training_time_seconds)

    return {
        "metrics": {
            "val_mape": result.training_result.avg_val_mape,
            "test_mape": result.training_result.avg_test_mape,
        },
        "model_path": str(Path(settings.paths.models_dir) / model_name),
        "best_params": result.best_params,
    }


def run_catboost(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> dict[str, Any]:
    """Run CatBoost training pipeline.

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.

    Returns:
        Dict with metrics and model_path for DB recording.
    """
    return _run_model("catboost", settings, data, no_mlflow=no_mlflow)


def run_tsmixerx(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> dict[str, Any]:
    """Run TSMixerx training pipeline.

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.

    Returns:
        Dict with metrics and model_path for DB recording.
    """
    return _run_model("tsmixerx", settings, data, no_mlflow=no_mlflow)


def run_tft(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
) -> dict[str, Any]:
    """Run TFT training pipeline.

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.

    Returns:
        Dict with metrics and model_path for DB recording.
    """
    return _run_model("tft", settings, data, no_mlflow=no_mlflow)


def run_ensemble(
    settings: Settings,
    data: pd.DataFrame,
    *,
    no_mlflow: bool = False,
    active_models_override: list[str] | None = None,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Run Ensemble training pipeline (CatBoost + TFT + TSMixerx).

    Args:
        settings: Full application settings.
        data: Feature-engineered DataFrame.
        no_mlflow: If True, disable MLflow tracking.
        active_models_override: Override active models from config.
        no_cache: If True, force retrain even if OOF cache exists.

    Returns:
        Dict with metrics and model_path for DB recording.
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
        settings, tracker, active_models_override=active_models_override, no_cache=no_cache
    )
    result = trainer.run(data)

    # Save ensemble artifacts to fixed directory (overwrite previous)
    ensemble_dir = Path(settings.paths.models_dir) / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Save weights/config
    weights_path = ensemble_dir / "ensemble_weights.json"
    save_ensemble_weights(result.training_result.optimized_weights, weights_path)

    # Save meta-learner if stacking mode
    if result.meta_model is not None:
        meta_path = ensemble_dir / "meta_model.cbm"
        result.meta_model.save_model(str(meta_path))
        logger.info("Saved meta-learner to {}", meta_path)

    mode = result.training_result.mode
    logger.info("Ensemble mode: {}", mode)
    logger.info("Ensemble val MAPE: {:.2f}%", result.training_result.avg_val_mape)
    logger.info("Ensemble test MAPE: {:.2f}%", result.training_result.avg_test_mape)
    for model_name, mape_val in result.training_result.model_avg_val_mapes.items():
        logger.info("{} val MAPE: {:.2f}%", model_name.capitalize(), mape_val)
    if mode == "weighted_average":
        logger.info("Optimized weights: {}", result.training_result.optimized_weights)
    logger.info("Training time: {:.1f}s", result.training_time_seconds)

    return {
        "metrics": {
            "val_mape": result.training_result.avg_val_mape,
            "test_mape": result.training_result.avg_test_mape,
        },
        "model_path": str(ensemble_dir),
        "best_params": result.training_result.optimized_weights,
    }


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Optional list of arguments for testing.
    """
    from dotenv import load_dotenv

    load_dotenv()

    args = parse_args(argv)

    logger.info("Loading config from {}", args.configs)
    settings = load_config(args.configs)

    # Apply override config if specified
    if args.config is not None:
        apply_config_overrides(settings, args.config)

    # Override n_trials if specified (applies to all models for ensemble)
    if args.n_trials is not None:
        catboost_config = settings.hyperparameters.catboost
        tft_config = settings.hyperparameters.tft
        tsmixerx_config = settings.hyperparameters.tsmixerx
        object.__setattr__(catboost_config, "n_trials", args.n_trials)
        object.__setattr__(tft_config, "n_trials", args.n_trials)
        object.__setattr__(tsmixerx_config, "n_trials", args.n_trials)
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
        "tft": lambda: run_tft(settings, data, no_mlflow=args.no_mlflow),
        "tsmixerx": lambda: run_tsmixerx(settings, data, no_mlflow=args.no_mlflow),
        "ensemble": lambda: run_ensemble(
            settings,
            data,
            no_mlflow=args.no_mlflow,
            active_models_override=active_models_override,
            no_cache=args.no_cache,
        ),
    }

    runner = model_runners.get(args.model)
    if runner is None:
        logger.error("Unknown model: {}", args.model)
        sys.exit(1)

    if args.model in ("tft", "tsmixerx", "ensemble"):
        from energy_forecast.utils.logging import suppress_training_noise

        suppress_training_noise()

    # DB recording (non-fatal)
    model_run = _start_model_run(
        args.model,
        n_trials=args.n_trials or _get_n_trials(settings, args.model),
        n_splits=settings.hyperparameters.cross_validation.n_splits,
        feature_count=len(data.columns),
    )

    t0 = time.monotonic()
    try:
        run_result = runner()
        duration = int(time.monotonic() - t0)
        _complete_model_run(
            model_run,
            duration=duration,
            metrics=run_result.get("metrics", {}),
            model_path=run_result.get("model_path", ""),
            hyperparams=run_result.get("best_params"),
        )
    except Exception:
        duration = int(time.monotonic() - t0)
        _fail_model_run(model_run, duration=duration)
        raise


def _get_n_trials(settings: Settings, model: str) -> int:
    """Get n_trials for a model from settings."""
    trial_map = {
        "catboost": settings.hyperparameters.catboost.n_trials,
        "tft": settings.hyperparameters.tft.n_trials,
        "tsmixerx": settings.hyperparameters.tsmixerx.n_trials,
        "ensemble": 0,
    }
    return trial_map.get(model, 0)


def _get_db_session() -> tuple[Any, Any] | tuple[None, None]:
    """Get a sync DB session factory and engine, or (None, None) if DB not configured."""
    db_url = os.environ.get("DATABASE_URL_SYNC", "")
    if not db_url:
        return None, None
    from energy_forecast.db.engine import (
        create_sync_engine,
        create_sync_session_factory,
    )

    engine = create_sync_engine(db_url)
    factory = create_sync_session_factory(engine)
    return engine, factory


def _start_model_run(
    model_type: str,
    *,
    n_trials: int,
    n_splits: int,
    feature_count: int,
) -> int | None:
    """Record training start in DB (non-fatal, returns run ID or None)."""
    engine, factory = _get_db_session()
    if engine is None or factory is None:
        return None
    try:
        from energy_forecast.db.repositories.model_repo import ModelRunRepository

        with factory() as session:
            repo = ModelRunRepository(session)
            run = repo.create_run(
                model_type,
                n_trials=n_trials,
                n_splits=n_splits,
                feature_count=feature_count,
            )
            session.commit()
            run_id: int = run.id
            logger.info("Model run #{} started ({})", run_id, model_type)
        return run_id
    except Exception as e:
        logger.warning("Failed to record model run start (non-fatal): {}", e)
        return None
    finally:
        engine.dispose()


def _complete_model_run(
    run_id: int | None,
    *,
    duration: int,
    metrics: dict[str, float] | None = None,
    model_path: str = "",
    hyperparams: dict[str, Any] | None = None,
) -> None:
    """Record training completion in DB (non-fatal)."""
    if run_id is None:
        return
    engine, factory = _get_db_session()
    if engine is None or factory is None:
        return
    try:
        from energy_forecast.db.repositories.model_repo import ModelRunRepository

        with factory() as session:
            repo = ModelRunRepository(session)
            repo.complete_run(
                run_id,
                metrics=metrics or {},
                model_path=model_path,
                hyperparams=hyperparams,
                duration_seconds=duration,
            )
            session.commit()
            logger.info("Model run #{} completed ({}s)", run_id, duration)
    except Exception as e:
        logger.warning("Failed to record model run completion (non-fatal): {}", e)
    finally:
        engine.dispose()


def _fail_model_run(run_id: int | None, *, duration: int) -> None:
    """Record training failure in DB (non-fatal)."""
    if run_id is None:
        return
    engine, factory = _get_db_session()
    if engine is None or factory is None:
        return
    try:
        import traceback

        from energy_forecast.db.repositories.model_repo import ModelRunRepository

        with factory() as session:
            repo = ModelRunRepository(session)
            repo.fail_run(run_id, traceback.format_exc())
            session.commit()
            logger.info("Model run #{} failed ({}s)", run_id, duration)
    except Exception as e:
        logger.warning("Failed to record model run failure (non-fatal): {}", e)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
