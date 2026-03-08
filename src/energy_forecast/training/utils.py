"""Shared training utilities.

Provides common helper functions used across CatBoost, Prophet, and TFT
training pipelines (e.g. Optuna storage selection).
"""

from __future__ import annotations

import os
from pathlib import Path

import optuna


def optuna_storage(
    n_trials: int,
    model_name: str,
    models_dir: str | Path,
) -> optuna.storages.RDBStorage | str | None:
    """Return Optuna storage backend: PostgreSQL if available, else SQLite.

    Returns None for very short runs (n_trials <= 3) to avoid persistence
    overhead during quick smoke tests.

    Args:
        n_trials: Number of Optuna trials planned.
        model_name: Model name used for SQLite file naming.
        models_dir: Base directory for model artifacts.
    """
    if n_trials <= 3:
        return None
    db_url = os.environ.get("DATABASE_URL_SYNC", "")
    if db_url:
        return optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={"pool_size": 1, "max_overflow": 0},
        )
    studies_dir = Path(models_dir) / "optuna_studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{studies_dir / model_name}.db"
