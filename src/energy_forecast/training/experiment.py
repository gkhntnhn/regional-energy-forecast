"""MLflow experiment tracker for all models.

Provides a thin wrapper around MLflow with graceful noop when disabled.
All models (CatBoost, Prophet, TFT) use this same tracker.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from energy_forecast.training.metrics import MetricsResult


class ExperimentTracker:
    """MLflow experiment tracker.

    When ``enabled=False`` all logging calls are silent no-ops,
    making it safe to use in tests and development without a running
    MLflow server.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking server URI.
        enabled: Whether to actually log to MLflow.
    """

    def __init__(
        self,
        experiment_name: str = "energy-forecast",
        tracking_uri: str = "http://localhost:5000",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._mlflow: Any = None
        if enabled:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow tracking enabled: {}", tracking_uri)

    @contextmanager
    def start_run(self, run_name: str) -> Iterator[str | None]:
        """Start an MLflow run as a context manager.

        Yields:
            Run ID string, or ``None`` when disabled.
        """
        if not self._enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name) as run:
            yield str(run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not self._enabled:
            return
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a batch of metrics."""
        if not self._enabled:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a model artifact."""
        if not self._enabled:
            return
        self._mlflow.catboost.log_model(model, artifact_path=artifact_path)

    def log_feature_importance(
        self,
        importance: dict[str, float],
        top_n: int = 20,
    ) -> None:
        """Log feature importance as metrics (top N features)."""
        if not self._enabled:
            return
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (name, value) in enumerate(sorted_feats[:top_n]):
            self._mlflow.log_metric(f"feat_importance_{i:02d}_{name}", value)

    def log_split_metrics(
        self,
        split_idx: int,
        train_metrics: MetricsResult,
        val_metrics: MetricsResult,
        test_metrics: MetricsResult,
    ) -> None:
        """Log per-split train/val/test metrics."""
        if not self._enabled:
            return
        prefix_map = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        batch: dict[str, float] = {}
        for prefix, m in prefix_map.items():
            batch[f"split_{split_idx:02d}_{prefix}_mape"] = m.mape
            batch[f"split_{split_idx:02d}_{prefix}_mae"] = m.mae
            batch[f"split_{split_idx:02d}_{prefix}_rmse"] = m.rmse
        self._mlflow.log_metrics(batch)
