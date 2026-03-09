"""MLflow experiment tracker for all models.

Provides a thin wrapper around MLflow with graceful noop when disabled.
All models (CatBoost, Prophet, TFT) use this same tracker.
"""

from __future__ import annotations

import io
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from loguru import logger

# MLflow prints emoji (🏃) on run start/end; Windows cp1254 can't encode it.
# Reconfigure stdout/stderr to utf-8 with replace error handler.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if isinstance(_stream, io.TextIOWrapper) and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

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
        """Log a CatBoost model artifact via native MLflow CatBoost flavor."""
        if not self._enabled:
            return
        self._mlflow.catboost.log_model(model, name=artifact_path)

    def log_prophet_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a Prophet model artifact using pickle.

        Args:
            model: Prophet model instance.
            artifact_path: Path within the run's artifacts.
        """
        if not self._enabled:
            return

        import pickle
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "prophet_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            self._mlflow.log_artifact(str(model_path), artifact_path)

    def log_tft_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a TFT model artifact using PyTorch checkpoint.

        Saves the TFT model's checkpoint files and metadata to MLflow artifacts.

        Args:
            model: TFTForecaster instance.
            artifact_path: Path within the run's artifacts.
        """
        if not self._enabled:
            return

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            model.save(model_dir)
            self._mlflow.log_artifacts(str(model_dir), artifact_path)

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
            batch[f"split_{split_idx:02d}_{prefix}_r2"] = m.r2
            batch[f"split_{split_idx:02d}_{prefix}_smape"] = m.smape
            batch[f"split_{split_idx:02d}_{prefix}_wmape"] = m.wmape
            batch[f"split_{split_idx:02d}_{prefix}_mbe"] = m.mbe
        self._mlflow.log_metrics(batch)

    def log_ensemble_weights(
        self,
        weights: dict[str, float],
    ) -> None:
        """Log ensemble model weights.

        Args:
            weights: Dictionary with model names as keys and weights as values.
        """
        if not self._enabled:
            return
        for model_name, weight in weights.items():
            self._mlflow.log_metric(f"ensemble_weight_{model_name}", weight)

    def log_training_meta(self, meta: dict[str, Any]) -> None:
        """Log training metadata (data shape, timing, environment).

        Numeric values in ``meta`` are logged as metrics, the rest as params.

        Args:
            meta: Dictionary with keys like ``data_rows``, ``data_cols``,
                ``training_time_seconds``, ``n_splits``, ``n_trials``,
                ``best_trial_number``, ``python_version``, ``platform``.
        """
        if not self._enabled:
            return
        numeric_keys = {"training_time_seconds"}
        params: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        for k, v in meta.items():
            if k in numeric_keys:
                metrics[k] = float(v)
            else:
                params[k] = v
        if params:
            self._mlflow.log_params(params)
        if metrics:
            self._mlflow.log_metrics(metrics)

    def log_config_snapshot(
        self, config_dict: dict[str, Any], filename: str = "config.yaml",
    ) -> None:
        """Log full config as a YAML artifact for reproducibility.

        Args:
            config_dict: Model config dictionary (e.g. from ``model_dump()``).
            filename: Artifact filename.
        """
        if not self._enabled:
            return
        import tempfile
        from pathlib import Path

        import yaml

        yaml_str = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / filename
            fpath.write_text(yaml_str, encoding="utf-8")
            self._mlflow.log_artifact(str(fpath))

    def log_predictions_summary(
        self,
        y_true: np.ndarray[Any, np.dtype[np.floating[Any]]],
        y_pred: np.ndarray[Any, np.dtype[np.floating[Any]]],
        prefix: str = "final",
    ) -> None:
        """Log prediction distribution statistics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.
            prefix: Metric name prefix.
        """
        if not self._enabled:
            return
        residuals = y_pred - y_true
        self._mlflow.log_metrics(
            {
                f"{prefix}_pred_mean": float(np.mean(y_pred)),
                f"{prefix}_pred_std": float(np.std(y_pred)),
                f"{prefix}_pred_min": float(np.min(y_pred)),
                f"{prefix}_pred_max": float(np.max(y_pred)),
                f"{prefix}_actual_mean": float(np.mean(y_true)),
                f"{prefix}_actual_std": float(np.std(y_true)),
                f"{prefix}_residual_mean": float(np.mean(residuals)),
                f"{prefix}_residual_std": float(np.std(residuals)),
                f"{prefix}_max_abs_residual": float(np.max(np.abs(residuals))),
            }
        )

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an MLflow artifact.

        Args:
            local_path: Path to the file on disk.
            artifact_path: Optional subdirectory within the run's artifacts.
        """
        if not self._enabled:
            return
        self._mlflow.log_artifact(local_path, artifact_path)

