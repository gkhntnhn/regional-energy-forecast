"""OOF prediction cache: save/load split results to/from disk.

Caches per-split predictions, actuals, and metrics so that ensemble
training can skip individual model re-training when config is unchanged.

Cache layout::

    models/{model}/oof_cache/
        metadata.json          # config_hash, n_splits, per-split metrics
        split_00_val.npz       # val_predictions + val_actuals
        split_00_test.npz      # test_predictions + test_actuals
        ...
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from energy_forecast.training.metrics import MetricsResult
from energy_forecast.training.results import SplitResult
from energy_forecast.utils import TZ_ISTANBUL

# ---------------------------------------------------------------------------
# Cached result wrappers (duck-type compatible with PipelineResult)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedTrainingResult:
    """Lightweight training result reconstructed from OOF cache."""

    split_results: list[SplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float


@dataclass(frozen=True)
class CachedPipelineResult:
    """Lightweight pipeline result from OOF cache.

    Duck-type compatible with CatBoostPipelineResult / ProphetPipelineResult /
    TFTPipelineResult — only exposes fields that ensemble actually reads.
    """

    training_result: CachedTrainingResult


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_oof_cache(
    model_name: str,
    split_results: list[SplitResult],
    models_dir: str | Path,
    config_hash: str,
) -> Path:
    """Save split results (OOF predictions + metrics) to disk.

    Args:
        model_name: Model identifier (catboost/prophet/tft).
        split_results: List of SplitResult from training.
        models_dir: Base models directory.
        config_hash: SHA256 hash of model config for staleness check.

    Returns:
        Path to the oof_cache directory.
    """
    cache_dir = Path(models_dir) / model_name / "oof_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for sr in split_results:
        idx = f"{sr.split_idx:02d}"
        if sr.val_predictions is not None and sr.val_actuals is not None:
            np.savez_compressed(
                cache_dir / f"split_{idx}_val.npz",
                predictions=sr.val_predictions,
                actuals=sr.val_actuals,
            )
        if sr.test_predictions is not None and sr.test_actuals is not None:
            np.savez_compressed(
                cache_dir / f"split_{idx}_test.npz",
                predictions=sr.test_predictions,
                actuals=sr.test_actuals,
            )

    metadata = {
        "config_hash": config_hash,
        "n_splits": len(split_results),
        "timestamp": datetime.now(tz=TZ_ISTANBUL).isoformat(),
        "splits": [
            {
                "split_idx": sr.split_idx,
                "val_month": sr.val_month,
                "test_month": sr.test_month,
                "best_iteration": sr.best_iteration,
                "train_metrics": _metrics_to_dict(sr.train_metrics),
                "val_metrics": _metrics_to_dict(sr.val_metrics),
                "test_metrics": _metrics_to_dict(sr.test_metrics),
            }
            for sr in split_results
        ],
    }

    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(
        "OOF cache saved: {} ({} splits) -> {}",
        model_name,
        len(split_results),
        cache_dir,
    )
    return cache_dir


def load_oof_cache(
    model_name: str,
    models_dir: str | Path,
    expected_hash: str,
) -> list[SplitResult] | None:
    """Load cached OOF predictions if valid.

    Args:
        model_name: Model identifier.
        models_dir: Base models directory.
        expected_hash: Expected config hash for validation.

    Returns:
        List of SplitResult if cache is valid, None otherwise.
    """
    cache_dir = Path(models_dir) / model_name / "oof_cache"
    metadata_path = cache_dir / "metadata.json"

    if not metadata_path.exists():
        logger.debug("OOF cache not found for {}", model_name)
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("OOF cache metadata corrupt for {}: {}", model_name, e)
        return None

    if metadata.get("config_hash") != expected_hash:
        logger.info(
            "OOF cache stale for {} (hash mismatch)",
            model_name,
        )
        return None

    split_results: list[SplitResult] = []
    for split_meta in metadata["splits"]:
        idx = f"{split_meta['split_idx']:02d}"

        val_path = cache_dir / f"split_{idx}_val.npz"
        test_path = cache_dir / f"split_{idx}_test.npz"

        val_preds: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
        val_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
        test_preds: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
        test_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

        if val_path.exists():
            val_data = np.load(val_path)
            val_preds = val_data["predictions"]
            val_actuals = val_data["actuals"]

        if test_path.exists():
            test_data = np.load(test_path)
            test_preds = test_data["predictions"]
            test_actuals = test_data["actuals"]

        split_results.append(
            SplitResult(
                split_idx=split_meta["split_idx"],
                train_metrics=_dict_to_metrics(split_meta["train_metrics"]),
                val_metrics=_dict_to_metrics(split_meta["val_metrics"]),
                test_metrics=_dict_to_metrics(split_meta["test_metrics"]),
                val_month=split_meta["val_month"],
                test_month=split_meta["test_month"],
                best_iteration=split_meta.get("best_iteration", 0),
                val_predictions=val_preds,
                val_actuals=val_actuals,
                test_predictions=test_preds,
                test_actuals=test_actuals,
            )
        )

    logger.info("OOF cache loaded: {} ({} splits)", model_name, len(split_results))
    return split_results


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


def compute_config_hash(settings: Any, model_name: str) -> str:
    """Compute SHA256 hash of model-relevant config for cache invalidation.

    Includes model-specific config, hyperparameter search space, and CV config.
    Any change to these invalidates the cache automatically.

    Args:
        settings: Full Settings object.
        model_name: Model identifier (catboost/prophet/tft).

    Returns:
        Hex digest string.
    """
    parts: list[dict[str, Any]] = []

    # Model-specific config
    model_config_map: dict[str, Any] = {
        "catboost": settings.catboost,
        "tft": settings.tft,
        "tsmixerx": settings.tsmixerx,
    }
    if model_name in model_config_map:
        parts.append(model_config_map[model_name].model_dump(exclude={"best_params"}))

    # Hyperparameter search space
    hp_map: dict[str, Any] = {
        "catboost": settings.hyperparameters.catboost,
        "tft": settings.hyperparameters.tft,
        "tsmixerx": settings.hyperparameters.tsmixerx,
    }
    if model_name in hp_map:
        parts.append(hp_map[model_name].model_dump())

    # Cross-validation config (shared across all models)
    parts.append(settings.hyperparameters.cross_validation.model_dump())

    # Selected features (CatBoost only)
    if model_name == "catboost":
        features_path = Path("configs/models/catboost_selected_features.json")
        if features_path.exists():
            parts.append({"selected_features_hash": _file_hash(features_path)})

    content = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_to_dict(m: MetricsResult) -> dict[str, float]:
    """Convert MetricsResult to serializable dict."""
    return {
        "mape": m.mape,
        "mae": m.mae,
        "rmse": m.rmse,
        "r2": m.r2,
        "smape": m.smape,
        "wmape": m.wmape,
        "mbe": m.mbe,
    }


def _dict_to_metrics(d: dict[str, float]) -> MetricsResult:
    """Reconstruct MetricsResult from dict."""
    return MetricsResult(
        mape=d["mape"],
        mae=d["mae"],
        rmse=d["rmse"],
        r2=d["r2"],
        smape=d["smape"],
        wmape=d["wmape"],
        mbe=d["mbe"],
    )


def _file_hash(path: Path) -> str:
    """Compute SHA256 of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
