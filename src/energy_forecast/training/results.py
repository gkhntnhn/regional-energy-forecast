"""Shared training result dataclasses.

Provides the common SplitResult used by CatBoost, Prophet, and TFT trainers,
eliminating triple-duplication of nearly identical frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from energy_forecast.training.metrics import MetricsResult


@dataclass(frozen=True)
class SplitResult:
    """Result from a single CV split.

    Used by all three trainers. CatBoost sets ``best_iteration`` to the actual
    early-stopping iteration; Prophet and TFT leave it at the default (0).
    """

    split_idx: int
    train_metrics: MetricsResult
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    val_month: str
    test_month: str
    best_iteration: int = 0
    val_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    val_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
