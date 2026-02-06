"""Evaluation metrics for forecast quality assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


def mape(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Mean Absolute Percentage Error (%).

    Zero actuals are excluded to avoid division by zero.
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(
    y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]
) -> float:
    """Coefficient of determination (R^2)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def smape(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wmape(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Weighted Mean Absolute Percentage Error (%)."""
    total = float(np.sum(np.abs(y_true)))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def mbe(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Mean Bias Error (positive = over-prediction)."""
    return float(np.mean(y_pred - y_true))


@dataclass(frozen=True)
class MetricsResult:
    """All evaluation metrics bundled together."""

    mape: float
    mae: float
    rmse: float
    r2: float
    smape: float
    wmape: float
    mbe: float


def compute_all(
    y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]
) -> MetricsResult:
    """Compute all metrics at once."""
    return MetricsResult(
        mape=mape(y_true, y_pred),
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        r2=r_squared(y_true, y_pred),
        smape=smape(y_true, y_pred),
        wmape=wmape(y_true, y_pred),
        mbe=mbe(y_true, y_pred),
    )
