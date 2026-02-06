"""Evaluation metrics for forecast quality assessment."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def mape(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Mean Absolute Percentage Error (primary metric)."""
    raise NotImplementedError


def mae(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Mean Absolute Error."""
    raise NotImplementedError


def rmse(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Root Mean Squared Error."""
    raise NotImplementedError


def r_squared(
    y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]
) -> float:
    """Coefficient of determination (R^2)."""
    raise NotImplementedError


def smape(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    raise NotImplementedError


def wmape(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Weighted Mean Absolute Percentage Error."""
    raise NotImplementedError


def mbe(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Mean Bias Error (over/under-prediction indicator)."""
    raise NotImplementedError
