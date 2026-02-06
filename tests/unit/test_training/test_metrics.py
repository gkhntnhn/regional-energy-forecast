"""Tests for evaluation metrics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from energy_forecast.training.metrics import (
    MetricsResult,
    compute_all,
    mae,
    mape,
    mbe,
    r_squared,
    rmse,
    smape,
    wmape,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(*values: float) -> npt.NDArray[np.floating[Any]]:
    """Shortcut to create float64 arrays."""
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------


class TestMape:
    """Tests for Mean Absolute Percentage Error."""

    def test_mape_perfect(self) -> None:
        """Identical arrays produce MAPE = 0."""
        y = _arr(100.0, 200.0, 300.0)
        assert mape(y, y) == pytest.approx(0.0)

    def test_mape_known(self) -> None:
        """Known calculation: |10/100| + |20/200| = 0.1 + 0.1 → mean 0.1 → 10%."""
        y_true = _arr(100.0, 200.0)
        y_pred = _arr(110.0, 220.0)
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_zero_actual(self) -> None:
        """Zero actuals are excluded from calculation."""
        y_true = _arr(0.0, 100.0, 200.0)
        y_pred = _arr(50.0, 110.0, 220.0)
        # Only non-zero actuals: |10/100| + |20/200| = 0.1 + 0.1 → mean 0.1 → 10%
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_all_zeros(self) -> None:
        """All-zero actuals return 0.0."""
        y_true = _arr(0.0, 0.0)
        y_pred = _arr(1.0, 2.0)
        assert mape(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


class TestMae:
    """Tests for Mean Absolute Error."""

    def test_mae_known(self) -> None:
        """Known calculation: (10 + 10 + 10) / 3 = 10."""
        y_true = _arr(100.0, 200.0, 300.0)
        y_pred = _arr(110.0, 190.0, 310.0)
        assert mae(y_true, y_pred) == pytest.approx(10.0)

    def test_mae_perfect(self) -> None:
        """Identical arrays produce MAE = 0."""
        y = _arr(1.0, 2.0, 3.0)
        assert mae(y, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


class TestRmse:
    """Tests for Root Mean Squared Error."""

    def test_rmse_known(self) -> None:
        """RMSE >= MAE for non-uniform errors."""
        y_true = _arr(100.0, 200.0, 300.0)
        y_pred = _arr(110.0, 190.0, 310.0)
        result = rmse(y_true, y_pred)
        mae_val = mae(y_true, y_pred)
        assert result >= mae_val
        # sqrt(mean([100, 100, 100])) = sqrt(100) = 10.0
        assert result == pytest.approx(10.0)

    def test_rmse_perfect(self) -> None:
        """Identical arrays produce RMSE = 0."""
        y = _arr(5.0, 10.0, 15.0)
        assert rmse(y, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# R-squared
# ---------------------------------------------------------------------------


class TestRSquared:
    """Tests for coefficient of determination."""

    def test_r_squared_perfect(self) -> None:
        """Identical arrays produce R2 = 1.0."""
        y = _arr(100.0, 200.0, 300.0, 400.0)
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_r_squared_poor(self) -> None:
        """Constant prediction (mean) produces R2 ~ 0."""
        y_true = _arr(100.0, 200.0, 300.0, 400.0)
        y_pred = np.full_like(y_true, np.mean(y_true))
        assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_r_squared_constant_true(self) -> None:
        """Constant y_true (ss_tot=0) returns 0.0."""
        y_true = _arr(5.0, 5.0, 5.0)
        y_pred = _arr(4.0, 5.0, 6.0)
        assert r_squared(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SMAPE
# ---------------------------------------------------------------------------


class TestSmape:
    """Tests for Symmetric MAPE."""

    def test_smape_symmetric(self) -> None:
        """SMAPE(a, b) == SMAPE(b, a)."""
        a = _arr(100.0, 200.0, 300.0)
        b = _arr(120.0, 180.0, 330.0)
        assert smape(a, b) == pytest.approx(smape(b, a))

    def test_smape_known(self) -> None:
        """Known calculation for simple case."""
        y_true = _arr(100.0, 200.0)
        y_pred = _arr(110.0, 180.0)
        # point1: 2*10 / (100+110) = 20/210
        # point2: 2*20 / (200+180) = 40/380
        expected = ((20.0 / 210.0 + 40.0 / 380.0) / 2.0) * 100.0
        assert smape(y_true, y_pred) == pytest.approx(expected)

    def test_smape_both_zero(self) -> None:
        """Both zero produces 0.0."""
        y_true = _arr(0.0, 0.0)
        y_pred = _arr(0.0, 0.0)
        assert smape(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WMAPE
# ---------------------------------------------------------------------------


class TestWmape:
    """Tests for Weighted MAPE."""

    def test_wmape_known(self) -> None:
        """Known calculation: sum(|errors|) / sum(|actuals|) * 100."""
        y_true = _arr(100.0, 200.0, 300.0)
        y_pred = _arr(110.0, 190.0, 310.0)
        # errors: 10, 10, 10 → sum = 30
        # actuals sum: 600
        # WMAPE = 30/600 * 100 = 5.0%
        assert wmape(y_true, y_pred) == pytest.approx(5.0)

    def test_wmape_zero_actuals(self) -> None:
        """All-zero actuals return 0.0."""
        y_true = _arr(0.0, 0.0)
        y_pred = _arr(1.0, 2.0)
        assert wmape(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MBE
# ---------------------------------------------------------------------------


class TestMbe:
    """Tests for Mean Bias Error."""

    def test_mbe_overpredict(self) -> None:
        """Over-prediction yields positive MBE."""
        y_true = _arr(100.0, 200.0, 300.0)
        y_pred = _arr(110.0, 220.0, 330.0)
        assert mbe(y_true, y_pred) > 0

    def test_mbe_underpredict(self) -> None:
        """Under-prediction yields negative MBE."""
        y_true = _arr(100.0, 200.0, 300.0)
        y_pred = _arr(90.0, 180.0, 270.0)
        assert mbe(y_true, y_pred) < 0

    def test_mbe_zero(self) -> None:
        """Identical arrays produce MBE = 0."""
        y = _arr(100.0, 200.0, 300.0)
        assert mbe(y, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


class TestComputeAll:
    """Tests for compute_all bundle function."""

    def test_compute_all_returns_all(self) -> None:
        """compute_all returns MetricsResult with all 7 fields."""
        y_true = _arr(100.0, 200.0, 300.0, 400.0)
        y_pred = _arr(110.0, 190.0, 310.0, 390.0)
        result = compute_all(y_true, y_pred)

        assert isinstance(result, MetricsResult)
        assert result.mape == pytest.approx(mape(y_true, y_pred))
        assert result.mae == pytest.approx(mae(y_true, y_pred))
        assert result.rmse == pytest.approx(rmse(y_true, y_pred))
        assert result.r2 == pytest.approx(r_squared(y_true, y_pred))
        assert result.smape == pytest.approx(smape(y_true, y_pred))
        assert result.wmape == pytest.approx(wmape(y_true, y_pred))
        assert result.mbe == pytest.approx(mbe(y_true, y_pred))

    def test_compute_all_frozen(self) -> None:
        """MetricsResult is frozen (immutable)."""
        y = _arr(1.0, 2.0, 3.0)
        result = compute_all(y, y)
        with pytest.raises(AttributeError):
            result.mape = 999.0  # type: ignore[misc]
