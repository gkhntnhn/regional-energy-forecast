"""Tests for BaseForecaster."""

from __future__ import annotations

from energy_forecast.models.base import DEFAULT_TARGET_COL, PREDICTION_COL


class TestBaseForecasterConstants:
    """Test module-level constants."""

    def test_default_target_col(self) -> None:
        """Default target column is consumption."""
        assert DEFAULT_TARGET_COL == "consumption"

    def test_prediction_col(self) -> None:
        """Standard prediction column is consumption_mwh."""
        assert PREDICTION_COL == "consumption_mwh"


class TestBaseForecasterTargetCol:
    """Test target_col property via a concrete subclass."""

    def test_target_col_default(self) -> None:
        """target_col defaults to DEFAULT_TARGET_COL."""
        from energy_forecast.models.catboost import CatBoostForecaster

        f = CatBoostForecaster({})
        assert f.target_col == DEFAULT_TARGET_COL

    def test_target_col_custom(self) -> None:
        """target_col can be overridden via config."""
        from energy_forecast.models.catboost import CatBoostForecaster

        f = CatBoostForecaster({"target_col": "custom_col"})
        assert f.target_col == "custom_col"
