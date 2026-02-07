"""Tests for the Ensemble training pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import Settings, get_default_config
from energy_forecast.training.ensemble_trainer import (
    EnsembleSplitResult,
    EnsembleTrainer,
    EnsembleTrainingResult,
    load_ensemble_weights,
    save_ensemble_weights,
)
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(n_months: int = 18) -> pd.DataFrame:
    """Create a minimal feature-engineered DataFrame.

    Generates n_months of hourly data with consumption target + numeric features.
    """
    rng = np.random.default_rng(42)
    n_rows = n_months * 30 * 24
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="h")

    data: dict[str, Any] = {
        "consumption": 800.0 + rng.random(n_rows) * 400,
        "temperature_2m": rng.uniform(-5, 35, n_rows),
        "hour": idx.hour.astype(float),
        "day_of_week": idx.dayofweek.astype(float),
        "month": idx.month.astype(float),
        "is_holiday": rng.choice([0.0, 1.0], n_rows, p=[0.95, 0.05]),
        "is_weekend": (idx.dayofweek >= 5).astype(float),
    }
    return pd.DataFrame(data, index=idx).rename_axis("datetime")


def _get_test_settings() -> Settings:
    """Get settings configured for fast tests."""
    settings = get_default_config()
    # Override for fast tests using object.__setattr__ on frozen models
    cv_config = settings.hyperparameters.cross_validation
    object.__setattr__(cv_config, "n_splits", 3)

    catboost_config = settings.hyperparameters.catboost
    object.__setattr__(catboost_config, "n_trials", 1)
    object.__setattr__(catboost_config, "search_space", {})

    prophet_config = settings.hyperparameters.prophet
    object.__setattr__(prophet_config, "n_trials", 1)
    object.__setattr__(prophet_config, "search_space", {})

    return settings


def _make_mock_metrics() -> MetricsResult:
    """Create mock metrics for testing."""
    return MetricsResult(
        mape=5.0,
        mae=50.0,
        rmse=75.0,
        r2=0.95,
        smape=4.8,
        wmape=5.1,
        mbe=2.0,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestEnsembleConfig:
    """Tests for ensemble configuration loading."""

    def test_default_weights_sum_to_one(self) -> None:
        settings = get_default_config()
        weights = settings.ensemble.weights
        total = weights.catboost + weights.prophet
        assert abs(total - 1.0) < 1e-6

    def test_default_catboost_weight(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.weights.catboost == 0.6

    def test_default_prophet_weight(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.weights.prophet == 0.4

    def test_optimization_enabled_by_default(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.optimization.enabled is True

    def test_fallback_enabled_by_default(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.fallback.enabled is True


# ---------------------------------------------------------------------------
# Trainer initialization
# ---------------------------------------------------------------------------


class TestEnsembleTrainerInit:
    """Tests for EnsembleTrainer initialization."""

    def test_trainer_initializes(self) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        trainer = EnsembleTrainer(settings, tracker)
        assert trainer is not None

    def test_trainer_creates_sub_trainers(self) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        trainer = EnsembleTrainer(settings, tracker)
        assert trainer._catboost_trainer is not None
        assert trainer._prophet_trainer is not None


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------


class TestWeightOptimization:
    """Tests for ensemble weight optimization."""

    def test_optimize_weights_returns_valid_weights(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(settings, ExperimentTracker(enabled=False))

        # Create mock split results
        split_results = [
            EnsembleSplitResult(
                split_idx=i,
                catboost_metrics=MetricsResult(
                    mape=5.0 + i * 0.5,
                    mae=50.0,
                    rmse=75.0,
                    r2=0.95,
                    smape=4.8,
                    wmape=5.1,
                    mbe=2.0,
                ),
                prophet_metrics=MetricsResult(
                    mape=7.0 + i * 0.3,
                    mae=60.0,
                    rmse=85.0,
                    r2=0.92,
                    smape=6.5,
                    wmape=6.8,
                    mbe=3.0,
                ),
                ensemble_metrics=_make_mock_metrics(),
                catboost_predictions=np.zeros(100, dtype=np.float64),
                prophet_predictions=np.zeros(100, dtype=np.float64),
                ensemble_predictions=np.zeros(100, dtype=np.float64),
                y_true=np.ones(100, dtype=np.float64),
                prophet_weight=0.4,
            )
            for i in range(3)
        ]

        weights = trainer._optimize_weights(split_results)

        assert "catboost" in weights
        assert "prophet" in weights
        assert abs(weights["catboost"] + weights["prophet"] - 1.0) < 1e-6
        assert 0.0 <= weights["catboost"] <= 1.0
        assert 0.0 <= weights["prophet"] <= 1.0

    def test_optimize_weights_respects_bounds(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(settings, ExperimentTracker(enabled=False))

        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                catboost_metrics=MetricsResult(
                    mape=5.0, mae=50.0, rmse=75.0, r2=0.95, smape=4.8, wmape=5.1, mbe=2.0
                ),
                prophet_metrics=MetricsResult(
                    mape=10.0, mae=80.0, rmse=100.0, r2=0.85, smape=9.0, wmape=9.5, mbe=5.0
                ),
                ensemble_metrics=_make_mock_metrics(),
                catboost_predictions=np.zeros(100, dtype=np.float64),
                prophet_predictions=np.zeros(100, dtype=np.float64),
                ensemble_predictions=np.zeros(100, dtype=np.float64),
                y_true=np.ones(100, dtype=np.float64),
                prophet_weight=0.4,
            )
        ]

        weights = trainer._optimize_weights(split_results)
        opt_config = settings.ensemble.optimization

        # Prophet weight should be within bounds
        assert weights["prophet"] >= opt_config.prophet_weight_min
        assert weights["prophet"] <= opt_config.prophet_weight_max


# ---------------------------------------------------------------------------
# Comparison DataFrame
# ---------------------------------------------------------------------------


class TestComparisonDF:
    """Tests for comparison DataFrame generation."""

    def test_comparison_df_has_correct_columns(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(settings, ExperimentTracker(enabled=False))

        # Mock results
        catboost_result = MagicMock()
        catboost_result.training_result.avg_val_mape = 5.0
        catboost_result.training_result.avg_test_mape = 5.5

        prophet_result = MagicMock()
        prophet_result.training_result.avg_val_mape = 7.0
        prophet_result.training_result.avg_test_mape = 7.5

        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.8,
            std_val_mape=0.5,
            catboost_avg_val_mape=5.0,
            prophet_avg_val_mape=7.0,
            optimized_weights={"catboost": 0.7, "prophet": 0.3},
        )

        df = trainer._generate_comparison_df(catboost_result, prophet_result, training_result)

        assert "Model" in df.columns
        assert "Val MAPE (%)" in df.columns
        assert "Test MAPE (%)" in df.columns
        assert "Improvement" in df.columns

    def test_comparison_df_has_three_rows(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(settings, ExperimentTracker(enabled=False))

        catboost_result = MagicMock()
        catboost_result.training_result.avg_val_mape = 5.0
        catboost_result.training_result.avg_test_mape = 5.5

        prophet_result = MagicMock()
        prophet_result.training_result.avg_val_mape = 7.0
        prophet_result.training_result.avg_test_mape = 7.5

        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.8,
            std_val_mape=0.5,
            catboost_avg_val_mape=5.0,
            prophet_avg_val_mape=7.0,
            optimized_weights={"catboost": 0.7, "prophet": 0.3},
        )

        df = trainer._generate_comparison_df(catboost_result, prophet_result, training_result)

        assert len(df) == 3
        assert list(df["Model"]) == ["CatBoost", "Prophet", "Ensemble"]


# ---------------------------------------------------------------------------
# Weight save/load
# ---------------------------------------------------------------------------


class TestWeightPersistence:
    """Tests for saving and loading ensemble weights."""

    def test_save_and_load_weights(self, tmp_path: Any) -> None:
        weights = {"catboost": 0.65, "prophet": 0.35}
        path = tmp_path / "weights.json"

        save_ensemble_weights(weights, path)
        loaded = load_ensemble_weights(path)

        assert loaded["catboost"] == pytest.approx(0.65)
        assert loaded["prophet"] == pytest.approx(0.35)

    def test_save_creates_parent_dirs(self, tmp_path: Any) -> None:
        weights = {"catboost": 0.6, "prophet": 0.4}
        path = tmp_path / "nested" / "dir" / "weights.json"

        save_ensemble_weights(weights, path)

        assert path.exists()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    """Tests for result dataclasses."""

    def test_ensemble_split_result_immutable(self) -> None:
        result = EnsembleSplitResult(
            split_idx=0,
            catboost_metrics=_make_mock_metrics(),
            prophet_metrics=_make_mock_metrics(),
            ensemble_metrics=_make_mock_metrics(),
            catboost_predictions=np.zeros(10, dtype=np.float64),
            prophet_predictions=np.zeros(10, dtype=np.float64),
            ensemble_predictions=np.zeros(10, dtype=np.float64),
            y_true=np.ones(10, dtype=np.float64),
            prophet_weight=0.4,
        )

        with pytest.raises(AttributeError):
            result.split_idx = 1  # type: ignore[misc]

    def test_ensemble_training_result_immutable(self) -> None:
        result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=5.0,
            std_val_mape=0.5,
            catboost_avg_val_mape=5.0,
            prophet_avg_val_mape=7.0,
            optimized_weights={"catboost": 0.6, "prophet": 0.4},
        )

        with pytest.raises(AttributeError):
            result.avg_val_mape = 10.0  # type: ignore[misc]
