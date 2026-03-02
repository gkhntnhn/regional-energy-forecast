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

    tft_config = settings.hyperparameters.tft
    object.__setattr__(tft_config, "n_trials", 1)
    object.__setattr__(tft_config, "search_space", {})

    return settings


def _make_mock_metrics(mape: float = 5.0) -> MetricsResult:
    """Create mock metrics for testing."""
    return MetricsResult(
        mape=mape,
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
        total = weights.catboost + weights.prophet + weights.tft
        assert abs(total - 1.0) < 1e-6

    def test_default_catboost_weight(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.weights.catboost == 0.45

    def test_default_prophet_weight(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.weights.prophet == 0.30

    def test_default_tft_weight(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.weights.tft == 0.25

    def test_optimization_enabled_by_default(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.optimization.enabled is True

    def test_fallback_enabled_by_default(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.fallback.enabled is True

    def test_default_active_models(self) -> None:
        settings = get_default_config()
        assert settings.ensemble.active_models == ["catboost", "prophet", "tft"]

    def test_weight_normalization_all_models(self) -> None:
        settings = get_default_config()
        normalized = settings.ensemble.weights.get_normalized(
            ["catboost", "prophet", "tft"]
        )
        assert abs(sum(normalized.values()) - 1.0) < 1e-6
        assert normalized["catboost"] == pytest.approx(0.45)
        assert normalized["prophet"] == pytest.approx(0.30)
        assert normalized["tft"] == pytest.approx(0.25)

    def test_weight_normalization_two_models(self) -> None:
        settings = get_default_config()
        normalized = settings.ensemble.weights.get_normalized(["catboost", "prophet"])
        assert abs(sum(normalized.values()) - 1.0) < 1e-6
        # 0.45 / (0.45 + 0.30) = 0.6, 0.30 / 0.75 = 0.4
        assert normalized["catboost"] == pytest.approx(0.6)
        assert normalized["prophet"] == pytest.approx(0.4)


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

    def test_trainer_creates_all_sub_trainers(self) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        trainer = EnsembleTrainer(settings, tracker)
        assert "catboost" in trainer._trainers
        assert "prophet" in trainer._trainers
        assert "tft" in trainer._trainers

    def test_trainer_active_models_override(self) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        trainer = EnsembleTrainer(
            settings, tracker, active_models_override=["catboost", "prophet"]
        )
        assert trainer._active_models == ["catboost", "prophet"]
        assert "catboost" in trainer._trainers
        assert "prophet" in trainer._trainers
        assert "tft" not in trainer._trainers

    def test_trainer_single_model_override(self) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        trainer = EnsembleTrainer(
            settings, tracker, active_models_override=["catboost"]
        )
        assert trainer._active_models == ["catboost"]
        assert len(trainer._trainers) == 1

    def test_trainer_invalid_model_raises(self) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        with pytest.raises(ValueError, match="Unknown model"):
            EnsembleTrainer(
                settings, tracker, active_models_override=["catboost", "invalid"]
            )


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------


class TestWeightOptimization:
    """Tests for ensemble weight optimization."""

    def test_optimize_weights_returns_valid_weights(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        y_true = 800.0 + rng.random(100) * 400

        # Create mock split results with real predictions
        split_results = [
            EnsembleSplitResult(
                split_idx=i,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0 + i * 0.5),
                    "prophet": _make_mock_metrics(7.0 + i * 0.3),
                },
                ensemble_metrics=_make_mock_metrics(),
                model_predictions={
                    "catboost": y_true * (1 + rng.normal(0, 0.05, 100)),
                    "prophet": y_true * (1 + rng.normal(0, 0.08, 100)),
                },
                ensemble_predictions=np.zeros(100, dtype=np.float64),
                y_true=y_true,
                weights={"catboost": 0.6, "prophet": 0.4},
            )
            for i in range(3)
        ]

        initial_weights = {"catboost": 0.6, "prophet": 0.4}
        weights = trainer._optimize_weights(split_results, initial_weights)

        assert "catboost" in weights
        assert "prophet" in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert 0.0 <= weights["catboost"] <= 1.0
        assert 0.0 <= weights["prophet"] <= 1.0

    def test_optimize_weights_respects_bounds(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(123)
        y_true = 800.0 + rng.random(100) * 400

        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0),
                    "prophet": _make_mock_metrics(10.0),
                },
                ensemble_metrics=_make_mock_metrics(),
                model_predictions={
                    "catboost": y_true * (1 + rng.normal(0, 0.04, 100)),
                    "prophet": y_true * (1 + rng.normal(0, 0.12, 100)),
                },
                ensemble_predictions=np.zeros(100, dtype=np.float64),
                y_true=y_true,
                weights={"catboost": 0.6, "prophet": 0.4},
            )
        ]

        initial_weights = {"catboost": 0.6, "prophet": 0.4}
        weights = trainer._optimize_weights(split_results, initial_weights)

        bounds_cfg = settings.ensemble.optimization.bounds
        # Check bounds
        assert weights["catboost"] >= bounds_cfg.catboost[0]
        assert weights["catboost"] <= bounds_cfg.catboost[1]
        assert weights["prophet"] >= bounds_cfg.prophet[0]
        assert weights["prophet"] <= bounds_cfg.prophet[1]

    def test_optimize_weights_three_models(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(settings, ExperimentTracker(enabled=False))

        rng = np.random.default_rng(99)
        y_true = 800.0 + rng.random(100) * 400

        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0),
                    "prophet": _make_mock_metrics(7.0),
                    "tft": _make_mock_metrics(6.0),
                },
                ensemble_metrics=_make_mock_metrics(),
                model_predictions={
                    "catboost": y_true * (1 + rng.normal(0, 0.05, 100)),
                    "prophet": y_true * (1 + rng.normal(0, 0.08, 100)),
                    "tft": y_true * (1 + rng.normal(0, 0.06, 100)),
                },
                ensemble_predictions=np.zeros(100, dtype=np.float64),
                y_true=y_true,
                weights={"catboost": 0.45, "prophet": 0.30, "tft": 0.25},
            )
        ]

        initial_weights = {"catboost": 0.45, "prophet": 0.30, "tft": 0.25}
        weights = trainer._optimize_weights(split_results, initial_weights)

        assert "catboost" in weights
        assert "prophet" in weights
        assert "tft" in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Comparison DataFrame
# ---------------------------------------------------------------------------


class TestComparisonDF:
    """Tests for comparison DataFrame generation."""

    def test_comparison_df_has_correct_columns(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        # Mock results
        catboost_result = MagicMock()
        catboost_result.training_result.avg_val_mape = 5.0
        catboost_result.training_result.avg_test_mape = 5.5

        prophet_result = MagicMock()
        prophet_result.training_result.avg_val_mape = 7.0
        prophet_result.training_result.avg_test_mape = 7.5

        model_results = {
            "catboost": catboost_result,
            "prophet": prophet_result,
        }

        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.8,
            std_val_mape=0.5,
            model_avg_val_mapes={"catboost": 5.0, "prophet": 7.0},
            optimized_weights={"catboost": 0.7, "prophet": 0.3},
        )

        df = trainer._generate_comparison_df(model_results, training_result)

        assert "Model" in df.columns
        assert "Val MAPE (%)" in df.columns
        assert "Test MAPE (%)" in df.columns
        assert "Weight" in df.columns

    def test_comparison_df_three_models_has_four_rows(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(settings, ExperimentTracker(enabled=False))

        catboost_result = MagicMock()
        catboost_result.training_result.avg_val_mape = 5.0
        catboost_result.training_result.avg_test_mape = 5.5

        prophet_result = MagicMock()
        prophet_result.training_result.avg_val_mape = 7.0
        prophet_result.training_result.avg_test_mape = 7.5

        tft_result = MagicMock()
        tft_result.training_result.avg_val_mape = 6.0
        tft_result.training_result.avg_test_mape = 6.5

        model_results = {
            "catboost": catboost_result,
            "prophet": prophet_result,
            "tft": tft_result,
        }

        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.5,
            std_val_mape=0.5,
            model_avg_val_mapes={"catboost": 5.0, "prophet": 7.0, "tft": 6.0},
            optimized_weights={"catboost": 0.5, "prophet": 0.2, "tft": 0.3},
        )

        df = trainer._generate_comparison_df(model_results, training_result)

        assert len(df) == 4  # 3 models + 1 ensemble
        assert any("Ensemble" in m for m in df["Model"].values)

    def test_comparison_df_two_models_has_three_rows(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        catboost_result = MagicMock()
        catboost_result.training_result.avg_val_mape = 5.0
        catboost_result.training_result.avg_test_mape = 5.5

        prophet_result = MagicMock()
        prophet_result.training_result.avg_val_mape = 7.0
        prophet_result.training_result.avg_test_mape = 7.5

        model_results = {
            "catboost": catboost_result,
            "prophet": prophet_result,
        }

        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.8,
            std_val_mape=0.5,
            model_avg_val_mapes={"catboost": 5.0, "prophet": 7.0},
            optimized_weights={"catboost": 0.7, "prophet": 0.3},
        )

        df = trainer._generate_comparison_df(model_results, training_result)

        assert len(df) == 3
        model_names = list(df["Model"])
        assert "Catboost" in model_names
        assert "Prophet" in model_names
        assert any("Ensemble" in m for m in model_names)


# ---------------------------------------------------------------------------
# Weight save/load
# ---------------------------------------------------------------------------


class TestWeightPersistence:
    """Tests for saving and loading ensemble weights."""

    def test_save_and_load_weights(self, tmp_path: Any) -> None:
        weights = {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}
        path = tmp_path / "weights.json"

        save_ensemble_weights(weights, path)
        loaded = load_ensemble_weights(path)

        assert loaded["catboost"] == pytest.approx(0.5)
        assert loaded["prophet"] == pytest.approx(0.3)
        assert loaded["tft"] == pytest.approx(0.2)

    def test_save_creates_parent_dirs(self, tmp_path: Any) -> None:
        weights = {"catboost": 0.45, "prophet": 0.30, "tft": 0.25}
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
            model_metrics={"catboost": _make_mock_metrics(), "prophet": _make_mock_metrics()},
            ensemble_metrics=_make_mock_metrics(),
            model_predictions={
                "catboost": np.zeros(10, dtype=np.float64),
                "prophet": np.zeros(10, dtype=np.float64),
            },
            ensemble_predictions=np.zeros(10, dtype=np.float64),
            y_true=np.ones(10, dtype=np.float64),
            weights={"catboost": 0.6, "prophet": 0.4},
        )

        with pytest.raises(AttributeError):
            result.split_idx = 1  # type: ignore[misc]

    def test_ensemble_training_result_immutable(self) -> None:
        result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=5.0,
            std_val_mape=0.5,
            model_avg_val_mapes={"catboost": 5.0, "prophet": 7.0},
            optimized_weights={"catboost": 0.6, "prophet": 0.4},
        )

        with pytest.raises(AttributeError):
            result.avg_val_mape = 10.0  # type: ignore[misc]
