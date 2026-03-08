"""Tests for the Ensemble training pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# _train_models
# ---------------------------------------------------------------------------


def _make_mock_split_result(
    split_idx: int, mape: float = 5.0
) -> MagicMock:
    """Create a mock SplitResult with val/test predictions."""
    rng = np.random.default_rng(42 + split_idx)
    y_true = 800.0 + rng.random(100) * 400

    sr = MagicMock()
    sr.split_idx = split_idx
    sr.val_metrics = _make_mock_metrics(mape)
    sr.test_metrics = _make_mock_metrics(mape + 0.5)
    sr.val_predictions = y_true * (1 + rng.normal(0, 0.05, 100))
    sr.val_actuals = y_true
    sr.test_predictions = y_true * (1 + rng.normal(0, 0.06, 100))
    sr.test_actuals = y_true
    sr.val_month = "2023-06"
    sr.test_month = "2023-07"
    sr.train_metrics = _make_mock_metrics(mape - 1)
    return sr


def _make_mock_pipeline_result(
    n_splits: int = 3, base_mape: float = 5.0
) -> MagicMock:
    """Create a mock pipeline result with multiple splits."""
    result = MagicMock()
    result.training_result.split_results = [
        _make_mock_split_result(i, base_mape + i * 0.5) for i in range(n_splits)
    ]
    result.training_result.avg_val_mape = base_mape + 0.5
    result.training_result.avg_test_mape = base_mape + 1.0
    return result


class TestTrainModels:
    """Tests for _train_models method."""

    def test_train_models_success(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        # Mock sub-trainers
        cb_result = _make_mock_pipeline_result(3, 5.0)
        pr_result = _make_mock_pipeline_result(3, 7.0)
        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.return_value = cb_result
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.return_value = pr_result

        results, errors = trainer._train_models(_make_feature_df())

        assert "catboost" in results
        assert "prophet" in results
        assert len(errors) == 0

    def test_train_models_with_failure_fallback(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        cb_result = _make_mock_pipeline_result(3, 5.0)
        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.return_value = cb_result
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.side_effect = RuntimeError("Prophet failed")

        results, errors = trainer._train_models(_make_feature_df())

        assert "catboost" in results
        assert "prophet" in errors
        assert "Prophet failed" in str(errors["prophet"])


# ---------------------------------------------------------------------------
# _collect_split_metrics
# ---------------------------------------------------------------------------


class TestCollectSplitMetrics:
    """Tests for _collect_split_metrics method."""

    def test_collects_metrics_from_all_splits(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        model_results = {
            "catboost": _make_mock_pipeline_result(3, 5.0),
            "prophet": _make_mock_pipeline_result(3, 7.0),
        }

        split_results = trainer._collect_split_metrics(model_results)

        assert len(split_results) == 3
        for sr in split_results:
            assert "catboost" in sr.model_metrics
            assert "prophet" in sr.model_metrics
            assert sr.ensemble_metrics.mape > 0

    def test_split_count_mismatch_raises(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        model_results = {
            "catboost": _make_mock_pipeline_result(3, 5.0),
            "prophet": _make_mock_pipeline_result(2, 7.0),
        }

        with pytest.raises(ValueError, match="Split count mismatch"):
            trainer._collect_split_metrics(model_results)


# ---------------------------------------------------------------------------
# Weighted average ensemble
# ---------------------------------------------------------------------------


class TestComputeWeightedAverage:
    """Tests for weighted average ensemble computation."""

    def test_compute_weighted_ensemble_updates_metrics(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        y_true = 800.0 + rng.random(100) * 400

        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0),
                    "prophet": _make_mock_metrics(7.0),
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
        ]

        weights = {"catboost": 0.7, "prophet": 0.3}
        updated = trainer._compute_weighted_ensemble(split_results, weights)

        assert len(updated) == 1
        assert updated[0].weights == weights
        assert updated[0].ensemble_metrics.mape > 0


class TestComputeWeightedTestMape:
    """Tests for _compute_weighted_test_mape."""

    def test_returns_test_mapes(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        model_results = {
            "catboost": _make_mock_pipeline_result(3, 5.0),
            "prophet": _make_mock_pipeline_result(3, 7.0),
        }

        weights = {"catboost": 0.6, "prophet": 0.4}
        test_mapes = trainer._compute_weighted_test_mape(model_results, weights)

        assert len(test_mapes) == 3
        for m in test_mapes:
            assert m > 0


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Tests for _print_summary method."""

    def test_print_summary_weighted_average(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        comparison_df = pd.DataFrame(
            {
                "Model": ["Catboost", "Prophet", "Ensemble (weighted_average)"],
                "Val MAPE (%)": [5.0, 7.0, 4.8],
                "Test MAPE (%)": [5.5, 7.5, 5.0],
                "Weight": [0.7, 0.3, 1.0],
            }
        )
        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.8,
            std_val_mape=0.5,
            model_avg_val_mapes={"catboost": 5.0, "prophet": 7.0},
            optimized_weights={"catboost": 0.7, "prophet": 0.3},
            mode="weighted_average",
        )

        # Should not raise
        trainer._print_summary(comparison_df, training_result)

    def test_print_summary_stacking(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        comparison_df = pd.DataFrame(
            {
                "Model": ["Catboost", "Prophet", "Ensemble (stacking)"],
                "Val MAPE (%)": [5.0, 7.0, 4.5],
                "Test MAPE (%)": [5.5, 7.5, 4.8],
                "Weight": [0.7, 0.3, 1.0],
            }
        )
        training_result = EnsembleTrainingResult(
            split_results=[],
            avg_val_mape=4.5,
            std_val_mape=0.0,
            model_avg_val_mapes={"catboost": 5.0, "prophet": 7.0},
            optimized_weights={"catboost": 0.5, "prophet": 0.5},
            mode="stacking",
        )

        trainer._print_summary(comparison_df, training_result)


# ---------------------------------------------------------------------------
# Full run pipeline (mocked)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for the full ensemble run() pipeline."""

    def test_run_returns_pipeline_result(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        # Mock sub-trainers
        cb_result = _make_mock_pipeline_result(3, 5.0)
        pr_result = _make_mock_pipeline_result(3, 7.0)
        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.return_value = cb_result
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.return_value = pr_result

        # Force weighted_average mode for simpler test
        trainer._mode = "weighted_average"

        result = trainer.run(_make_feature_df())

        assert result.model_results is not None
        assert result.training_result.avg_val_mape > 0
        assert result.comparison_df is not None
        assert len(result.comparison_df) == 3  # 2 models + ensemble
        assert result.training_time_seconds >= 0

    def test_run_all_models_fail_raises(self) -> None:
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.side_effect = RuntimeError("Fail")
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.side_effect = RuntimeError("Fail")

        with pytest.raises(RuntimeError, match="All models failed"):
            trainer.run(_make_feature_df())


# ---------------------------------------------------------------------------
# _train_models fallback disabled
# ---------------------------------------------------------------------------


class TestTrainModelsFallbackDisabled:
    """Tests for _train_models when fallback is disabled."""

    def test_fallback_disabled_raises_on_failure(self) -> None:
        """Test that RuntimeError is raised when fallback is disabled and a model fails."""
        settings = _get_test_settings()
        # Disable fallback
        object.__setattr__(settings.ensemble.fallback, "enabled", False)

        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        cb_result = _make_mock_pipeline_result(3, 5.0)
        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.return_value = cb_result
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.side_effect = RuntimeError("Prophet crashed")

        with pytest.raises(RuntimeError, match="prophet failed and fallback disabled"):
            trainer._train_models(_make_feature_df())


# ---------------------------------------------------------------------------
# run() with partial failure — stacking fallback to weighted_average
# ---------------------------------------------------------------------------


class TestRunWithPartialFailure:
    """Tests for run() when some models fail."""

    def test_stacking_falls_back_to_weighted_average_when_lt_2_models(self) -> None:
        """Test run() falls back to weighted_average when stacking has < 2 models."""
        settings = _get_test_settings()
        # Mode is stacking by default
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )
        trainer._mode = "stacking"

        # catboost succeeds, prophet fails
        cb_result = _make_mock_pipeline_result(3, 5.0)
        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.return_value = cb_result
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.side_effect = RuntimeError("Prophet failed")

        result = trainer.run(_make_feature_df())

        # Only 1 model survived — stacking requires >=2, should fall back
        assert result.training_result.mode == "weighted_average"
        assert "catboost" in result.model_results
        assert "prophet" not in result.model_results

    def test_partial_failure_continues_with_remaining_models(self) -> None:
        """Test run() continues when one of three models fails."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet", "tft"],
        )
        trainer._mode = "weighted_average"

        cb_result = _make_mock_pipeline_result(3, 5.0)
        pr_result = _make_mock_pipeline_result(3, 7.0)
        trainer._trainers["catboost"] = MagicMock()
        trainer._trainers["catboost"].run.return_value = cb_result
        trainer._trainers["prophet"] = MagicMock()
        trainer._trainers["prophet"].run.return_value = pr_result
        trainer._trainers["tft"] = MagicMock()
        trainer._trainers["tft"].run.side_effect = RuntimeError("TFT OOM")

        result = trainer.run(_make_feature_df())

        assert "catboost" in result.model_results
        assert "prophet" in result.model_results
        assert "tft" not in result.model_results
        assert result.training_result.avg_val_mape > 0


# ---------------------------------------------------------------------------
# _build_oof_dataframe
# ---------------------------------------------------------------------------


class TestBuildOofDataframe:
    """Tests for _build_oof_dataframe method."""

    def test_build_oof_returns_correct_columns(self) -> None:
        """Test OOF DataFrame has pred_, context, and y_true columns."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        n_rows = 500
        idx = pd.date_range("2023-01-01", periods=n_rows, freq="h")
        df = pd.DataFrame(
            {
                "consumption": rng.random(n_rows) * 400 + 800,
                "is_holiday": np.zeros(n_rows, dtype=int),
            },
            index=idx,
        ).rename_axis("datetime")

        # Create mock model results with 3 splits, each having val predictions
        n_val = 100
        model_results: dict[str, Any] = {}
        for model_name in ["catboost", "prophet"]:
            mock_result = MagicMock()
            split_results_list = []
            for _i in range(3):
                sr = MagicMock()
                sr.val_predictions = rng.random(n_val) * 400 + 800
                sr.val_actuals = rng.random(n_val) * 400 + 800
                sr.test_predictions = rng.random(n_val) * 400 + 800
                sr.test_actuals = rng.random(n_val) * 400 + 800
                split_results_list.append(sr)
            mock_result.training_result.split_results = split_results_list
            model_results[model_name] = mock_result

        # Mock TimeSeriesSplitter to return 3 splits with matching val slices
        with patch(
            "energy_forecast.training.ensemble_stacking.TimeSeriesSplitter"
        ) as _mock_tss:
            mock_splitter = MagicMock()
            splits = []
            for i in range(3):
                mock_split_info = MagicMock()
                mock_split_info.split_idx = i
                start = i * n_val
                end = start + n_val
                train_df = df.iloc[:start + 50]
                val_slice = df.iloc[start:end]
                test_df = df.iloc[end:end + 50] if end + 50 <= n_rows else df.iloc[-50:]
                splits.append((mock_split_info, train_df, val_slice, test_df))
            mock_splitter.iter_splits.return_value = splits
            _mock_tss.from_config.return_value = mock_splitter

            oof_df = trainer._build_oof_dataframe(model_results, df)

        # Verify columns
        assert "pred_catboost" in oof_df.columns
        assert "pred_prophet" in oof_df.columns
        assert "hour" in oof_df.columns
        assert "day_of_week" in oof_df.columns
        assert "is_weekend" in oof_df.columns
        assert "month" in oof_df.columns
        assert "is_holiday" in oof_df.columns
        assert "y_true" in oof_df.columns

    def test_build_oof_row_count_matches_splits(self) -> None:
        """Test OOF DataFrame has correct total row count from all splits."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(99)
        n_rows = 600
        idx = pd.date_range("2023-01-01", periods=n_rows, freq="h")
        df = pd.DataFrame(
            {
                "consumption": rng.random(n_rows) * 400 + 800,
                "is_holiday": np.zeros(n_rows, dtype=int),
            },
            index=idx,
        ).rename_axis("datetime")

        n_val = 80
        model_results: dict[str, Any] = {}
        for model_name in ["catboost", "prophet"]:
            mock_result = MagicMock()
            split_results_list = []
            for _ in range(2):
                sr = MagicMock()
                sr.val_predictions = rng.random(n_val) * 400 + 800
                sr.val_actuals = rng.random(n_val) * 400 + 800
                split_results_list.append(sr)
            mock_result.training_result.split_results = split_results_list
            model_results[model_name] = mock_result

        with patch(
            "energy_forecast.training.ensemble_stacking.TimeSeriesSplitter"
        ) as _mock_tss:
            mock_splitter = MagicMock()
            splits = []
            for i in range(2):
                mock_split_info = MagicMock()
                mock_split_info.split_idx = i
                start = i * n_val + 100
                val_slice = df.iloc[start : start + n_val]
                splits.append(
                    (mock_split_info, df.iloc[:start], val_slice, df.iloc[start + n_val :])
                )
            mock_splitter.iter_splits.return_value = splits
            _mock_tss.from_config.return_value = mock_splitter

            oof_df = trainer._build_oof_dataframe(model_results, df)

        assert len(oof_df) == 2 * n_val


# ---------------------------------------------------------------------------
# _train_meta_learner
# ---------------------------------------------------------------------------


class TestTrainMetaLearner:
    """Tests for _train_meta_learner method."""

    def test_train_meta_learner_returns_model_and_mape(self) -> None:
        """Test _train_meta_learner returns (model, val_mape) tuple."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        # Create synthetic OOF DataFrame
        rng = np.random.default_rng(42)
        n_rows = 200
        oof_df = pd.DataFrame(
            {
                "pred_catboost": rng.random(n_rows) * 400 + 800,
                "pred_prophet": rng.random(n_rows) * 400 + 800,
                "hour": np.tile(np.arange(24), n_rows // 24 + 1)[:n_rows],
                "day_of_week": np.tile(np.arange(7), n_rows // 7 + 1)[:n_rows],
                "is_weekend": np.tile([0, 0, 0, 0, 0, 1, 1], n_rows // 7 + 1)[:n_rows],
                "month": np.ones(n_rows, dtype=int),
                "is_holiday": np.zeros(n_rows, dtype=int),
                "y_true": rng.random(n_rows) * 400 + 800,
            }
        )

        with patch(
            "energy_forecast.training.ensemble_stacking.CatBoostRegressor"
        ) as _mock_cbr:
            mock_meta = MagicMock()
            mock_meta.predict.return_value = rng.random(40) * 400 + 800
            mock_meta.get_feature_importance.return_value = np.array(
                [50.0, 30.0, 5.0, 5.0, 5.0, 3.0, 2.0]
            )
            _mock_cbr.return_value = mock_meta

            meta_model, val_mape = trainer._train_meta_learner(oof_df)

        assert meta_model is mock_meta
        assert isinstance(val_mape, float)
        assert val_mape >= 0
        mock_meta.fit.assert_called_once()

    def test_train_meta_learner_categorical_conversion(self) -> None:
        """Test that categorical columns are converted to string."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(7)
        n_rows = 100
        oof_df = pd.DataFrame(
            {
                "pred_catboost": rng.random(n_rows) * 400 + 800,
                "pred_prophet": rng.random(n_rows) * 400 + 800,
                "hour": np.tile(np.arange(24), n_rows // 24 + 1)[:n_rows],
                "day_of_week": np.tile(np.arange(7), n_rows // 7 + 1)[:n_rows],
                "is_weekend": np.zeros(n_rows, dtype=int),
                "month": np.ones(n_rows, dtype=int),
                "is_holiday": np.zeros(n_rows, dtype=int),
                "y_true": rng.random(n_rows) * 400 + 800,
            }
        )

        with patch(
            "energy_forecast.training.ensemble_stacking.CatBoostRegressor"
        ) as _mock_cbr:
            mock_meta = MagicMock()
            mock_meta.predict.return_value = np.ones(20) * 1000
            mock_meta.get_feature_importance.return_value = np.array(
                [50.0, 30.0, 5.0, 5.0, 5.0, 3.0, 2.0]
            )
            _mock_cbr.return_value = mock_meta

            trainer._train_meta_learner(oof_df)

        # Verify CatBoostRegressor was created with cat_features
        call_kwargs = _mock_cbr.call_args[1]
        assert "cat_features" in call_kwargs
        # hour, day_of_week, month → indices 2, 3, 5 (relative to feature_cols)
        assert len(call_kwargs["cat_features"]) == 3


# ---------------------------------------------------------------------------
# _compute_stacking_test_mape
# ---------------------------------------------------------------------------


class TestComputeStackingTestMape:
    """Tests for _compute_stacking_test_mape method."""

    def test_returns_empty_when_no_meta_model(self) -> None:
        """Test returns [] when meta_model is None."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )
        trainer._meta_model = None

        result = trainer._compute_stacking_test_mape({}, _make_feature_df())
        assert result == []

    def test_returns_test_mapes_with_meta_model(self) -> None:
        """Test returns list of test MAPEs when meta_model is set."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        n_rows = 500
        idx = pd.date_range("2023-01-01", periods=n_rows, freq="h")
        df = pd.DataFrame(
            {
                "consumption": rng.random(n_rows) * 400 + 800,
                "is_holiday": np.zeros(n_rows, dtype=int),
            },
            index=idx,
        ).rename_axis("datetime")

        # Set up mock meta model
        mock_meta = MagicMock()
        mock_meta.predict.return_value = rng.random(100) * 400 + 800
        trainer._meta_model = mock_meta

        # Create mock model results with test predictions
        n_test = 100
        model_results: dict[str, Any] = {}
        for model_name in ["catboost", "prophet"]:
            mock_result = MagicMock()
            split_results_list = []
            for _ in range(2):
                sr = MagicMock()
                sr.test_predictions = rng.random(n_test) * 400 + 800
                sr.test_actuals = rng.random(n_test) * 400 + 800
                split_results_list.append(sr)
            mock_result.training_result.split_results = split_results_list
            model_results[model_name] = mock_result

        with patch(
            "energy_forecast.training.ensemble_stacking.TimeSeriesSplitter"
        ) as _mock_tss:
            mock_splitter = MagicMock()
            splits = []
            for i in range(2):
                mock_split_info = MagicMock()
                mock_split_info.split_idx = i
                start = i * n_test + 100
                test_slice = df.iloc[start : start + n_test]
                splits.append(
                    (mock_split_info, df.iloc[:start], df.iloc[:50], test_slice)
                )
            mock_splitter.iter_splits.return_value = splits
            _mock_tss.from_config.return_value = mock_splitter

            test_mapes = trainer._compute_stacking_test_mape(model_results, df)

        assert len(test_mapes) == 2
        for m in test_mapes:
            assert isinstance(m, float)
            assert m >= 0


# ---------------------------------------------------------------------------
# _compute_stacking_ensemble
# ---------------------------------------------------------------------------


class TestComputeStackingEnsemble:
    """Tests for _compute_stacking_ensemble method."""

    def test_returns_stacking_result(self) -> None:
        """Test _compute_stacking_ensemble sets meta_model and returns stacking result."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        y_true = rng.random(100) * 400 + 800
        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0),
                    "prophet": _make_mock_metrics(7.0),
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
        ]
        default_weights = {"catboost": 0.6, "prophet": 0.4}
        df = _make_feature_df()

        mock_meta = MagicMock()
        mock_oof_df = pd.DataFrame(
            {
                "pred_catboost": rng.random(100),
                "pred_prophet": rng.random(100),
                "y_true": rng.random(100),
            }
        )

        with (
            patch.object(
                trainer, "_build_oof_dataframe", return_value=mock_oof_df
            ) as mock_build,
            patch.object(
                trainer,
                "_train_meta_learner",
                return_value=(mock_meta, 3.5),
            ) as mock_train,
            patch.object(
                trainer,
                "_compute_stacking_test_mape",
                return_value=[3.2, 3.8],
            ) as mock_test,
        ):
            result = trainer._compute_stacking_ensemble(
                {}, split_results, default_weights, df
            )

        assert result.mode == "stacking"
        assert result.avg_val_mape == 3.5
        assert result.avg_test_mape == pytest.approx(3.5)  # mean of [3.2, 3.8]
        assert trainer._meta_model is mock_meta
        mock_build.assert_called_once()
        mock_train.assert_called_once()
        mock_test.assert_called_once()


# ---------------------------------------------------------------------------
# _collect_split_metrics — no raw predictions fallback
# ---------------------------------------------------------------------------


class TestCollectSplitMetricsNoRawPredictions:
    """Tests for _collect_split_metrics fallback when val_predictions is None."""

    def test_fallback_metric_approximation(self) -> None:
        """Test metric-level approximation when no raw predictions available."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        # Create mock results where val_predictions is None
        model_results: dict[str, Any] = {}
        for model_name in ["catboost", "prophet"]:
            mock_result = MagicMock()
            split_results_list = []
            for i in range(3):
                sr = MagicMock()
                sr.val_metrics = _make_mock_metrics(5.0 + i)
                sr.test_metrics = _make_mock_metrics(6.0 + i)
                sr.val_predictions = None
                sr.val_actuals = None
                sr.test_predictions = None
                sr.test_actuals = None
                split_results_list.append(sr)
            mock_result.training_result.split_results = split_results_list
            model_results[model_name] = mock_result

        split_results = trainer._collect_split_metrics(model_results)

        assert len(split_results) == 3
        for ens_sr in split_results:
            # Verify ensemble metrics are computed from metric-level approximation
            assert ens_sr.ensemble_metrics.mape > 0
            # Predictions should be dummy arrays
            assert len(ens_sr.ensemble_predictions) == 1
            assert len(ens_sr.y_true) == 1


# ---------------------------------------------------------------------------
# _compute_weighted_ensemble — fallback (no raw predictions)
# ---------------------------------------------------------------------------


class TestComputeWeightedEnsembleFallback:
    """Tests for _compute_weighted_ensemble metric-level fallback."""

    def test_metric_level_fallback_when_no_predictions(self) -> None:
        """Test fallback when model_predictions has <= 1 element."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        # Split results with empty model_predictions (simulating no raw preds)
        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0),
                    "prophet": _make_mock_metrics(7.0),
                },
                ensemble_metrics=_make_mock_metrics(),
                model_predictions={},  # Empty — triggers fallback
                ensemble_predictions=np.zeros(1, dtype=np.float64),
                y_true=np.ones(1, dtype=np.float64),
                weights={"catboost": 0.6, "prophet": 0.4},
            )
        ]

        weights = {"catboost": 0.7, "prophet": 0.3}
        updated = trainer._compute_weighted_ensemble(split_results, weights)

        assert len(updated) == 1
        # Fallback should produce weighted average of individual MAPEs
        expected_mape = 0.7 * 5.0 + 0.3 * 7.0  # = 5.6
        assert updated[0].ensemble_metrics.mape == pytest.approx(expected_mape)

    def test_metric_level_fallback_with_single_element_predictions(self) -> None:
        """Test fallback when model_predictions has arrays of length 1."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(4.0),
                    "prophet": _make_mock_metrics(6.0),
                },
                ensemble_metrics=_make_mock_metrics(),
                model_predictions={
                    "catboost": np.array([1000.0]),  # len == 1 triggers fallback
                    "prophet": np.array([1100.0]),
                },
                ensemble_predictions=np.zeros(1, dtype=np.float64),
                y_true=np.ones(1, dtype=np.float64),
                weights={"catboost": 0.6, "prophet": 0.4},
            )
        ]

        weights = {"catboost": 0.5, "prophet": 0.5}
        updated = trainer._compute_weighted_ensemble(split_results, weights)

        assert len(updated) == 1
        # Should be metric-level fallback
        expected_mape = 0.5 * 4.0 + 0.5 * 6.0  # = 5.0
        assert updated[0].ensemble_metrics.mape == pytest.approx(expected_mape)


# ---------------------------------------------------------------------------
# _compute_weighted_test_mape — missing predictions skip
# ---------------------------------------------------------------------------


class TestComputeWeightedTestMapeMissingPredictions:
    """Tests for _compute_weighted_test_mape when predictions are missing."""

    def test_skips_splits_when_all_predictions_missing(self) -> None:
        """Test that splits where ALL models have None test_predictions are skipped."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        y_test = rng.random(100) * 400 + 800

        model_results: dict[str, Any] = {}

        # CatBoost: first split None, second has predictions
        cb_result = MagicMock()
        cb_sr0 = MagicMock()
        cb_sr0.test_predictions = None
        cb_sr0.test_actuals = None
        cb_sr1 = MagicMock()
        cb_sr1.test_predictions = y_test * (1 + rng.normal(0, 0.05, 100))
        cb_sr1.test_actuals = y_test
        cb_result.training_result.split_results = [cb_sr0, cb_sr1]
        model_results["catboost"] = cb_result

        # Prophet: first split None, second has predictions
        pr_result = MagicMock()
        pr_sr0 = MagicMock()
        pr_sr0.test_predictions = None
        pr_sr0.test_actuals = None
        pr_sr1 = MagicMock()
        pr_sr1.test_predictions = y_test * (1 + rng.normal(0, 0.08, 100))
        pr_sr1.test_actuals = y_test
        pr_result.training_result.split_results = [pr_sr0, pr_sr1]
        model_results["prophet"] = pr_result

        weights = {"catboost": 0.6, "prophet": 0.4}
        test_mapes = trainer._compute_weighted_test_mape(model_results, weights)

        # First split skipped (both models missing), only second split computed
        assert len(test_mapes) == 1
        assert test_mapes[0] > 0

    def test_skips_splits_when_y_test_is_none(self) -> None:
        """Test that splits where test_actuals is None are skipped."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost", "prophet"],
        )

        rng = np.random.default_rng(42)
        y_test = rng.random(100) * 400 + 800

        model_results: dict[str, Any] = {}

        # CatBoost: first split has predictions but no actuals
        cb_result = MagicMock()
        cb_sr0 = MagicMock()
        cb_sr0.test_predictions = y_test * (1 + rng.normal(0, 0.05, 100))
        cb_sr0.test_actuals = None
        cb_sr1 = MagicMock()
        cb_sr1.test_predictions = y_test * (1 + rng.normal(0, 0.05, 100))
        cb_sr1.test_actuals = y_test
        cb_result.training_result.split_results = [cb_sr0, cb_sr1]
        model_results["catboost"] = cb_result

        # Prophet: same pattern
        pr_result = MagicMock()
        pr_sr0 = MagicMock()
        pr_sr0.test_predictions = y_test * (1 + rng.normal(0, 0.08, 100))
        pr_sr0.test_actuals = None
        pr_sr1 = MagicMock()
        pr_sr1.test_predictions = y_test * (1 + rng.normal(0, 0.08, 100))
        pr_sr1.test_actuals = y_test
        pr_result.training_result.split_results = [pr_sr0, pr_sr1]
        model_results["prophet"] = pr_result

        weights = {"catboost": 0.6, "prophet": 0.4}
        test_mapes = trainer._compute_weighted_test_mape(model_results, weights)

        # First split skipped (no actuals), only second split computed
        assert len(test_mapes) == 1
        assert test_mapes[0] > 0


# ---------------------------------------------------------------------------
# _compute_weighted_average_ensemble — optimization disabled single model
# ---------------------------------------------------------------------------


class TestComputeWeightedAverageOptDisabled:
    """Tests for _compute_weighted_average_ensemble optimization skip."""

    def test_single_model_skips_optimization(self) -> None:
        """Test that single model uses default weights without optimization."""
        settings = _get_test_settings()
        trainer = EnsembleTrainer(
            settings,
            ExperimentTracker(enabled=False),
            active_models_override=["catboost"],
        )
        trainer._mode = "weighted_average"

        rng = np.random.default_rng(42)
        y_true = rng.random(100) * 400 + 800

        split_results = [
            EnsembleSplitResult(
                split_idx=0,
                model_metrics={
                    "catboost": _make_mock_metrics(5.0),
                },
                ensemble_metrics=_make_mock_metrics(),
                model_predictions={
                    "catboost": y_true * (1 + rng.normal(0, 0.05, 100)),
                },
                ensemble_predictions=np.zeros(100, dtype=np.float64),
                y_true=y_true,
                weights={"catboost": 1.0},
            )
        ]

        model_results: dict[str, Any] = {"catboost": _make_mock_pipeline_result(1, 5.0)}
        default_weights = {"catboost": 1.0}

        result = trainer._compute_weighted_average_ensemble(
            model_results, split_results, default_weights
        )

        # With single model, optimization is skipped, weights stay default
        assert result.optimized_weights == {"catboost": 1.0}
        assert result.mode == "weighted_average"
