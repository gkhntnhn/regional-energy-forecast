"""Tests for the CatBoost training pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import Settings, get_default_config
from energy_forecast.training.catboost_trainer import (
    CatBoostTrainer,
    PipelineResult,
    SplitResult,
    TrainingResult,
)
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(n_months: int = 18) -> pd.DataFrame:
    """Create a minimal feature-engineered DataFrame.

    Generates n_months of hourly data with consumption target + numeric features.
    Enough data for n_splits=3 with 1 val + 1 test month per split.
    """
    rng = np.random.default_rng(42)
    n_rows = n_months * 30 * 24  # approximate
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
    """Get settings configured for fast tests (3 splits, 1 trial)."""
    settings = get_default_config()
    # Override for fast tests using object.__setattr__ on frozen models
    cv_config = settings.hyperparameters.cross_validation
    object.__setattr__(cv_config, "n_splits", 3)

    search_config = settings.hyperparameters.catboost
    object.__setattr__(search_config, "n_trials", 1)
    object.__setattr__(search_config, "search_space", {})

    return settings


# ---------------------------------------------------------------------------
# X/y split
# ---------------------------------------------------------------------------


class TestSplitXY:
    """Tests for target/feature separation."""

    def test_split_xy_separates_target(self) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))
        df = _make_feature_df(6)
        x, y = trainer._split_xy(df)
        assert "consumption" not in x.columns
        assert y.name == "consumption"
        assert len(x) == len(y) == len(df)

    def test_split_xy_preserves_index(self) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))
        df = _make_feature_df(6)
        x, y = trainer._split_xy(df)
        assert x.index.equals(df.index)
        assert y.index.equals(df.index)

    def test_split_xy_all_features_present(self) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))
        df = _make_feature_df(6)
        x, _ = trainer._split_xy(df)
        expected_cols = [c for c in df.columns if c != "consumption"]
        assert list(x.columns) == expected_cols


# ---------------------------------------------------------------------------
# Categorical preparation
# ---------------------------------------------------------------------------


class TestPrepareCategoricals:
    """Tests for categorical column preparation."""

    def test_categoricals_converted_to_str(self) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))
        df = _make_feature_df(6)
        x, _ = trainer._split_xy(df)

        x_prepared, cat_idx = trainer._prepare_categoricals(x)

        # Only columns present in x should be processed
        expected_cats = [c for c in settings.catboost.categorical_features if c in x.columns]
        assert len(cat_idx) == len(expected_cats)

        # All categorical columns should be string dtype (on the returned copy)
        for col in expected_cats:
            assert x_prepared[col].dtype == object  # str in pandas

    def test_categoricals_nan_filled(self) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))
        df = _make_feature_df(6)
        x, _ = trainer._split_xy(df)

        # Inject NaN into a categorical column
        x.loc[x.index[:10], "is_holiday"] = np.nan

        x_prepared, _ = trainer._prepare_categoricals(x)
        assert x_prepared["is_holiday"].isna().sum() == 0
        assert (x_prepared["is_holiday"].iloc[:10] == "missing").all()

    def test_categoricals_returns_indices(self) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))
        df = _make_feature_df(6)
        x, _ = trainer._split_xy(df)

        _, cat_idx = trainer._prepare_categoricals(x)
        for idx in cat_idx:
            assert isinstance(idx, int)
            assert 0 <= idx < len(x.columns)


# ---------------------------------------------------------------------------
# Single split training (with mocked CatBoost)
# ---------------------------------------------------------------------------


class TestTrainSplit:
    """Tests for single CV split training with mocked CatBoost."""

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_train_split_returns_split_result(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))

        # Mock CatBoost model — predict returns array matching input length
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        mock_model.best_iteration_ = 150
        mock_cb_cls.return_value = mock_model

        # Create data slices
        df = _make_feature_df(6)
        train_df = df.iloc[:2000]
        val_df = df.iloc[2000:2500]
        test_df = df.iloc[2500:3000]

        from energy_forecast.training.splitter import SplitInfo

        info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp(train_df.index[0]),
            train_end=pd.Timestamp(train_df.index[-1]),
            val_start=pd.Timestamp(val_df.index[0]),
            val_end=pd.Timestamp(val_df.index[-1]),
            test_start=pd.Timestamp(test_df.index[0]),
            test_end=pd.Timestamp(test_df.index[-1]),
        )

        params: dict[str, Any] = {
            "task_type": "CPU",
            "eval_metric": "MAPE",
            "random_seed": 42,
            "has_time": True,
            "use_best_model": True,
            "iterations": 100,
        }

        result = trainer._train_split(info, train_df, val_df, test_df, params)

        assert isinstance(result, SplitResult)
        assert result.split_idx == 0
        assert result.best_iteration == 150
        assert isinstance(result.train_metrics, MetricsResult)
        assert isinstance(result.val_metrics, MetricsResult)
        assert isinstance(result.test_metrics, MetricsResult)

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_train_split_month_labels(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))

        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        mock_model.best_iteration_ = 100
        mock_cb_cls.return_value = mock_model

        df = _make_feature_df(6)
        train_df = df.iloc[:2000]
        val_df = df.iloc[2000:2500]
        test_df = df.iloc[2500:3000]

        from energy_forecast.training.splitter import SplitInfo

        val_start = pd.Timestamp(val_df.index[0])
        test_start = pd.Timestamp(test_df.index[0])

        info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp(train_df.index[0]),
            train_end=pd.Timestamp(train_df.index[-1]),
            val_start=val_start,
            val_end=pd.Timestamp(val_df.index[-1]),
            test_start=test_start,
            test_end=pd.Timestamp(test_df.index[-1]),
        )

        params: dict[str, Any] = {
            "task_type": "CPU",
            "iterations": 100,
            "use_best_model": True,
        }

        result = trainer._train_split(info, train_df, val_df, test_df, params)
        assert result.val_month == val_start.strftime("%Y-%m")
        assert result.test_month == test_start.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# All splits training (mocked)
# ---------------------------------------------------------------------------


class TestTrainAllSplits:
    """Tests for multi-split training with mocked CatBoost."""

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_train_all_splits_returns_training_result(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))

        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        mock_model.best_iteration_ = 200
        mock_cb_cls.return_value = mock_model

        df = _make_feature_df(18)

        params: dict[str, Any] = {
            "task_type": "CPU",
            "iterations": 100,
            "use_best_model": True,
        }

        result = trainer._train_all_splits(df, params)

        assert isinstance(result, TrainingResult)
        assert len(result.split_results) == 3  # n_splits=3
        assert result.avg_val_mape >= 0
        assert result.avg_test_mape >= 0
        assert result.std_val_mape >= 0
        assert result.avg_best_iteration == 200
        assert len(result.feature_names) > 0
        assert "consumption" not in result.feature_names


# ---------------------------------------------------------------------------
# Optimize (mocked)
# ---------------------------------------------------------------------------


class TestOptimize:
    """Tests for Optuna optimization with mocked CatBoost."""

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_optimize_returns_study_and_result(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))

        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        mock_model.best_iteration_ = 200
        mock_cb_cls.return_value = mock_model

        df = _make_feature_df(18)

        study, result = trainer.optimize(df)

        assert study.best_value >= 0
        assert isinstance(result, TrainingResult)
        assert len(result.split_results) == 3


# ---------------------------------------------------------------------------
# Train final (mocked)
# ---------------------------------------------------------------------------


class TestTrainFinal:
    """Tests for final model training with mocked CatBoost."""

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_train_final_returns_model(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))

        mock_model = MagicMock()
        mock_cb_cls.return_value = mock_model

        df = _make_feature_df(6)
        params: dict[str, Any] = {"depth": 6, "learning_rate": 0.05}

        model = trainer.train_final(df, params, n_iterations=200)

        assert model is mock_model
        mock_model.fit.assert_called_once()

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_train_final_passes_iterations(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        trainer = CatBoostTrainer(settings, ExperimentTracker(enabled=False))

        mock_model = MagicMock()
        mock_cb_cls.return_value = mock_model

        df = _make_feature_df(6)
        params: dict[str, Any] = {"depth": 6}

        trainer.train_final(df, params, n_iterations=500)

        # CatBoostRegressor called with iterations=500
        call_kwargs = mock_cb_cls.call_args[1]
        assert call_kwargs["iterations"] == 500


# ---------------------------------------------------------------------------
# Full pipeline (mocked)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for the full training pipeline with mocked CatBoost."""

    @patch("energy_forecast.training.catboost_trainer.CatBoostRegressor")
    @patch("energy_forecast.training.catboost_trainer.Pool")
    def test_run_returns_pipeline_result(
        self,
        mock_pool_cls: MagicMock,
        mock_cb_cls: MagicMock,
    ) -> None:
        settings = _get_test_settings()
        tracker = ExperimentTracker(enabled=False)
        trainer = CatBoostTrainer(settings, tracker)

        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        mock_model.best_iteration_ = 200
        mock_cb_cls.return_value = mock_model

        df = _make_feature_df(18)
        n_features = len(df.columns) - 1  # minus target
        mock_model.get_feature_importance.return_value = np.ones(n_features) / n_features

        result = trainer.run(df)

        assert isinstance(result, PipelineResult)
        assert result.final_model is mock_model
        assert result.training_time_seconds > 0
        assert isinstance(result.best_params, dict)
        assert isinstance(result.training_result, TrainingResult)


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for frozen dataclass correctness."""

    def test_split_result_frozen(self) -> None:
        m = MetricsResult(mape=5.0, mae=10.0, rmse=12.0, r2=0.9, smape=5.0, wmape=5.0, mbe=1.0)
        sr = SplitResult(
            split_idx=0,
            train_metrics=m,
            val_metrics=m,
            test_metrics=m,
            best_iteration=100,
            val_month="2024-01",
            test_month="2024-02",
        )
        with pytest.raises(AttributeError):
            sr.split_idx = 1  # type: ignore[misc]

    def test_training_result_frozen(self) -> None:
        tr = TrainingResult(
            split_results=[],
            avg_val_mape=5.0,
            avg_test_mape=6.0,
            std_val_mape=1.0,
            avg_best_iteration=200,
            feature_names=["a", "b"],
        )
        with pytest.raises(AttributeError):
            tr.avg_val_mape = 10.0  # type: ignore[misc]
