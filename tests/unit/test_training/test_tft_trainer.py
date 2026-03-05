"""Unit tests for TFTTrainer (NeuralForecast implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config.settings import (
    CrossValidationConfig,
    HyperparameterConfig,
    ModelSearchConfig,
    SearchParamConfig,
    TFTArchitectureConfig,
    TFTConfig,
    TFTCovariatesConfig,
    TFTTrainingConfig,
)
from energy_forecast.training.metrics import MetricsResult
from energy_forecast.training.tft_trainer import TFTSplitResult, TFTTrainingResult


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample feature-engineered DataFrame for testing."""
    n_samples = 24 * 30 * 6  # 6 months for TSCV splits
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="h")

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "consumption": (
                1000
                + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)
                + rng.standard_normal(n_samples) * 50
            ),
            "hour_sin": np.sin(np.arange(n_samples) % 24 * 2 * np.pi / 24),
            "hour_cos": np.cos(np.arange(n_samples) % 24 * 2 * np.pi / 24),
            "day_of_week_sin": np.sin(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "day_of_week_cos": np.cos(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "month_sin": np.zeros(n_samples),
            "month_cos": np.ones(n_samples),
            "is_holiday": np.zeros(n_samples, dtype=int),
            "is_weekend": (np.arange(n_samples) // 24 % 7 >= 5).astype(int),
            "is_ramadan": np.zeros(n_samples, dtype=int),
            "sol_elevation": np.maximum(
                0, 45 * np.sin((np.arange(n_samples) % 24 - 6) * np.pi / 12)
            ),
            "sol_azimuth": 180 + 90 * np.sin((np.arange(n_samples) % 24 - 12) * np.pi / 12),
            "temperature_2m": 15 + 10 * np.sin((np.arange(n_samples) % 24 - 14) * np.pi / 12),
            "relative_humidity_2m": 60 + 20 * rng.standard_normal(n_samples),
            "apparent_temperature": 14 + 10 * np.sin((np.arange(n_samples) % 24 - 14) * np.pi / 12),
        },
        index=dates,
    )
    return df


@pytest.fixture
def tft_config() -> TFTConfig:
    """Create minimal TFT config for fast tests."""
    return TFTConfig(
        architecture=TFTArchitectureConfig(
            hidden_size=16,
            n_head=1,
            n_rnn_layers=1,
            dropout=0.1,
        ),
        training=TFTTrainingConfig(
            encoder_length=48,  # 2 days
            prediction_length=24,  # 1 day
            max_steps=20,
            windows_batch_size=64,
            learning_rate=0.01,
            early_stop_patience_steps=-1,
            val_check_steps=10,
            gradient_clip_val=0.1,
            random_seed=42,
            accelerator="cpu",
            num_workers=0,
            precision="32-true",
            scaler_type="robust",
            rnn_type="lstm",
        ),
        covariates=TFTCovariatesConfig(
            time_varying_known=[
                "hour_sin",
                "hour_cos",
                "day_of_week_sin",
                "day_of_week_cos",
                "temperature_2m",
            ],
            time_varying_unknown=[],
        ),
        quantiles=[0.10, 0.50, 0.90],
        loss="quantile",
    )


@pytest.fixture
def mock_settings(tft_config: TFTConfig) -> MagicMock:
    """Create mock settings with TFT config."""
    settings = MagicMock()
    settings.tft = tft_config
    settings.hyperparameters = HyperparameterConfig(
        tft=ModelSearchConfig(
            n_trials=1,
            search_space={
                "hidden_size": SearchParamConfig(type="categorical", choices=[16, 32]),
                "learning_rate": SearchParamConfig(type="float", low=0.001, high=0.01, log=True),
            },
        ),
        cross_validation=CrossValidationConfig(
            n_splits=2,
            val_months=1,
            test_months=1,
        ),
        target_col="consumption",
    )
    return settings


class TestTFTTrainerInit:
    """Tests for TFTTrainer initialization."""

    def test_init_with_settings(self, mock_settings: MagicMock) -> None:
        """Test trainer initializes with settings."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        assert trainer._tft_config == mock_settings.tft
        assert trainer._target_col == "consumption"

    def test_init_with_disabled_tracker(self, mock_settings: MagicMock) -> None:
        """Test trainer initializes with disabled tracker."""
        from energy_forecast.training.experiment import ExperimentTracker
        from energy_forecast.training.tft_trainer import TFTTrainer

        tracker = ExperimentTracker(enabled=False)
        trainer = TFTTrainer(mock_settings, tracker)
        assert trainer._tracker._enabled is False


class TestTFTTrainerBuildConfig:
    """Tests for config building with Optuna params."""

    def test_build_config_with_overrides(self, mock_settings: MagicMock) -> None:
        """Test config building applies Optuna params."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        params = {"hidden_size": 32, "learning_rate": 0.005}
        new_config = trainer._build_tft_config(params)

        assert new_config.architecture.hidden_size == 32
        assert new_config.training.learning_rate == 0.005
        # Base params preserved
        assert new_config.training.encoder_length == mock_settings.tft.training.encoder_length

    def test_build_config_nf_field_names(self, mock_settings: MagicMock) -> None:
        """Test config uses NeuralForecast field names."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        params = {"n_head": 2, "n_rnn_layers": 2, "windows_batch_size": 512}
        new_config = trainer._build_tft_config(params)

        assert new_config.architecture.n_head == 2
        assert new_config.architecture.n_rnn_layers == 2
        assert new_config.training.windows_batch_size == 512


class TestTFTTrainerSplit:
    """Tests for single split training."""

    @pytest.mark.slow
    def test_train_split_returns_result(
        self,
        mock_settings: MagicMock,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test single split training returns valid result."""
        from energy_forecast.training.splitter import SplitInfo
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        # Create manual split info
        split_info = SplitInfo(
            split_idx=0,
            train_start=sample_df.index[0],
            train_end=sample_df.index[299],
            val_start=sample_df.index[300],
            val_end=sample_df.index[399],
            test_start=sample_df.index[400],
            test_end=sample_df.index[499],
        )

        train_df = sample_df.iloc[:300]
        val_df = sample_df.iloc[300:400]
        test_df = sample_df.iloc[400:]

        result = trainer._train_split(
            split_info,
            train_df,
            val_df,
            test_df,
            params={},
            max_steps=20,
        )

        assert result.split_idx == 0
        assert result.val_metrics.mape > 0
        assert result.test_metrics.mape > 0


class TestTFTOptunaObjective:
    """Tests for Optuna objective function."""

    def test_create_objective_returns_callable(
        self,
        mock_settings: MagicMock,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test objective creation returns callable."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        objective, trial_results = trainer._create_objective(sample_df)

        assert callable(objective)
        assert isinstance(trial_results, dict)

    @pytest.mark.slow
    def test_objective_returns_float(
        self,
        mock_settings: MagicMock,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test objective function returns float MAPE."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        objective, _trial_results = trainer._create_objective(sample_df)

        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_categorical.side_effect = [16]
        mock_trial.suggest_float.return_value = 0.005
        mock_trial.should_prune.return_value = False

        # Patch suggest_params to return simple params
        with patch(
            "energy_forecast.training.tft_trainer.suggest_params",
            return_value={"hidden_size": 16, "learning_rate": 0.005},
        ):
            result = objective(mock_trial)

        assert isinstance(result, float)
        assert result > 0 or result == float("inf")


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_metrics(mape: float = 5.0) -> MetricsResult:
    """Create a MetricsResult with sensible defaults."""
    return MetricsResult(
        mape=mape, mae=50.0, rmse=60.0, r2=0.9, smape=5.0, wmape=5.0, mbe=2.0,
    )


def _make_split_result(
    split_idx: int = 0,
    val_mape: float = 5.0,
    test_mape: float = 6.0,
    val_month: str = "2023-06",
    test_month: str = "2023-07",
) -> TFTSplitResult:
    """Create a TFTSplitResult with sensible defaults."""
    return TFTSplitResult(
        split_idx=split_idx,
        train_metrics=_make_metrics(4.0),
        val_metrics=_make_metrics(val_mape),
        test_metrics=_make_metrics(test_mape),
        val_month=val_month,
        test_month=test_month,
        val_predictions=np.array([100.0, 200.0]),
        val_actuals=np.array([105.0, 195.0]),
        test_predictions=np.array([110.0, 210.0]),
        test_actuals=np.array([115.0, 205.0]),
    )


def _make_split_info(split_idx: int = 0) -> Any:
    """Create a SplitInfo with dummy timestamps."""
    from energy_forecast.training.splitter import SplitInfo

    base = pd.Timestamp("2023-01-01")
    return SplitInfo(
        split_idx=split_idx,
        train_start=base,
        train_end=base + pd.DateOffset(months=3),
        val_start=base + pd.DateOffset(months=3),
        val_end=base + pd.DateOffset(months=4),
        test_start=base + pd.DateOffset(months=4),
        test_end=base + pd.DateOffset(months=5),
    )


# ---------------------------------------------------------------------------
# TestOptunaStorage
# ---------------------------------------------------------------------------


class TestOptunaStorage:
    """Tests for _optuna_storage method."""

    def test_optuna_storage_returns_none_when_few_trials(
        self, mock_settings: MagicMock,
    ) -> None:
        """Test _optuna_storage returns None when n_trials <= 3."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        # mock_settings already has n_trials=1
        trainer = TFTTrainer(mock_settings)
        result = trainer._optuna_storage("tft")
        assert result is None

    def test_optuna_storage_returns_sqlite_url_when_many_trials(
        self, tft_config: TFTConfig, tmp_path: Path,
    ) -> None:
        """Test _optuna_storage returns SQLite URL when n_trials > 3."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        # Build a fresh mock_settings with n_trials=10 (frozen Pydantic, so create new)
        settings = MagicMock()
        settings.tft = tft_config
        settings.hyperparameters = HyperparameterConfig(
            tft=ModelSearchConfig(
                n_trials=10,
                search_space={
                    "hidden_size": SearchParamConfig(type="categorical", choices=[16]),
                },
            ),
            cross_validation=CrossValidationConfig(n_splits=2, val_months=1, test_months=1),
            target_col="consumption",
        )
        settings.paths.models_dir = str(tmp_path)

        trainer = TFTTrainer(settings)
        result = trainer._optuna_storage("tft")

        assert result is not None
        assert result.startswith("sqlite:///")
        assert "optuna_studies" in result
        assert result.endswith("tft.db")
        # Verify directory was created
        studies_dir = tmp_path / "optuna_studies"
        assert studies_dir.exists()


# ---------------------------------------------------------------------------
# TestTrainAllSplits
# ---------------------------------------------------------------------------


class TestTrainAllSplits:
    """Tests for _train_all_splits method."""

    def test_train_all_splits_aggregates_results(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test _train_all_splits iterates splits and aggregates metrics."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        split_result_0 = _make_split_result(split_idx=0, val_mape=4.0, test_mape=5.0)
        split_result_1 = _make_split_result(split_idx=1, val_mape=6.0, test_mape=7.0)

        # Build fake iter_splits output
        info_0 = _make_split_info(0)
        info_1 = _make_split_info(1)
        fake_splits = [
            (info_0, sample_df.iloc[:100], sample_df.iloc[100:200], sample_df.iloc[200:300]),
            (info_1, sample_df.iloc[:200], sample_df.iloc[200:300], sample_df.iloc[300:400]),
        ]

        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter(fake_splits)),
            patch.object(
                trainer,
                "_train_split",
                side_effect=[split_result_0, split_result_1],
            ),
        ):
            result = trainer._train_all_splits(sample_df, params={"hidden_size": 16})

        assert isinstance(result, TFTTrainingResult)
        assert len(result.split_results) == 2
        assert result.avg_val_mape == pytest.approx(5.0)  # mean(4.0, 6.0)
        assert result.avg_test_mape == pytest.approx(6.0)  # mean(5.0, 7.0)
        assert result.std_val_mape == pytest.approx(float(np.std([4.0, 6.0])))

    def test_train_all_splits_single_split(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test _train_all_splits works with a single split."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        split_result = _make_split_result(split_idx=0, val_mape=3.5, test_mape=4.2)
        info = _make_split_info(0)
        fake_splits = [
            (info, sample_df.iloc[:100], sample_df.iloc[100:200], sample_df.iloc[200:300]),
        ]

        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter(fake_splits)),
            patch.object(trainer, "_train_split", return_value=split_result),
        ):
            result = trainer._train_all_splits(sample_df, params={})

        assert len(result.split_results) == 1
        assert result.avg_val_mape == pytest.approx(3.5)
        assert result.avg_test_mape == pytest.approx(4.2)
        assert result.std_val_mape == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestCreateObjectiveEdgeCases
# ---------------------------------------------------------------------------


class TestCreateObjectiveEdgeCases:
    """Tests for _create_objective edge cases."""

    def test_create_objective_raises_when_no_splits(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test _create_objective raises ValueError when no CV splits available."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        # Mock iter_splits to return empty list
        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter([])),
            pytest.raises(ValueError, match="No CV splits available"),
        ):
            trainer._create_objective(sample_df)

    def test_create_objective_linspace_selection(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test _create_objective uses np.linspace when optuna_splits < total splits."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        # Create 6 fake splits
        fake_splits = []
        for i in range(6):
            info = _make_split_info(i)
            chunk = sample_df.iloc[:100]
            fake_splits.append((info, chunk, chunk, chunk))

        # optuna_splits defaults to 2, so with 6 splits it should select 2 via linspace
        # np.linspace(0, 5, 2, dtype=int) = [0, 5]
        with patch.object(trainer._splitter, "iter_splits", return_value=iter(fake_splits)):
            objective, trial_results = trainer._create_objective(sample_df)

        assert callable(objective)
        assert isinstance(trial_results, dict)

    def test_create_objective_all_splits_when_optuna_splits_exceeds_total(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test _create_objective uses all splits when optuna_splits >= total splits."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        # Create 1 fake split (less than optuna_splits=2)
        info = _make_split_info(0)
        chunk = sample_df.iloc[:100]
        fake_splits = [(info, chunk, chunk, chunk)]

        with patch.object(trainer._splitter, "iter_splits", return_value=iter(fake_splits)):
            objective, _trial_results = trainer._create_objective(sample_df)

        assert callable(objective)

    def test_objective_function_returns_inf_on_split_failure(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test objective returns inf when a split raises an exception."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        info = _make_split_info(0)
        chunk = sample_df.iloc[:100]
        fake_splits = [(info, chunk, chunk, chunk)]

        with patch.object(trainer._splitter, "iter_splits", return_value=iter(fake_splits)):
            objective, _trial_results = trainer._create_objective(sample_df)

        mock_trial = MagicMock()
        mock_trial.number = 0

        # _train_split raises a generic exception (not TrialPruned)
        with (
            patch(
                "energy_forecast.training.tft_trainer.suggest_params",
                return_value={"hidden_size": 16},
            ),
            patch.object(
                trainer,
                "_train_split",
                side_effect=RuntimeError("GPU OOM"),
            ),
        ):
            result = objective(mock_trial)

        assert result == float("inf")

    def test_objective_function_reraises_trial_pruned(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test objective re-raises TrialPruned exceptions."""
        from optuna import TrialPruned

        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)

        info = _make_split_info(0)
        chunk = sample_df.iloc[:100]
        fake_splits = [(info, chunk, chunk, chunk)]

        with patch.object(trainer._splitter, "iter_splits", return_value=iter(fake_splits)):
            objective, _trial_results = trainer._create_objective(sample_df)

        mock_trial = MagicMock()
        mock_trial.number = 0

        with (
            patch(
                "energy_forecast.training.tft_trainer.suggest_params",
                return_value={"hidden_size": 16},
            ),
            patch.object(
                trainer,
                "_train_split",
                side_effect=TrialPruned("Step-level prune"),
            ),
            pytest.raises(TrialPruned),
        ):
            objective(mock_trial)


# ---------------------------------------------------------------------------
# TestOptimize
# ---------------------------------------------------------------------------


class TestOptimize:
    """Tests for the optimize method."""

    def _make_trainer(self, mock_settings: MagicMock) -> Any:
        """Create a TFTTrainer for optimization tests."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        return TFTTrainer(mock_settings)

    def _make_mock_study(
        self,
        best_value: float = 3.5,
        best_trial_number: int = 0,
        best_params: dict[str, Any] | None = None,
    ) -> MagicMock:
        """Create a mock Optuna study with typical attributes."""
        study = MagicMock()
        study.best_value = best_value
        study.best_params = best_params or {"hidden_size": 16, "learning_rate": 0.005}
        study.best_trial.number = best_trial_number
        study.best_trial.user_attrs = {"avg_test_mape": 4.0}
        return study

    def test_optimize_cache_hit(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test optimize uses cached splits when best trial is in trial_results."""
        trainer = self._make_trainer(mock_settings)

        cached_split = _make_split_result(split_idx=0, val_mape=3.5, test_mape=4.0)
        trial_results: dict[int, list[TFTSplitResult]] = {0: [cached_split]}
        mock_study = self._make_mock_study(best_value=3.5, best_trial_number=0)

        with (
            patch(
                "energy_forecast.training.tft_trainer.create_study",
                return_value=mock_study,
            ),
            patch.object(
                trainer,
                "_create_objective",
                return_value=(MagicMock(), trial_results),
            ),
            patch.object(trainer, "_optuna_storage", return_value=None),
        ):
            study, result = trainer.optimize(sample_df)

        assert study is mock_study
        assert isinstance(result, TFTTrainingResult)
        assert len(result.split_results) == 1
        assert result.avg_val_mape == pytest.approx(3.5)
        assert result.avg_test_mape == pytest.approx(4.0)

    def test_optimize_skip_validation(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test optimize skips validation when _skip_validation is True."""
        trainer = self._make_trainer(mock_settings)
        trainer._skip_validation = True

        # Empty trial_results cache forces cache miss, but skip_validation kicks in
        trial_results: dict[int, list[TFTSplitResult]] = {}
        mock_study = self._make_mock_study(best_value=4.0, best_trial_number=99)

        with (
            patch(
                "energy_forecast.training.tft_trainer.create_study",
                return_value=mock_study,
            ),
            patch.object(
                trainer,
                "_create_objective",
                return_value=(MagicMock(), trial_results),
            ),
            patch.object(trainer, "_optuna_storage", return_value=None),
        ):
            study, result = trainer.optimize(sample_df)

        assert study is mock_study
        assert isinstance(result, TFTTrainingResult)
        assert result.split_results == []
        assert result.avg_val_mape == pytest.approx(4.0)
        assert result.std_val_mape == pytest.approx(0.0)

    def test_optimize_cache_miss_retrains(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test optimize retrains on all splits when cache miss and no skip."""
        trainer = self._make_trainer(mock_settings)
        trainer._skip_validation = False

        # Empty cache with best_trial_number not in it
        trial_results: dict[int, list[TFTSplitResult]] = {}
        mock_study = self._make_mock_study(best_value=3.0, best_trial_number=5)

        retrained_result = TFTTrainingResult(
            split_results=[_make_split_result(0)],
            avg_val_mape=3.0,
            avg_test_mape=3.5,
            std_val_mape=0.5,
        )

        with (
            patch(
                "energy_forecast.training.tft_trainer.create_study",
                return_value=mock_study,
            ),
            patch.object(
                trainer,
                "_create_objective",
                return_value=(MagicMock(), trial_results),
            ),
            patch.object(trainer, "_optuna_storage", return_value=None),
            patch.object(
                trainer,
                "_train_all_splits",
                return_value=retrained_result,
            ) as mock_retrain,
        ):
            study, result = trainer.optimize(sample_df)

        assert study is mock_study
        assert result is retrained_result
        mock_retrain.assert_called_once_with(sample_df, mock_study.best_params)


# ---------------------------------------------------------------------------
# TestTrainFinal
# ---------------------------------------------------------------------------


class TestTrainFinal:
    """Tests for train_final method."""

    def test_train_final_with_val_split(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """Test train_final splits data when dataset is large enough."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        params: dict[str, Any] = {"hidden_size": 16}

        mock_model = MagicMock()
        mock_forecaster_cls = MagicMock(return_value=mock_model)

        with patch(
            "energy_forecast.training.tft_trainer.TFTForecaster",
            mock_forecaster_cls,
        ):
            result = trainer.train_final(sample_df, params)

        # Verify TFTForecaster was created
        mock_forecaster_cls.assert_called_once()

        # Verify train was called with train and val DataFrames
        mock_model.train.assert_called_once()
        call_args = mock_model.train.call_args
        train_df_arg = call_args[0][0]
        val_df_arg = call_args[0][1]

        # val_size_hours defaults to 720
        val_size = trainer._tft_config.optimization.val_size_hours
        assert len(train_df_arg) == len(sample_df) - val_size
        assert len(val_df_arg) == val_size
        assert result is mock_model

    def test_train_final_without_val_split_small_data(
        self, mock_settings: MagicMock,
    ) -> None:
        """Test train_final uses all data when dataset is too small for val split."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        trainer = TFTTrainer(mock_settings)
        params: dict[str, Any] = {"hidden_size": 16}

        # Create a small DataFrame (less than val_size_hours * 2)
        val_size = trainer._tft_config.optimization.val_size_hours
        small_n = val_size  # exactly val_size, so len <= val_size * 2
        dates = pd.date_range("2023-01-01", periods=small_n, freq="h")
        small_df = pd.DataFrame(
            {"consumption": np.ones(small_n) * 100.0},
            index=dates,
        )

        mock_model = MagicMock()
        mock_forecaster_cls = MagicMock(return_value=mock_model)

        with patch(
            "energy_forecast.training.tft_trainer.TFTForecaster",
            mock_forecaster_cls,
        ):
            result = trainer.train_final(small_df, params)

        # Verify train was called with all data and val=None
        mock_model.train.assert_called_once()
        call_args = mock_model.train.call_args
        train_df_arg = call_args[0][0]
        val_df_arg = call_args[0][1]

        assert len(train_df_arg) == small_n
        assert val_df_arg is None
        assert result is mock_model


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the full run pipeline."""

    def test_run_executes_full_pipeline(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame, tmp_path: Path,
    ) -> None:
        """Test run calls optimize, train_final, save, and returns pipeline result."""
        from energy_forecast.training.tft_trainer import TFTPipelineResult, TFTTrainer

        mock_settings.paths.models_dir = str(tmp_path)
        trainer = TFTTrainer(mock_settings)

        # Mock study
        mock_study = MagicMock()
        mock_study.best_value = 3.0
        mock_study.best_params = {"hidden_size": 16}
        mock_study.best_trial.number = 0
        mock_study.best_trial.user_attrs = {"avg_test_mape": 3.5}

        best_result = TFTTrainingResult(
            split_results=[_make_split_result(0, val_mape=3.0, test_mape=3.5)],
            avg_val_mape=3.0,
            avg_test_mape=3.5,
            std_val_mape=0.0,
        )

        mock_model = MagicMock()

        with (
            patch.object(trainer, "optimize", return_value=(mock_study, best_result)),
            patch.object(trainer, "train_final", return_value=mock_model),
        ):
            result = trainer.run(sample_df)

        assert isinstance(result, TFTPipelineResult)
        assert result.study is mock_study
        assert result.best_params == {"hidden_size": 16}
        assert result.training_result is best_result
        assert result.final_model is mock_model
        assert result.training_time_seconds >= 0

        # Verify save was called on the model
        mock_model.save.assert_called_once()
        save_path = mock_model.save.call_args[0][0]
        assert "tft" in str(save_path)

    def test_run_creates_model_directory(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame, tmp_path: Path,
    ) -> None:
        """Test run creates timestamped model directory."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        mock_settings.paths.models_dir = str(tmp_path)
        trainer = TFTTrainer(mock_settings)

        mock_study = MagicMock()
        mock_study.best_value = 3.0
        mock_study.best_params = {}
        mock_study.best_trial.number = 0
        mock_study.best_trial.user_attrs = {}

        best_result = TFTTrainingResult(
            split_results=[], avg_val_mape=3.0, avg_test_mape=3.5, std_val_mape=0.0,
        )
        mock_model = MagicMock()

        with (
            patch.object(trainer, "optimize", return_value=(mock_study, best_result)),
            patch.object(trainer, "train_final", return_value=mock_model),
        ):
            trainer.run(sample_df)

        # The tft subdirectory should exist
        tft_dir = tmp_path / "tft"
        assert tft_dir.exists()
        # There should be a timestamped subdirectory
        subdirs = list(tft_dir.iterdir())
        assert len(subdirs) == 1
        assert subdirs[0].name.startswith("tft_")

    def test_run_logs_to_tracker(
        self, mock_settings: MagicMock, sample_df: pd.DataFrame, tmp_path: Path,
    ) -> None:
        """Test run logs params and metrics to the experiment tracker."""
        from energy_forecast.training.tft_trainer import TFTTrainer

        mock_settings.paths.models_dir = str(tmp_path)

        mock_tracker = MagicMock()
        mock_tracker.start_run = MagicMock()
        # Make start_run a context manager that yields None
        mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

        trainer = TFTTrainer(mock_settings, tracker=mock_tracker)

        mock_study = MagicMock()
        mock_study.best_value = 2.5
        mock_study.best_params = {"hidden_size": 32}
        mock_study.best_trial.number = 0
        mock_study.best_trial.user_attrs = {"avg_test_mape": 3.0}

        split = _make_split_result(0, val_mape=2.5, test_mape=3.0)
        best_result = TFTTrainingResult(
            split_results=[split],
            avg_val_mape=2.5,
            avg_test_mape=3.0,
            std_val_mape=0.0,
        )
        mock_model = MagicMock()

        with (
            patch.object(trainer, "optimize", return_value=(mock_study, best_result)),
            patch.object(trainer, "train_final", return_value=mock_model),
        ):
            trainer.run(sample_df)

        # Verify tracker interactions
        assert mock_tracker.start_run.call_count == 2
        mock_tracker.log_params.assert_called_once_with({"hidden_size": 32})
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.log_split_metrics.assert_called_once()
        mock_tracker.log_tft_model.assert_called_once_with(mock_model, "tft_model")
