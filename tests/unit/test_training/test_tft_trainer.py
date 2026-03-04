"""Unit tests for TFTTrainer (NeuralForecast implementation)."""

from __future__ import annotations

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


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample feature-engineered DataFrame for testing."""
    n_samples = 24 * 30 * 6  # 6 months for TSCV splits
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="h")

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "consumption": 1000 + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + rng.standard_normal(n_samples) * 50,
            "hour_sin": np.sin(np.arange(n_samples) % 24 * 2 * np.pi / 24),
            "hour_cos": np.cos(np.arange(n_samples) % 24 * 2 * np.pi / 24),
            "day_of_week_sin": np.sin(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "day_of_week_cos": np.cos(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "month_sin": np.zeros(n_samples),
            "month_cos": np.ones(n_samples),
            "is_holiday": np.zeros(n_samples, dtype=int),
            "is_weekend": (np.arange(n_samples) // 24 % 7 >= 5).astype(int),
            "is_ramadan": np.zeros(n_samples, dtype=int),
            "sol_elevation": np.maximum(0, 45 * np.sin((np.arange(n_samples) % 24 - 6) * np.pi / 12)),
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
