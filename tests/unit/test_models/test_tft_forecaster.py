"""Unit tests for TFTForecaster."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config.settings import (
    TFTArchitectureConfig,
    TFTConfig,
    TFTCovariatesConfig,
    TFTTrainingConfig,
)
from energy_forecast.models.tft import TFTForecaster


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample feature-engineered DataFrame for testing."""
    n_samples = 300  # Small for fast tests
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="h")

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "consumption": 1000 + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 50,
            "hour_sin": np.sin(np.arange(n_samples) % 24 * 2 * np.pi / 24),
            "hour_cos": np.cos(np.arange(n_samples) % 24 * 2 * np.pi / 24),
            "day_of_week_sin": np.sin(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "day_of_week_cos": np.cos(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "temperature_2m": 15 + 10 * np.sin((np.arange(n_samples) % 24 - 14) * np.pi / 12),
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
            attention_head_size=1,
            lstm_layers=1,
            dropout=0.1,
            hidden_continuous_size=8,
        ),
        training=TFTTrainingConfig(
            encoder_length=24,  # 1 day
            prediction_length=12,  # Half day
            batch_size=16,
            max_epochs=2,
            learning_rate=0.01,
            early_stop_patience=1,
            gradient_clip_val=0.1,
            random_seed=42,
            accelerator="cpu",
            num_workers=0,
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


class TestTFTForecasterInit:
    """Tests for TFTForecaster initialization."""

    def test_init_with_config(self, tft_config: TFTConfig) -> None:
        """Test initialization with config."""
        model = TFTForecaster(tft_config)
        assert model._tft_config == tft_config
        assert model._quantiles == [0.10, 0.50, 0.90]
        assert model.is_fitted is False

    def test_init_sets_quantiles(self, tft_config: TFTConfig) -> None:
        """Test initialization stores quantiles."""
        model = TFTForecaster(tft_config)
        assert 0.50 in model._quantiles


class TestTFTForecasterPrepare:
    """Tests for data preparation."""

    def test_prepare_dataframe_adds_columns(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test DataFrame preparation adds required columns."""
        model = TFTForecaster(tft_config)
        prepared = model._prepare_dataframe(sample_df, "consumption")

        assert "_time_idx" in prepared.columns
        assert "_group_id" in prepared.columns
        assert "timestamp" in prepared.columns

    def test_prepare_dataframe_time_idx_sequential(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test time index is sequential integers."""
        model = TFTForecaster(tft_config)
        prepared = model._prepare_dataframe(sample_df, "consumption")

        expected = list(range(len(sample_df)))
        assert prepared["_time_idx"].tolist() == expected

    def test_prepare_dataframe_rejects_non_datetime_index(
        self,
        tft_config: TFTConfig,
    ) -> None:
        """Test preparation rejects non-DatetimeIndex."""
        model = TFTForecaster(tft_config)
        bad_df = pd.DataFrame({"consumption": [1, 2, 3]})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            model._prepare_dataframe(bad_df, "consumption")


class TestTFTForecasterTrain:
    """Tests for model training."""

    @pytest.mark.slow
    def test_train_sets_fitted(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test training sets is_fitted to True."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_epochs=2)

        assert model.is_fitted is True
        assert model._model is not None

    @pytest.mark.slow
    def test_train_returns_metrics(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test training returns metrics dict."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        metrics = model.train(train_df, val_df, max_epochs=2)

        assert isinstance(metrics, dict)


class TestTFTForecasterPredict:
    """Tests for prediction."""

    @pytest.mark.slow
    def test_predict_returns_dataframe(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test predict returns DataFrame with yhat column."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:250]
        test_df = sample_df.iloc[250:]

        model.train(train_df, val_df, max_epochs=2)
        predictions = model.predict(test_df)

        assert isinstance(predictions, pd.DataFrame)
        assert "yhat" in predictions.columns

    @pytest.mark.slow
    def test_predict_stores_all_quantiles(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test predict stores all quantile predictions."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:250]
        test_df = sample_df.iloc[250:]

        model.train(train_df, val_df, max_epochs=2)
        model.predict(test_df)

        quantiles = model.get_quantile_predictions()
        assert 0.10 in quantiles
        assert 0.50 in quantiles
        assert 0.90 in quantiles

    def test_predict_raises_if_not_fitted(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test predict raises if model not fitted."""
        model = TFTForecaster(tft_config)

        with pytest.raises(RuntimeError, match="trained"):
            model.predict(sample_df)


class TestTFTForecasterSaveLoad:
    """Tests for model serialization."""

    @pytest.mark.slow
    def test_save_creates_files(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test save creates checkpoint and metadata files."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save(save_path)

            assert (save_path / "tft_model.ckpt").exists()
            assert (save_path / "metadata.json").exists()
            assert (save_path / "dataset_params.json").exists()

    def test_save_raises_if_not_fitted(
        self,
        tft_config: TFTConfig,
    ) -> None:
        """Test save raises if model not fitted."""
        model = TFTForecaster(tft_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="unfitted"):
                model.save(Path(tmpdir))

    @pytest.mark.slow
    def test_load_restores_metadata(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test load restores metadata and quantiles."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save(save_path)

            # Load into new model
            new_model = TFTForecaster(tft_config)
            new_model.load(save_path)

            assert new_model._quantiles == model._quantiles
            assert new_model._dataset_params == model._dataset_params


class TestTFTForecasterQuantiles:
    """Tests for quantile prediction access."""

    def test_get_quantile_predictions_raises_if_no_predictions(
        self,
        tft_config: TFTConfig,
    ) -> None:
        """Test get_quantile_predictions raises without prior predict."""
        model = TFTForecaster(tft_config)

        with pytest.raises(RuntimeError, match="No predictions"):
            model.get_quantile_predictions()
