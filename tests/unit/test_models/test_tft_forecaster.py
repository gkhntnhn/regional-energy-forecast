"""Unit tests for TFTForecaster (NeuralForecast implementation)."""

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

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "consumption": 1000 + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + rng.standard_normal(n_samples) * 50,
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
            n_head=1,
            n_rnn_layers=1,
            dropout=0.1,
        ),
        training=TFTTrainingConfig(
            encoder_length=24,  # 1 day
            prediction_length=12,  # Half day
            max_steps=20,  # Minimal for fast tests
            windows_batch_size=64,
            learning_rate=0.01,
            early_stop_patience_steps=-1,  # Disabled for short tests
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


class TestNFFormatConversion:
    """Tests for NeuralForecast format conversion."""

    def test_to_nf_format_creates_required_columns(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test _to_nf_format adds unique_id, ds, y columns."""
        model = TFTForecaster(tft_config)
        nf_df = model._to_nf_format(sample_df, "consumption")

        assert "unique_id" in nf_df.columns
        assert "ds" in nf_df.columns
        assert "y" in nf_df.columns
        assert "consumption" not in nf_df.columns  # Renamed to y

    def test_to_nf_format_unique_id_is_constant(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test all rows have the same unique_id."""
        model = TFTForecaster(tft_config)
        nf_df = model._to_nf_format(sample_df, "consumption")

        assert (nf_df["unique_id"] == "uludag").all()

    def test_to_nf_format_preserves_covariates(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test covariates are preserved in NF format."""
        model = TFTForecaster(tft_config)
        nf_df = model._to_nf_format(sample_df, "consumption")

        assert "temperature_2m" in nf_df.columns
        assert "hour_cos" in nf_df.columns


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

        model.train(train_df, val_df, max_steps=20)

        assert model.is_fitted is True
        assert model._nf is not None

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

        metrics = model.train(train_df, val_df, max_steps=20)

        assert isinstance(metrics, dict)


class TestTFTForecasterPredict:
    """Tests for prediction."""

    @pytest.mark.slow
    def test_predict_returns_dataframe(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test predict returns DataFrame with consumption_mwh column."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:250]
        test_df = sample_df.iloc[250:]

        model.train(train_df, val_df, max_steps=20)
        predictions = model.predict(test_df)

        assert isinstance(predictions, pd.DataFrame)
        assert "consumption_mwh" in predictions.columns

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

        model.train(train_df, val_df, max_steps=20)
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
        """Test save creates NeuralForecast checkpoint and metadata files."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save(save_path)

            assert (save_path / "metadata.json").exists()
            # NeuralForecast creates its own checkpoint files
            nf_files = list(save_path.glob("*.ckpt")) + list(save_path.glob("*.pkl"))
            assert len(nf_files) > 0, "NeuralForecast should save checkpoint files"

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
    def test_load_restores_model(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test load restores model and makes it usable."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save(save_path)

            # Load into new model via instance method
            new_model = TFTForecaster(tft_config)
            new_model.load(save_path)

            assert new_model._quantiles == model._quantiles
            assert new_model.is_fitted is True

    @pytest.mark.slow
    def test_from_checkpoint_creates_functional_model(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test from_checkpoint returns a model that can predict."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:250]
        test_df = sample_df.iloc[250:]

        model.train(train_df, val_df, max_steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save(save_path)

            # Load via classmethod
            loaded = TFTForecaster.from_checkpoint(save_path)

            assert loaded.is_fitted is True
            assert loaded._quantiles == model._quantiles

            # Verify prediction works
            predictions = loaded.predict(test_df)
            assert isinstance(predictions, pd.DataFrame)
            assert "consumption_mwh" in predictions.columns
            assert len(predictions) > 0


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
