"""Unit tests for TSMixerxForecaster (NeuralForecast implementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import (
    TSMixerxArchitectureConfig,
    TSMixerxConfig,
    TSMixerxCovariatesConfig,
    TSMixerxTrainingConfig,
)
from energy_forecast.models.tsmixerx import TSMixerxForecaster


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample feature-engineered DataFrame for testing."""
    n_samples = 300
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="h")

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "consumption": (
                1000
                + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)
                + rng.standard_normal(n_samples) * 50
            ),
            "apparent_temperature": 15 + 10 * np.sin((np.arange(n_samples) % 24 - 14) * np.pi / 12),
            "day_of_week_sin": np.sin(np.arange(n_samples) // 24 % 7 * 2 * np.pi / 7),
            "wth_hdd": rng.random(n_samples) * 5,
        },
        index=dates,
    )
    return df


@pytest.fixture
def tsmixerx_config() -> TSMixerxConfig:
    """Create minimal TSMixerx config for fast tests."""
    return TSMixerxConfig(
        architecture=TSMixerxArchitectureConfig(
            n_block=1,
            ff_dim=16,
            dropout=0.0,
            input_size=24,
            revin=True,
        ),
        training=TSMixerxTrainingConfig(
            prediction_length=12,
            max_steps=20,
            windows_batch_size=32,
            step_size=6,
            learning_rate=0.01,
            early_stop_patience_steps=-1,
            val_check_steps=10,
            random_seed=42,
            accelerator="cpu",
            num_workers=0,
            scaler_type="robust",
        ),
        covariates=TSMixerxCovariatesConfig(
            futr_exog=[
                "apparent_temperature",
                "day_of_week_sin",
                "wth_hdd",
            ],
            hist_exog=[],
        ),
    )


class TestTSMixerxForecasterInit:
    """Tests for TSMixerxForecaster initialization."""

    def test_init_with_config(self, tsmixerx_config: TSMixerxConfig) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        assert model._tsmixerx_config == tsmixerx_config
        assert model.is_fitted is False

    def test_no_quantiles_attribute(self, tsmixerx_config: TSMixerxConfig) -> None:
        """TSMixerx is point forecast — no quantile attributes."""
        model = TSMixerxForecaster(tsmixerx_config)
        assert not hasattr(model, "_quantiles")


class TestNFFormatConversion:
    """Tests for NeuralForecast format conversion."""

    def test_to_nf_format_creates_required_columns(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        nf_df = model._to_nf_format(sample_df, "consumption")

        assert "unique_id" in nf_df.columns
        assert "ds" in nf_df.columns
        assert "y" in nf_df.columns
        assert "consumption" not in nf_df.columns

    def test_to_nf_format_unique_id_is_constant(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        nf_df = model._to_nf_format(sample_df, "consumption")

        assert (nf_df["unique_id"] == "uludag").all()

    def test_to_nf_format_preserves_covariates(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        nf_df = model._to_nf_format(sample_df, "consumption")

        assert "apparent_temperature" in nf_df.columns
        assert "day_of_week_sin" in nf_df.columns


class TestTSMixerxForecasterTrain:
    """Tests for model training."""

    @pytest.mark.slow
    def test_train_sets_fitted(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_steps=20)

        assert model.is_fitted is True
        assert model._nf is not None

    @pytest.mark.slow
    def test_train_returns_metrics(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        metrics = model.train(train_df, val_df, max_steps=20)

        assert isinstance(metrics, dict)


class TestTSMixerxForecasterPredict:
    """Tests for prediction."""

    def test_predict_without_train_raises(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        with pytest.raises(RuntimeError, match="must be trained"):
            model.predict(sample_df)

    @pytest.mark.slow
    def test_predict_returns_dataframe(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:250]
        test_df = sample_df.iloc[250:]

        model.train(train_df, val_df, max_steps=20)
        predictions = model.predict(test_df)

        assert isinstance(predictions, pd.DataFrame)
        assert "consumption_mwh" in predictions.columns


class TestTSMixerxSaveLoad:
    """Tests for save/load."""

    @pytest.mark.slow
    def test_save_creates_metadata(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
        tmp_path: pd.Timestamp,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]
        model.train(train_df, val_df, max_steps=20)

        save_dir = tmp_path / "tsmixerx_model"  # type: ignore[operator]
        model.save(save_dir)

        assert (save_dir / "metadata.json").exists()

    @pytest.mark.slow
    def test_from_checkpoint_creates_functional_model(
        self,
        tsmixerx_config: TSMixerxConfig,
        sample_df: pd.DataFrame,
        tmp_path: pd.Timestamp,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:250]
        test_df = sample_df.iloc[250:]
        model.train(train_df, val_df, max_steps=20)

        save_dir = tmp_path / "tsmixerx_model"  # type: ignore[operator]
        model.save(save_dir)

        loaded = TSMixerxForecaster.from_checkpoint(save_dir)
        assert loaded.is_fitted is True

        predictions = loaded.predict(test_df)
        assert "consumption_mwh" in predictions.columns

    def test_save_unfitted_raises(
        self,
        tsmixerx_config: TSMixerxConfig,
        tmp_path: pd.Timestamp,
    ) -> None:
        model = TSMixerxForecaster(tsmixerx_config)
        with pytest.raises(ValueError, match="Cannot save unfitted"):
            model.save(tmp_path / "bad_model")  # type: ignore[operator]
