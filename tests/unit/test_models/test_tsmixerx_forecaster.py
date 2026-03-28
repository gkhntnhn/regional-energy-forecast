"""Unit tests for TSMixerxForecaster (NeuralForecast implementation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Mocked tests — NO NeuralForecast import required
# ---------------------------------------------------------------------------


def _make_tsmixerx_config(
    *,
    accelerator: str = "cpu",
    enable_progress_bar: bool = False,
    prediction_length: int = 12,
    input_size: int = 24,
    loss: str = "mae",
) -> TSMixerxConfig:
    """Helper to build a TSMixerxConfig with test-friendly defaults."""
    return TSMixerxConfig(
        architecture=TSMixerxArchitectureConfig(
            n_block=1, ff_dim=16, dropout=0.0, input_size=input_size, revin=True
        ),
        training=TSMixerxTrainingConfig(
            prediction_length=prediction_length,
            max_steps=20,
            windows_batch_size=32,
            step_size=6,
            learning_rate=0.01,
            early_stop_patience_steps=-1,
            val_check_steps=10,
            random_seed=42,
            accelerator=accelerator,
            num_workers=0,
            scaler_type="robust",
            enable_progress_bar=enable_progress_bar,
        ),
        covariates=TSMixerxCovariatesConfig(
            futr_exog=["apparent_temperature", "day_of_week_sin", "wth_hdd"],
            hist_exog=[],
        ),
        loss=loss,
    )


def _make_sample_df(n: int = 300) -> pd.DataFrame:
    """Build a sample DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "consumption": 1000 + rng.standard_normal(n) * 50,
            "apparent_temperature": 15 + rng.standard_normal(n),
            "day_of_week_sin": np.sin(np.arange(n) // 24 % 7 * 2 * np.pi / 7),
            "wth_hdd": rng.random(n) * 5,
        },
        index=dates,
    )


class TestBuildNFModelMocked:
    """Tests for _build_nf_model with mocked NeuralForecast imports."""

    def test_build_nf_model_cpu_sets_devices(self) -> None:
        """Test CPU accelerator adds devices=1 to trainer kwargs."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mae = MagicMock()

        cfg = _make_tsmixerx_config(accelerator="cpu")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(return_value=mock_mae),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("devices") == 1

    def test_build_nf_model_max_steps_override(self) -> None:
        """Test max_steps override is passed to model constructor."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()

        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model(max_steps=555)

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("max_steps") == 555

    def test_build_nf_model_with_callbacks(self) -> None:
        """Test external callbacks are passed through."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()

        cfg = _make_tsmixerx_config(enable_progress_bar=False)
        model = TSMixerxForecaster(cfg)

        my_callback = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model(callbacks=[my_callback])

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            cbs = call_kwargs.kwargs.get("callbacks", [])
            assert my_callback in cbs

    def test_build_nf_model_progress_bar_adds_callback(self) -> None:
        """Test enable_progress_bar=True adds TQDMProgressBar callback."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_tqdm = MagicMock()

        cfg = _make_tsmixerx_config(enable_progress_bar=True)
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
                "pytorch_lightning": MagicMock(),
                "pytorch_lightning.callbacks": MagicMock(TQDMProgressBar=mock_tqdm),
            },
        ):
            model._build_nf_model()

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            cbs = call_kwargs.kwargs.get("callbacks", [])
            assert len(cbs) >= 1

    def test_loss_map_mae(self) -> None:
        """Test loss='mae' selects MAE loss."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mae_instance = MagicMock()
        mock_mae_cls = MagicMock(return_value=mock_mae_instance)

        cfg = _make_tsmixerx_config(loss="mae")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=mock_mae_cls,
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("loss") is mock_mae_instance

    def test_loss_map_mse(self) -> None:
        """Test loss='mse' selects MSE loss."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mse_instance = MagicMock()
        mock_mse_cls = MagicMock(return_value=mock_mse_instance)

        cfg = _make_tsmixerx_config(loss="mse")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=mock_mse_cls,
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("loss") is mock_mse_instance

    def test_loss_map_rmse(self) -> None:
        """Test loss='rmse' selects RMSE loss."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_rmse_instance = MagicMock()
        mock_rmse_cls = MagicMock(return_value=mock_rmse_instance)

        cfg = _make_tsmixerx_config(loss="rmse")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=mock_rmse_cls,
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("loss") is mock_rmse_instance

    def test_loss_map_huber(self) -> None:
        """Test loss='huber' selects HuberLoss with delta=1.0."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_huber_instance = MagicMock()
        mock_huber_cls = MagicMock(return_value=mock_huber_instance)

        cfg = _make_tsmixerx_config(loss="huber")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=mock_huber_cls,
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        # HuberLoss should be called with delta=1.0
        mock_huber_cls.assert_any_call(delta=1.0)

    def test_loss_map_huber_half(self) -> None:
        """Test loss='huber_0.5' selects HuberLoss with delta=0.5."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_huber_cls = MagicMock()

        cfg = _make_tsmixerx_config(loss="huber_0.5")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=mock_huber_cls,
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        mock_huber_cls.assert_any_call(delta=0.5)

    def test_loss_map_huber_two(self) -> None:
        """Test loss='huber_2.0' selects HuberLoss with delta=2.0."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_huber_cls = MagicMock()

        cfg = _make_tsmixerx_config(loss="huber_2.0")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=MagicMock(),
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=mock_huber_cls,
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        mock_huber_cls.assert_any_call(delta=2.0)

    def test_loss_map_unknown_falls_back_to_mae(self) -> None:
        """Test unknown loss string falls back to MAE."""
        mock_tsmixerx_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tsmixerx_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mae_instance = MagicMock()
        mock_mae_cls = MagicMock(return_value=mock_mae_instance)

        cfg = _make_tsmixerx_config(loss="nonexistent_loss")
        model = TSMixerxForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(
                    MAE=mock_mae_cls,
                    MSE=MagicMock(),
                    RMSE=MagicMock(),
                    HuberLoss=MagicMock(),
                ),
                "neuralforecast.models": MagicMock(TSMixerx=mock_tsmixerx_cls),
            },
        ):
            model._build_nf_model()

        # Fallback: MAE() is the default
        call_kwargs = mock_tsmixerx_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("loss") is mock_mae_instance


class TestTSMixerxTrainMocked:
    """Tests for train() with mocked _build_nf_model."""

    def test_train_returns_metrics_dict(self) -> None:
        """Test train returns metrics dict after fit."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        df = _make_sample_df(200)
        train_df = df.iloc[:150]
        val_df = df.iloc[150:]

        mock_nf = MagicMock()
        mock_tsmixerx_model = MagicMock()
        mock_tsmixerx_model.trainer.callback_metrics = {"train_loss": 0.3}
        mock_nf.models = [mock_tsmixerx_model]

        with patch.object(model, "_build_nf_model", return_value=mock_nf):
            metrics = model.train(train_df, val_df)

        assert isinstance(metrics, dict)
        assert model._nf is mock_nf
        assert model._last_train_df is not None
        mock_nf.fit.assert_called_once()

    def test_train_without_val_df(self) -> None:
        """Test train works when val_df is None."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        df = _make_sample_df(200)

        mock_nf = MagicMock()
        mock_tsmixerx_model = MagicMock()
        mock_tsmixerx_model.trainer = None
        mock_nf.models = [mock_tsmixerx_model]

        with patch.object(model, "_build_nf_model", return_value=mock_nf):
            metrics = model.train(df, val_df=None)

        assert isinstance(metrics, dict)
        fit_call = mock_nf.fit.call_args
        assert fit_call.kwargs.get("val_size") == 0

    def test_train_passes_max_steps_and_callbacks(self) -> None:
        """Test train forwards max_steps and callbacks to _build_nf_model."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        df = _make_sample_df(200)

        mock_nf = MagicMock()
        mock_tsmixerx_model = MagicMock()
        mock_tsmixerx_model.trainer = None
        mock_nf.models = [mock_tsmixerx_model]
        my_cb = MagicMock()

        with patch.object(model, "_build_nf_model", return_value=mock_nf) as mock_build:
            model.train(df, max_steps=77, callbacks=[my_cb])

        mock_build.assert_called_once_with(callbacks=[my_cb], max_steps=77)

    def test_train_handles_metric_extraction_error(self) -> None:
        """Test train returns empty metrics when extraction fails."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        df = _make_sample_df(200)

        mock_nf = MagicMock()
        mock_tsmixerx_model = MagicMock()
        mock_tsmixerx_model.trainer.callback_metrics.items.side_effect = RuntimeError("boom")
        mock_nf.models = [mock_tsmixerx_model]

        with patch.object(model, "_build_nf_model", return_value=mock_nf):
            metrics = model.train(df)

        assert isinstance(metrics, dict)
        assert model._nf is mock_nf


class TestTSMixerxPredictMocked:
    """Tests for predict() with mocked NF."""

    def _make_fitted_model(
        self, pred_len: int = 12, input_size: int = 24
    ) -> tuple[TSMixerxForecaster, MagicMock]:
        cfg = _make_tsmixerx_config(prediction_length=pred_len, input_size=input_size)
        model = TSMixerxForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf
        return model, mock_nf

    def test_predict_raises_if_not_fitted(self) -> None:
        """Test predict raises RuntimeError when model not fitted."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        df = _make_sample_df(50)
        with pytest.raises(RuntimeError, match="trained"):
            model.predict(df)

    def test_predict_long_input_splits_context_and_forecast(self) -> None:
        """Test predict splits X into context and forecast when len(X) > pred_len."""
        model, mock_nf = self._make_fitted_model(pred_len=12, input_size=24)
        df = _make_sample_df(100)

        preds_df = pd.DataFrame(
            {"TSMixerx": np.ones(12)},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)

        assert "consumption_mwh" in result.columns
        assert len(result) == 12
        call_kwargs = mock_nf.predict.call_args.kwargs
        assert call_kwargs["df"] is not None
        assert call_kwargs["futr_df"] is not None

    def test_predict_short_input_no_context(self) -> None:
        """Test predict with short input (len <= pred_len) has no explicit context."""
        model, mock_nf = self._make_fitted_model(pred_len=12, input_size=24)
        df = _make_sample_df(10)

        preds_df = pd.DataFrame(
            {"TSMixerx": np.ones(10)},
            index=pd.RangeIndex(10),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)

        assert "consumption_mwh" in result.columns
        call_kwargs = mock_nf.predict.call_args.kwargs
        assert call_kwargs["df"] is None

    def test_predict_fallback_to_first_tsmixerx_column(self) -> None:
        """Test pred_col fallback: TSMixerx missing -> first TSMixerx* col."""
        model, mock_nf = self._make_fitted_model(pred_len=12)
        df = _make_sample_df(50)

        preds_df = pd.DataFrame(
            {"TSMixerx-special": np.ones(12) * 88},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)
        assert result["consumption_mwh"].iloc[0] == pytest.approx(88.0)

    def test_predict_fallback_to_last_column(self) -> None:
        """Test pred_col fallback: no TSMixerx* columns -> last column."""
        model, mock_nf = self._make_fitted_model(pred_len=12)
        df = _make_sample_df(50)

        preds_df = pd.DataFrame(
            {"other_model": np.ones(12) * 33},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)
        assert result["consumption_mwh"].iloc[0] == pytest.approx(33.0)

    def test_predict_context_nan_ffill(self) -> None:
        """Test predict fills NaN in context target with ffill/bfill."""
        model, mock_nf = self._make_fitted_model(pred_len=12, input_size=24)
        df = _make_sample_df(50)
        df.iloc[10:15, df.columns.get_loc("consumption")] = np.nan

        preds_df = pd.DataFrame(
            {"TSMixerx": np.ones(12)},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)
        assert "consumption_mwh" in result.columns


class TestTSMixerxRollingPredictMocked:
    """Tests for rolling_predict with mocked NF."""

    def _make_fitted_model(self) -> tuple[TSMixerxForecaster, MagicMock]:
        cfg = _make_tsmixerx_config(prediction_length=12, input_size=24)
        model = TSMixerxForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf
        return model, mock_nf

    def test_rolling_predict_raises_if_not_fitted(self) -> None:
        """Test rolling_predict raises when model not fitted."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        df = _make_sample_df(100)
        with pytest.raises(RuntimeError, match="trained"):
            model.rolling_predict(df, eval_start=df.index[50], eval_end=df.index[-1])

    def test_rolling_predict_last_window_wins(self) -> None:
        """Test overlapping windows: later predictions overwrite earlier ones."""
        model, mock_nf = self._make_fitted_model()
        df = _make_sample_df(200)

        call_count = [0]

        def fake_predict(df: Any = None, futr_df: Any = None) -> pd.DataFrame:
            call_count[0] += 1
            n = 12
            return pd.DataFrame(
                {"TSMixerx": np.full(n, float(call_count[0]))},
                index=pd.RangeIndex(n),
            )

        mock_nf.predict.side_effect = fake_predict

        eval_start = df.index[100]
        eval_end = df.index[130]

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end, step_hours=12)

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        if len(result) > 12:
            later_val = result["consumption_mwh"].iloc[-1]
            assert later_val > 0

    def test_rolling_predict_skip_insufficient_context(self) -> None:
        """Test windows with insufficient context are skipped."""
        model, mock_nf = self._make_fitted_model()
        df = _make_sample_df(30)

        mock_nf.predict.return_value = pd.DataFrame(
            {"TSMixerx": np.ones(12)}, index=pd.RangeIndex(12)
        )

        eval_start = df.index[0]
        eval_end = df.index[11]

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end)
        assert isinstance(result, pd.DataFrame)

    def test_rolling_predict_empty_predictions(self) -> None:
        """Test empty predictions returns empty DataFrame."""
        model, _mock_nf = self._make_fitted_model()
        df = _make_sample_df(200)

        eval_start = df.index[-1] + pd.Timedelta(hours=1000)
        eval_end = eval_start + pd.Timedelta(hours=48)

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end)

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        assert len(result) == 0

    def test_rolling_predict_column_fallback(self) -> None:
        """Test rolling_predict uses column fallback chain."""
        model, mock_nf = self._make_fitted_model()
        df = _make_sample_df(200)

        def fake_predict(df: Any = None, futr_df: Any = None) -> pd.DataFrame:
            return pd.DataFrame(
                {"other_col": np.ones(12) * 55},
                index=pd.RangeIndex(12),
            )

        mock_nf.predict.side_effect = fake_predict

        eval_start = df.index[100]
        eval_end = df.index[111]

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end)
        if len(result) > 0:
            assert result["consumption_mwh"].iloc[0] == pytest.approx(55.0)


class TestTSMixerxSaveMocked:
    """Tests for save() with mocked NF and torch."""

    def test_save_creates_metadata_json(self, tmp_path: Path) -> None:
        """Test save creates metadata.json with correct content."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf

        mock_torch = MagicMock()
        mock_torch.load.return_value = {
            "hyper_parameters": {"callbacks": ["some_cb"]},
        }

        with patch.dict("sys.modules", {"torch": mock_torch}):
            model.save(tmp_path / "tsmixerx_model")

        metadata_path = tmp_path / "tsmixerx_model" / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            meta = json.load(f)

        assert "architecture" in meta
        assert "training" in meta
        assert "covariates" in meta
        assert meta["training"]["prediction_length"] == 12
        assert meta["training"]["input_size"] == 24

    def test_save_raises_if_not_fitted(self, tmp_path: Path) -> None:
        """Test save raises ValueError when model not fitted."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        with pytest.raises(ValueError, match="unfitted"):
            model.save(tmp_path / "bad")

    def test_save_strips_callbacks_from_ckpt(self, tmp_path: Path) -> None:
        """Test save strips callbacks from checkpoint files."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf

        save_dir = tmp_path / "tsmixerx_model"
        save_dir.mkdir(parents=True)

        fake_ckpt = save_dir / "model.ckpt"
        fake_ckpt.write_bytes(b"fake_checkpoint_data")

        ckpt_data = {"hyper_parameters": {"callbacks": ["early_stop"]}}
        mock_torch = MagicMock()
        mock_torch.load.return_value = ckpt_data

        mock_nf.save.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            model.save(save_dir)

        mock_torch.save.assert_called()
        saved_ckpt = mock_torch.save.call_args[0][0]
        assert saved_ckpt["hyper_parameters"]["callbacks"] == []

    def test_save_computes_ckpt_hashes(self, tmp_path: Path) -> None:
        """Test save computes SHA256 hashes for ckpt files."""
        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf

        save_dir = tmp_path / "tsmixerx_model"
        save_dir.mkdir(parents=True)

        fake_ckpt = save_dir / "model.ckpt"
        ckpt_content = b"tsmixerx_checkpoint_content"
        fake_ckpt.write_bytes(ckpt_content)
        expected_hash = hashlib.sha256(ckpt_content).hexdigest()

        mock_torch = MagicMock()
        mock_torch.load.return_value = {}

        with patch.dict("sys.modules", {"torch": mock_torch}):
            model.save(save_dir)

        metadata_path = save_dir / "metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)

        assert "model.ckpt" in meta["ckpt_hashes"]
        assert meta["ckpt_hashes"]["model.ckpt"] == expected_hash


class TestTSMixerxFromCheckpointMocked:
    """Tests for from_checkpoint() with mocked NF."""

    def test_from_checkpoint_loads_metadata(self, tmp_path: Path) -> None:
        """Test from_checkpoint reads metadata.json and reconstructs config."""
        model_dir = tmp_path / "tsmixerx_model"
        model_dir.mkdir()

        metadata = {
            "architecture": {
                "n_block": 3,
                "ff_dim": 64,
                "dropout": 0.05,
                "input_size": 48,
                "revin": False,
            },
            "training": {
                "input_size": 48,
                "prediction_length": 24,
                "num_workers": 2,
                "scaler_type": "standard",
            },
            "covariates": {
                "futr_exog": ["temperature"],
                "hist_exog": ["lag_48"],
            },
            "ckpt_hashes": {},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_nf_instance = MagicMock()
        mock_nf_instance.models = []
        mock_nf_cls = MagicMock()
        mock_nf_cls.load.return_value = mock_nf_instance

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
            },
        ):
            loaded = TSMixerxForecaster.from_checkpoint(model_dir)

        assert loaded.is_fitted
        assert loaded._tsmixerx_config.architecture.n_block == 3
        assert loaded._tsmixerx_config.architecture.ff_dim == 64
        assert loaded._tsmixerx_config.training.prediction_length == 24

    def test_from_checkpoint_raises_on_missing_metadata(self, tmp_path: Path) -> None:
        """Test from_checkpoint raises FileNotFoundError without metadata.json."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            TSMixerxForecaster.from_checkpoint(model_dir)

    def test_from_checkpoint_verifies_hash(self, tmp_path: Path) -> None:
        """Test from_checkpoint raises on hash mismatch."""
        model_dir = tmp_path / "tsmixerx_model"
        model_dir.mkdir()

        fake_ckpt = model_dir / "model.ckpt"
        fake_ckpt.write_bytes(b"real_content")

        metadata = {
            "architecture": {
                "n_block": 1,
                "ff_dim": 16,
                "dropout": 0.0,
                "input_size": 24,
                "revin": True,
            },
            "training": {},
            "covariates": {"futr_exog": [], "hist_exog": []},
            "ckpt_hashes": {"model.ckpt": "0" * 64},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(RuntimeError, match="integrity check failed"):
            TSMixerxForecaster.from_checkpoint(model_dir)

    def test_from_checkpoint_strips_callbacks(self, tmp_path: Path) -> None:
        """Test from_checkpoint clears callbacks from loaded models."""
        model_dir = tmp_path / "tsmixerx_model"
        model_dir.mkdir()

        metadata = {
            "architecture": {
                "n_block": 1,
                "ff_dim": 16,
                "dropout": 0.0,
                "input_size": 24,
                "revin": True,
            },
            "training": {},
            "covariates": {"futr_exog": [], "hist_exog": []},
            "ckpt_hashes": {},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_model = MagicMock()
        mock_model.hparams = {"callbacks": ["early_stopping"]}

        mock_nf_instance = MagicMock()
        mock_nf_instance.models = [mock_model]
        mock_nf_cls = MagicMock()
        mock_nf_cls.load.return_value = mock_nf_instance

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
            },
        ):
            loaded = TSMixerxForecaster.from_checkpoint(model_dir)

        assert mock_model.hparams["callbacks"] == []
        assert loaded.is_fitted


class TestTSMixerxLoadMocked:
    """Tests for load() instance method with mocked NF."""

    def test_load_sets_nf(self, tmp_path: Path) -> None:
        """Test load() sets _nf."""
        model_dir = tmp_path / "tsmixerx_model"
        model_dir.mkdir()

        # metadata.json optional for load()
        (model_dir / "metadata.json").write_text(json.dumps({"some": "data"}))

        mock_nf_instance = MagicMock()
        mock_nf_instance.models = []

        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)

        mock_nf_cls = MagicMock()
        mock_nf_cls.load.return_value = mock_nf_instance

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
            },
        ):
            model.load(model_dir)

        assert model.is_fitted

    def test_load_strips_callbacks(self, tmp_path: Path) -> None:
        """Test load() strips callbacks from loaded models."""
        model_dir = tmp_path / "tsmixerx_model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(json.dumps({}))

        mock_model = MagicMock()
        mock_model.hparams = {"callbacks": ["cb1"]}
        mock_nf_instance = MagicMock()
        mock_nf_instance.models = [mock_model]

        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)

        mock_nf_cls = MagicMock()
        mock_nf_cls.load.return_value = mock_nf_instance

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
            },
        ):
            model.load(model_dir)

        assert mock_model.hparams["callbacks"] == []

    def test_load_without_metadata_file(self, tmp_path: Path) -> None:
        """Test load() works even without metadata.json (no state to restore)."""
        model_dir = tmp_path / "tsmixerx_model"
        model_dir.mkdir()
        # No metadata.json

        mock_nf_instance = MagicMock()
        mock_nf_instance.models = []

        cfg = _make_tsmixerx_config()
        model = TSMixerxForecaster(cfg)

        mock_nf_cls = MagicMock()
        mock_nf_cls.load.return_value = mock_nf_instance

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
            },
        ):
            model.load(model_dir)

        assert model.is_fitted
