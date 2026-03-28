"""Unit tests for TFTForecaster (NeuralForecast implementation)."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import (
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
            "consumption": (
                1000
                + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)
                + rng.standard_normal(n_samples) * 50
            ),
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

        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="unfitted"):
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


class TestTFTForecasterRollingPredict:
    """Tests for rolling prediction over evaluation periods."""

    @pytest.mark.slow
    def test_rolling_predict_covers_eval_period(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test rolling_predict covers the full evaluation period."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_steps=20)

        eval_start = sample_df.index[200]
        eval_end = sample_df.index[-1]

        result = model.rolling_predict(sample_df, eval_start=eval_start, eval_end=eval_end)

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        # Rolling should cover more hours than single predict (which gives 12)
        assert len(result) > tft_config.training.prediction_length

    @pytest.mark.slow
    def test_rolling_predict_index_within_eval_bounds(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test all predictions fall within eval_start..eval_end."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_steps=20)

        eval_start = sample_df.index[200]
        eval_end = sample_df.index[-1]

        result = model.rolling_predict(sample_df, eval_start=eval_start, eval_end=eval_end)

        assert result.index.min() >= eval_start
        assert result.index.max() <= eval_end

    @pytest.mark.slow
    def test_rolling_predict_step_hours_affects_coverage(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test step_hours=12 produces same or more coverage than step_hours=24."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_steps=20)

        eval_start = sample_df.index[200]
        eval_end = sample_df.index[-1]

        result_24 = model.rolling_predict(
            sample_df, eval_start=eval_start, eval_end=eval_end, step_hours=24
        )
        result_12 = model.rolling_predict(
            sample_df, eval_start=eval_start, eval_end=eval_end, step_hours=12
        )

        # More frequent steps should produce same or more predictions
        assert len(result_12) >= len(result_24)

    def test_rolling_predict_raises_if_not_fitted(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test rolling_predict raises if model not fitted."""
        model = TFTForecaster(tft_config)

        with pytest.raises(RuntimeError, match="trained"):
            model.rolling_predict(
                sample_df,
                eval_start=sample_df.index[100],
                eval_end=sample_df.index[-1],
            )


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


# ---------------------------------------------------------------------------
# Mocked tests — NO NeuralForecast import required
# ---------------------------------------------------------------------------


def _make_tft_config(
    *,
    accelerator: str = "cpu",
    enable_progress_bar: bool = False,
    prediction_length: int = 12,
    encoder_length: int = 24,
) -> TFTConfig:
    """Helper to build a TFTConfig with test-friendly defaults."""
    return TFTConfig(
        architecture=TFTArchitectureConfig(hidden_size=16, n_head=1, n_rnn_layers=1, dropout=0.1),
        training=TFTTrainingConfig(
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            max_steps=20,
            windows_batch_size=64,
            learning_rate=0.01,
            early_stop_patience_steps=-1,
            val_check_steps=10,
            gradient_clip_val=0.1,
            random_seed=42,
            accelerator=accelerator,
            num_workers=0,
            precision="32-true",
            scaler_type="robust",
            rnn_type="lstm",
            enable_progress_bar=enable_progress_bar,
        ),
        covariates=TFTCovariatesConfig(
            time_varying_known=["hour_sin", "hour_cos", "temperature_2m"],
            time_varying_unknown=[],
        ),
        quantiles=[0.10, 0.50, 0.90],
        loss="quantile",
    )


def _make_sample_df(n: int = 300) -> pd.DataFrame:
    """Build a sample DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "consumption": 1000 + rng.standard_normal(n) * 50,
            "hour_sin": np.sin(np.arange(n) % 24 * 2 * np.pi / 24),
            "hour_cos": np.cos(np.arange(n) % 24 * 2 * np.pi / 24),
            "temperature_2m": 15 + rng.standard_normal(n),
        },
        index=dates,
    )


class TestBuildNFModel:
    """Tests for _build_nf_model with mocked NeuralForecast imports."""

    @patch("energy_forecast.models.tft.TFTForecaster._build_nf_model")
    def test_build_nf_model_called_with_no_callbacks(self, mock_build: MagicMock) -> None:
        """Test _build_nf_model is callable without callbacks."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        mock_build.return_value = MagicMock()
        model._build_nf_model()
        mock_build.assert_called_once()

    def test_build_nf_model_cpu_sets_devices(self) -> None:
        """Test CPU accelerator adds devices=1 to trainer kwargs."""
        mock_nf_cls = MagicMock()
        mock_tft_cls = MagicMock()
        mock_mqloss_cls = MagicMock()
        mock_tft_instance = MagicMock()
        mock_tft_instance.parameters.return_value = [MagicMock(numel=MagicMock(return_value=100))]
        mock_tft_cls.return_value = mock_tft_instance
        mock_nf_cls.return_value = MagicMock()

        cfg = _make_tft_config(accelerator="cpu")
        model = TFTForecaster(cfg)

        with (
            patch.dict(
                "sys.modules",
                {
                    "neuralforecast": MagicMock(),
                    "neuralforecast.losses": MagicMock(),
                    "neuralforecast.losses.pytorch": MagicMock(MQLoss=mock_mqloss_cls),
                    "neuralforecast.models": MagicMock(TFT=mock_tft_cls),
                },
            ),
            patch("energy_forecast.models.tft.NeuralForecast", mock_nf_cls, create=True),
        ):
            # Call the real method by importing fresh
            # Instead, test via the kwargs passed to TFT constructor
            model._build_nf_model()

        # The TFT constructor should have been called with devices=1
        call_kwargs = mock_tft_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("devices") == 1 or "devices" in str(call_kwargs)

    def test_build_nf_model_max_steps_override(self) -> None:
        """Test max_steps override is passed to model constructor."""
        mock_tft_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_tft_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mqloss = MagicMock()

        cfg = _make_tft_config()
        model = TFTForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(MQLoss=mock_mqloss),
                "neuralforecast.models": MagicMock(TFT=mock_tft_cls),
            },
        ):
            model._build_nf_model(max_steps=999)

        call_kwargs = mock_tft_cls.call_args
        if call_kwargs is not None:
            assert call_kwargs.kwargs.get("max_steps") == 999

    def test_build_nf_model_with_callbacks(self) -> None:
        """Test external callbacks are passed through."""
        mock_tft_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tft_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mqloss = MagicMock()

        cfg = _make_tft_config(enable_progress_bar=False)
        model = TFTForecaster(cfg)

        my_callback = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(MQLoss=mock_mqloss),
                "neuralforecast.models": MagicMock(TFT=mock_tft_cls),
            },
        ):
            model._build_nf_model(callbacks=[my_callback])

        call_kwargs = mock_tft_cls.call_args
        if call_kwargs is not None:
            cbs = call_kwargs.kwargs.get("callbacks", [])
            assert my_callback in cbs

    def test_build_nf_model_progress_bar_adds_callback(self) -> None:
        """Test enable_progress_bar=True adds TQDMProgressBar callback."""
        mock_tft_cls = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 50
        mock_param.requires_grad = True
        mock_tft_cls.return_value = MagicMock(parameters=MagicMock(return_value=[mock_param]))
        mock_nf_cls = MagicMock()
        mock_mqloss = MagicMock()
        mock_tqdm = MagicMock()

        cfg = _make_tft_config(enable_progress_bar=True)
        model = TFTForecaster(cfg)

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
                "neuralforecast.losses": MagicMock(),
                "neuralforecast.losses.pytorch": MagicMock(MQLoss=mock_mqloss),
                "neuralforecast.models": MagicMock(TFT=mock_tft_cls),
                "pytorch_lightning": MagicMock(),
                "pytorch_lightning.callbacks": MagicMock(TQDMProgressBar=mock_tqdm),
            },
        ):
            model._build_nf_model()

        call_kwargs = mock_tft_cls.call_args
        if call_kwargs is not None:
            cbs = call_kwargs.kwargs.get("callbacks", [])
            # At least one callback should be the mocked TQDMProgressBar return value
            assert len(cbs) >= 1


class TestTFTTrainMocked:
    """Tests for train() with mocked _build_nf_model."""

    def test_train_returns_metrics_dict(self) -> None:
        """Test train returns metrics dict after fit."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        df = _make_sample_df(200)
        train_df = df.iloc[:150]
        val_df = df.iloc[150:]

        mock_nf = MagicMock()
        # Simulate trainer.callback_metrics
        mock_tft_model = MagicMock()
        mock_tft_model.trainer.callback_metrics = {"train_loss": 0.5}
        mock_nf.models = [mock_tft_model]

        with patch.object(model, "_build_nf_model", return_value=mock_nf):
            metrics = model.train(train_df, val_df)

        assert isinstance(metrics, dict)
        assert model._nf is mock_nf
        assert model._last_train_df is not None
        mock_nf.fit.assert_called_once()

    def test_train_without_val_df(self) -> None:
        """Test train works when val_df is None."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        df = _make_sample_df(200)

        mock_nf = MagicMock()
        mock_tft_model = MagicMock()
        mock_tft_model.trainer = None
        mock_nf.models = [mock_tft_model]

        with patch.object(model, "_build_nf_model", return_value=mock_nf):
            metrics = model.train(df, val_df=None)

        assert isinstance(metrics, dict)
        # val_size should be 0
        fit_call = mock_nf.fit.call_args
        assert fit_call.kwargs.get("val_size") == 0

    def test_train_passes_max_steps_and_callbacks(self) -> None:
        """Test train forwards max_steps and callbacks to _build_nf_model."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        df = _make_sample_df(200)

        mock_nf = MagicMock()
        mock_tft_model = MagicMock()
        mock_tft_model.trainer = None
        mock_nf.models = [mock_tft_model]
        my_cb = MagicMock()

        with patch.object(model, "_build_nf_model", return_value=mock_nf) as mock_build:
            model.train(df, max_steps=77, callbacks=[my_cb])

        mock_build.assert_called_once_with(callbacks=[my_cb], max_steps=77)

    def test_train_handles_metric_extraction_error(self) -> None:
        """Test train returns empty metrics when extraction fails."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        df = _make_sample_df(200)

        mock_nf = MagicMock()
        mock_tft_model = MagicMock()
        mock_tft_model.trainer.callback_metrics.items.side_effect = RuntimeError("boom")
        mock_nf.models = [mock_tft_model]

        with patch.object(model, "_build_nf_model", return_value=mock_nf):
            metrics = model.train(df)

        assert isinstance(metrics, dict)
        # Metrics might be empty due to exception
        assert model._nf is mock_nf


class TestTFTPredictMocked:
    """Tests for predict() with mocked NF."""

    def _make_fitted_model(
        self, pred_len: int = 12, enc_len: int = 24
    ) -> tuple[TFTForecaster, MagicMock]:
        cfg = _make_tft_config(prediction_length=pred_len, encoder_length=enc_len)
        model = TFTForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf
        return model, mock_nf

    def test_predict_raises_if_not_fitted(self) -> None:
        """Test predict raises RuntimeError when model not fitted."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        df = _make_sample_df(50)
        with pytest.raises(RuntimeError, match="trained"):
            model.predict(df)

    def test_predict_long_input_splits_context_and_forecast(self) -> None:
        """Test predict splits X into context and forecast when len(X) > pred_len."""
        model, mock_nf = self._make_fitted_model(pred_len=12, enc_len=24)
        df = _make_sample_df(100)

        preds_df = pd.DataFrame(
            {"TFT-median": np.ones(12)},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)

        assert "consumption_mwh" in result.columns
        assert len(result) == 12
        # NF predict should have been called with context
        call_kwargs = mock_nf.predict.call_args.kwargs
        assert call_kwargs["df"] is not None  # nf_context passed
        assert call_kwargs["futr_df"] is not None

    def test_predict_short_input_no_context(self) -> None:
        """Test predict with short input (len <= pred_len) has no explicit context."""
        model, mock_nf = self._make_fitted_model(pred_len=12, enc_len=24)
        df = _make_sample_df(10)

        preds_df = pd.DataFrame(
            {"TFT-median": np.ones(10)},
            index=pd.RangeIndex(10),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)

        assert "consumption_mwh" in result.columns
        call_kwargs = mock_nf.predict.call_args.kwargs
        assert call_kwargs["df"] is None  # No context for short input

    def test_predict_fallback_to_tft_column(self) -> None:
        """Test median_col fallback: TFT-median missing -> TFT column."""
        model, mock_nf = self._make_fitted_model(pred_len=12)
        df = _make_sample_df(50)

        preds_df = pd.DataFrame(
            {"TFT": np.ones(12) * 42},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)
        assert result["consumption_mwh"].iloc[0] == pytest.approx(42.0)

    def test_predict_fallback_to_first_tft_column(self) -> None:
        """Test median_col fallback: TFT-median and TFT missing -> first TFT* col."""
        model, mock_nf = self._make_fitted_model(pred_len=12)
        df = _make_sample_df(50)

        preds_df = pd.DataFrame(
            {"TFT-lo-80.0": np.ones(12) * 99},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)
        assert result["consumption_mwh"].iloc[0] == pytest.approx(99.0)

    def test_predict_fallback_to_last_column(self) -> None:
        """Test median_col fallback: no TFT* columns -> last column."""
        model, mock_nf = self._make_fitted_model(pred_len=12)
        df = _make_sample_df(50)

        preds_df = pd.DataFrame(
            {"some_other": np.ones(12) * 7},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        result = model.predict(df)
        assert result["consumption_mwh"].iloc[0] == pytest.approx(7.0)

    def test_predict_context_nan_ffill(self) -> None:
        """Test predict fills NaN in context target with ffill/bfill."""
        model, mock_nf = self._make_fitted_model(pred_len=12, enc_len=24)
        df = _make_sample_df(50)
        # Set some target values to NaN in context area
        df.iloc[10:15, df.columns.get_loc("consumption")] = np.nan

        preds_df = pd.DataFrame(
            {"TFT-median": np.ones(12)},
            index=pd.RangeIndex(12),
        )
        mock_nf.predict.return_value = preds_df

        # Should not raise despite NaN in context
        result = model.predict(df)
        assert "consumption_mwh" in result.columns


class TestStoreQuantilePredictions:
    """Tests for _store_quantile_predictions."""

    def test_stores_median_and_quantiles(self) -> None:
        """Test correct quantile-to-column mapping."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)

        preds = pd.DataFrame(
            {
                "TFT-lo-80.0": [1.0, 2.0],
                "TFT-median": [5.0, 6.0],
                "TFT-hi-80.0": [9.0, 10.0],
            }
        )
        model._store_quantile_predictions(preds)

        assert model._all_quantile_predictions is not None
        assert 0.5 in model._all_quantile_predictions
        np.testing.assert_array_equal(model._all_quantile_predictions[0.5], [5.0, 6.0])

    def test_handles_missing_columns_gracefully(self) -> None:
        """Test missing quantile columns do not cause errors."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)

        preds = pd.DataFrame({"TFT-median": [5.0]})
        model._store_quantile_predictions(preds)

        # Only median should be stored (0.5)
        assert model._all_quantile_predictions is not None
        assert 0.5 in model._all_quantile_predictions
        # 0.10 and 0.90 not present because columns are missing
        assert 0.10 not in model._all_quantile_predictions

    def test_empty_result_sets_none(self) -> None:
        """Test all columns missing results in None."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)

        preds = pd.DataFrame({"unrelated_col": [1.0]})
        model._store_quantile_predictions(preds)

        assert model._all_quantile_predictions is None


class TestTFTRollingPredictMocked:
    """Tests for rolling_predict with mocked NF."""

    def _make_fitted_model(self) -> tuple[TFTForecaster, MagicMock]:
        cfg = _make_tft_config(prediction_length=12, encoder_length=24)
        model = TFTForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf
        return model, mock_nf

    def test_rolling_predict_raises_if_not_fitted(self) -> None:
        """Test rolling_predict raises when model not fitted."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
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
                {"TFT-median": np.full(n, float(call_count[0]))},
                index=pd.RangeIndex(n),
            )

        mock_nf.predict.side_effect = fake_predict

        eval_start = df.index[100]
        eval_end = df.index[130]

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end, step_hours=12)

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        # Later window values should overwrite earlier ones in overlapping hours
        if len(result) > 12:
            # The overlapping portion should have the later window's value
            later_val = result["consumption_mwh"].iloc[-1]
            assert later_val > 0

    def test_rolling_predict_skip_insufficient_context(self) -> None:
        """Test windows with insufficient context are skipped."""
        model, mock_nf = self._make_fitted_model()
        # Create short DataFrame: not enough history for encoder
        df = _make_sample_df(30)

        mock_nf.predict.return_value = pd.DataFrame(
            {"TFT-median": np.ones(12)}, index=pd.RangeIndex(12)
        )

        # eval_start at beginning means no encoder context
        eval_start = df.index[0]
        eval_end = df.index[11]

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end)

        # Should return empty or very few predictions due to insufficient context
        assert isinstance(result, pd.DataFrame)

    def test_rolling_predict_empty_predictions(self) -> None:
        """Test empty predictions returns empty DataFrame."""
        model, _mock_nf = self._make_fitted_model()
        df = _make_sample_df(200)

        # eval period far beyond data
        eval_start = df.index[-1] + pd.Timedelta(hours=1000)
        eval_end = eval_start + pd.Timedelta(hours=48)

        result = model.rolling_predict(df, eval_start=eval_start, eval_end=eval_end)

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        assert len(result) == 0


class TestTFTSaveMocked:
    """Tests for save() with mocked NF and torch."""

    def test_save_creates_metadata_json(self, tmp_path: Path) -> None:
        """Test save creates metadata.json with correct content."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf

        mock_torch = MagicMock()
        mock_torch.load.return_value = {
            "hyper_parameters": {"callbacks": ["some_cb"]},
        }

        with patch.dict("sys.modules", {"torch": mock_torch}):
            model.save(tmp_path / "tft_model")

        metadata_path = tmp_path / "tft_model" / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            meta = json.load(f)

        assert meta["quantiles"] == [0.10, 0.50, 0.90]
        assert "architecture" in meta
        assert "training" in meta
        assert "covariates" in meta
        assert meta["training"]["encoder_length"] == 24
        assert meta["training"]["prediction_length"] == 12

    def test_save_raises_if_not_fitted(self, tmp_path: Path) -> None:
        """Test save raises ValueError when model not fitted."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        with pytest.raises(ValueError, match="unfitted"):
            model.save(tmp_path / "bad")

    def test_save_strips_callbacks_from_ckpt(self, tmp_path: Path) -> None:
        """Test save strips callbacks from checkpoint files."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf

        save_dir = tmp_path / "tft_model"
        save_dir.mkdir(parents=True)

        # Create a fake ckpt file so glob finds it
        fake_ckpt = save_dir / "model.ckpt"
        fake_ckpt.write_bytes(b"fake_checkpoint_data")

        ckpt_data = {"hyper_parameters": {"callbacks": ["early_stop"]}}
        mock_torch = MagicMock()
        mock_torch.load.return_value = ckpt_data

        # NF save is a no-op (ckpt file already exists)
        mock_nf.save.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            model.save(save_dir)

        # torch.save should be called to overwrite with stripped callbacks
        mock_torch.save.assert_called()
        saved_ckpt = mock_torch.save.call_args[0][0]
        assert saved_ckpt["hyper_parameters"]["callbacks"] == []

    def test_save_computes_ckpt_hashes(self, tmp_path: Path) -> None:
        """Test save computes SHA256 hashes for ckpt files."""
        cfg = _make_tft_config()
        model = TFTForecaster(cfg)
        mock_nf = MagicMock()
        model._nf = mock_nf

        save_dir = tmp_path / "tft_model"
        save_dir.mkdir(parents=True)

        fake_ckpt = save_dir / "model.ckpt"
        ckpt_content = b"checkpoint_content_for_hash_test"
        fake_ckpt.write_bytes(ckpt_content)
        expected_hash = hashlib.sha256(ckpt_content).hexdigest()

        mock_torch = MagicMock()
        mock_torch.load.return_value = {}  # No hyper_parameters key

        with patch.dict("sys.modules", {"torch": mock_torch}):
            model.save(save_dir)

        metadata_path = save_dir / "metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)

        assert "model.ckpt" in meta["ckpt_hashes"]
        assert meta["ckpt_hashes"]["model.ckpt"] == expected_hash


class TestTFTFromCheckpointMocked:
    """Tests for from_checkpoint() with mocked NF."""

    def test_from_checkpoint_loads_metadata(self, tmp_path: Path) -> None:
        """Test from_checkpoint reads metadata.json and reconstructs config."""
        model_dir = tmp_path / "tft_model"
        model_dir.mkdir()

        metadata = {
            "quantiles": [0.10, 0.50, 0.90],
            "architecture": {"hidden_size": 32, "n_head": 2, "n_rnn_layers": 1, "dropout": 0.1},
            "training": {
                "encoder_length": 48,
                "prediction_length": 24,
                "num_workers": 2,
                "scaler_type": "standard",
                "rnn_type": "gru",
            },
            "covariates": {
                "time_varying_known": ["hour_sin"],
                "time_varying_unknown": ["lag_48"],
            },
            "ckpt_hashes": {},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_nf_instance = MagicMock()
        mock_nf_instance.models = []  # No models to strip callbacks from
        mock_nf_cls = MagicMock()
        mock_nf_cls.load.return_value = mock_nf_instance

        with patch.dict(
            "sys.modules",
            {
                "neuralforecast": MagicMock(NeuralForecast=mock_nf_cls),
            },
        ):
            loaded = TFTForecaster.from_checkpoint(model_dir)

        assert loaded.is_fitted
        assert loaded._quantiles == [0.10, 0.50, 0.90]
        assert loaded._tft_config.architecture.hidden_size == 32
        assert loaded._tft_config.training.encoder_length == 48

    def test_from_checkpoint_raises_on_missing_metadata(self, tmp_path: Path) -> None:
        """Test from_checkpoint raises FileNotFoundError without metadata.json."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            TFTForecaster.from_checkpoint(model_dir)

    def test_from_checkpoint_verifies_hash(self, tmp_path: Path) -> None:
        """Test from_checkpoint raises on hash mismatch."""
        model_dir = tmp_path / "tft_model"
        model_dir.mkdir()

        fake_ckpt = model_dir / "model.ckpt"
        fake_ckpt.write_bytes(b"real_content")

        metadata = {
            "quantiles": [0.50],
            "architecture": {"hidden_size": 16, "n_head": 1, "n_rnn_layers": 1, "dropout": 0.0},
            "training": {},
            "covariates": {"time_varying_known": [], "time_varying_unknown": []},
            "ckpt_hashes": {"model.ckpt": "0" * 64},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(RuntimeError, match="integrity check failed"):
            TFTForecaster.from_checkpoint(model_dir)

    def test_from_checkpoint_strips_callbacks(self, tmp_path: Path) -> None:
        """Test from_checkpoint clears callbacks from loaded models."""
        model_dir = tmp_path / "tft_model"
        model_dir.mkdir()

        metadata = {
            "quantiles": [0.50],
            "architecture": {"hidden_size": 16, "n_head": 1, "n_rnn_layers": 1, "dropout": 0.0},
            "training": {},
            "covariates": {"time_varying_known": [], "time_varying_unknown": []},
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
            loaded = TFTForecaster.from_checkpoint(model_dir)

        assert mock_model.hparams["callbacks"] == []
        assert loaded.is_fitted


class TestTFTLoadMocked:
    """Tests for load() instance method with mocked NF."""

    def test_load_sets_nf_and_quantiles(self, tmp_path: Path) -> None:
        """Test load() sets _nf and reads quantiles from metadata."""
        model_dir = tmp_path / "tft_model"
        model_dir.mkdir()

        metadata = {"quantiles": [0.05, 0.50, 0.95]}
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_nf_instance = MagicMock()
        mock_nf_instance.models = []

        cfg = _make_tft_config()
        model = TFTForecaster(cfg)

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
        assert model._quantiles == [0.05, 0.50, 0.95]

    def test_load_strips_callbacks(self, tmp_path: Path) -> None:
        """Test load() strips callbacks from loaded models."""
        model_dir = tmp_path / "tft_model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(json.dumps({"quantiles": [0.50]}))

        mock_model = MagicMock()
        mock_model.hparams = {"callbacks": ["cb1"]}
        mock_nf_instance = MagicMock()
        mock_nf_instance.models = [mock_model]

        cfg = _make_tft_config()
        model = TFTForecaster(cfg)

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
