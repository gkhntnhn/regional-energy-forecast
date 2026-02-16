"""Unit tests for TFTForecaster."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        """Test save creates checkpoint, metadata, and training dataset files."""
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
            assert (save_path / "training_dataset.pkl").exists()

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
        """Test load restores metadata, quantiles, and makes model usable."""
        model = TFTForecaster(tft_config)
        train_df = sample_df.iloc[:200]
        val_df = sample_df.iloc[200:]

        model.train(train_df, val_df, max_epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save(save_path)

            # Load into new model via instance method
            new_model = TFTForecaster(tft_config)
            new_model.load(save_path)

            assert new_model._quantiles == model._quantiles
            assert new_model._dataset_params == model._dataset_params
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

        model.train(train_df, val_df, max_epochs=2)

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
            assert "yhat" in predictions.columns
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


class TestPredictRolling:
    """Tests for rolling window prediction."""

    def _make_mock_model(
        self, tft_config: TFTConfig, pred_len: int
    ) -> TFTForecaster:
        """Create a TFTForecaster with mocked predict()."""
        model = TFTForecaster(tft_config)
        # Mark as fitted so predict_rolling doesn't fail early
        model._model = MagicMock()
        return model

    def test_short_input_delegates_to_predict(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Input <= enc+pred delegates directly to predict()."""
        model = self._make_mock_model(tft_config, pred_len=12)
        enc_len = tft_config.training.encoder_length  # 24
        pred_len = tft_config.training.prediction_length  # 12
        window_size = enc_len + pred_len  # 36

        # Use exactly window_size rows
        short_df = sample_df.iloc[:window_size]
        expected_result = pd.DataFrame(
            {"yhat": np.ones(pred_len)},
            index=short_df.index[-pred_len:],
        )

        with patch.object(model, "predict", return_value=expected_result) as mock_pred:
            result = model.predict_rolling(short_df)
            mock_pred.assert_called_once()
            assert len(result) == pred_len

    def test_rolling_produces_full_coverage(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """Input > enc+pred produces predictions beyond the first window."""
        model = self._make_mock_model(tft_config, pred_len=12)
        enc_len = tft_config.training.encoder_length  # 24
        pred_len = tft_config.training.prediction_length  # 12

        # Use 100 rows (much larger than enc+pred=36)
        long_df = sample_df.iloc[:100]

        def mock_predict(window_df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
            """Return pred_len predictions from end of window."""
            return pd.DataFrame(
                {"yhat": np.ones(pred_len) * 42.0},
                index=window_df.index[-pred_len:],
            )

        with patch.object(model, "predict", side_effect=mock_predict):
            result = model.predict_rolling(long_df)

        # Should have more than pred_len predictions
        assert len(result) > pred_len
        # All predictions should have the mocked value
        assert np.allclose(result["yhat"].values, 42.0)

    def test_non_overlapping_single_prediction_per_timestamp(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """With step=pred_len, each timestamp gets exactly 1 prediction."""
        model = self._make_mock_model(tft_config, pred_len=12)
        enc_len = tft_config.training.encoder_length  # 24
        pred_len = tft_config.training.prediction_length  # 12

        long_df = sample_df.iloc[:72]  # 3 full windows (24+12)*2
        call_count = 0

        def mock_predict(window_df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return pd.DataFrame(
                {"yhat": np.ones(pred_len) * float(call_count)},
                index=window_df.index[-pred_len:],
            )

        with patch.object(model, "predict", side_effect=mock_predict):
            result = model.predict_rolling(long_df, step=pred_len)

        # Each window produces different values, no averaging expected
        assert call_count >= 2
        # Timestamps should be unique (no duplicates in index)
        assert result.index.is_unique

    def test_overlapping_averages_predictions(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """With step < pred_len, overlapping timestamps are averaged."""
        model = self._make_mock_model(tft_config, pred_len=12)
        pred_len = tft_config.training.prediction_length  # 12
        step = pred_len // 2  # 6 — produces overlaps

        long_df = sample_df.iloc[:60]
        window_num = 0

        def mock_predict(window_df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
            nonlocal window_num
            window_num += 1
            # Alternate between 10.0 and 20.0 per window
            val = 10.0 if window_num % 2 == 1 else 20.0
            return pd.DataFrame(
                {"yhat": np.ones(pred_len) * val},
                index=window_df.index[-pred_len:],
            )

        with patch.object(model, "predict", side_effect=mock_predict):
            result = model.predict_rolling(long_df, step=step)

        # With overlapping windows returning 10 and 20, averaged should be 15
        # (for timestamps covered by both windows)
        assert result.index.is_unique
        # At least some predictions should be averaged (value = 15.0)
        values = result["yhat"].values
        has_averaged = any(np.isclose(v, 15.0) for v in values)
        assert has_averaged, f"Expected some averaged values (15.0), got: {np.unique(values)}"

    def test_predict_rolling_raises_if_not_fitted(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """predict_rolling raises if model not fitted (short input delegates to predict)."""
        model = TFTForecaster(tft_config)
        enc_len = tft_config.training.encoder_length
        pred_len = tft_config.training.prediction_length
        short_df = sample_df.iloc[: enc_len + pred_len]

        with pytest.raises(RuntimeError, match="trained"):
            model.predict_rolling(short_df)

    def test_all_windows_fail_raises_error(
        self,
        tft_config: TFTConfig,
        sample_df: pd.DataFrame,
    ) -> None:
        """If all windows fail, raises RuntimeError."""
        model = self._make_mock_model(tft_config, pred_len=12)
        long_df = sample_df.iloc[:100]

        def mock_predict_fail(
            window_df: pd.DataFrame, target_col: str | None = None
        ) -> pd.DataFrame:
            msg = "Simulated failure"
            raise ValueError(msg)

        with patch.object(model, "predict", side_effect=mock_predict_fail):
            with pytest.raises(RuntimeError, match="All.*windows failed"):
                model.predict_rolling(long_df)
