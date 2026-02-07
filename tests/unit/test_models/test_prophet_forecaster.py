"""Unit tests for ProphetForecaster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecast.models.prophet import ProphetForecaster

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample hourly data for testing."""
    dates = pd.date_range("2023-01-01", "2023-03-31 23:00", freq="h")
    n = len(dates)
    rng = np.random.default_rng(42)

    # Synthetic consumption with daily pattern
    hour_effect = np.sin(np.arange(n) * 2 * np.pi / 24) * 100
    trend = np.linspace(1000, 1100, n)
    noise = rng.normal(0, 30, n)
    consumption = trend + hour_effect + noise

    return pd.DataFrame(
        {
            "consumption": consumption,
            "temperature": rng.uniform(5, 30, n),
        },
        index=dates,
    )


@pytest.fixture
def forecaster() -> ProphetForecaster:
    """Create ProphetForecaster instance."""
    return ProphetForecaster({"target_col": "consumption"})


@pytest.fixture
def trained_forecaster(sample_df: pd.DataFrame) -> ProphetForecaster:
    """Create and train a ProphetForecaster."""
    forecaster = ProphetForecaster({"target_col": "consumption"})
    train_df = sample_df.iloc[:1000]
    forecaster.train(train_df)
    return forecaster


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for ProphetForecaster initialization."""

    def test_init_with_config(self) -> None:
        """Test initialization with config dict."""
        config = {"target_col": "consumption", "custom_key": "value"}
        forecaster = ProphetForecaster(config)

        assert forecaster.config["target_col"] == "consumption"
        assert forecaster.config["custom_key"] == "value"
        assert forecaster._model is None

    def test_init_default_regressor_names(self) -> None:
        """Test that regressor names start empty."""
        forecaster = ProphetForecaster({})
        assert forecaster._regressor_names == []


# ---------------------------------------------------------------------------
# Tests: Training
# ---------------------------------------------------------------------------


class TestTrain:
    """Tests for train method."""

    def test_train_simple(self, forecaster: ProphetForecaster, sample_df: pd.DataFrame) -> None:
        """Test simple training."""
        train_df = sample_df.iloc[:500]
        forecaster.train(train_df)

        assert forecaster._model is not None

    def test_train_with_val_df_ignored(
        self, forecaster: ProphetForecaster, sample_df: pd.DataFrame
    ) -> None:
        """Test that val_df parameter is ignored."""
        train_df = sample_df.iloc[:500]
        val_df = sample_df.iloc[500:600]
        forecaster.train(train_df, val_df)

        assert forecaster._model is not None


# ---------------------------------------------------------------------------
# Tests: Prediction
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests for predict method."""

    def test_predict_returns_correct_shape(
        self, trained_forecaster: ProphetForecaster, sample_df: pd.DataFrame
    ) -> None:
        """Test that predictions have correct shape."""
        test_df = sample_df.iloc[1000:1100]
        predictions = trained_forecaster.predict(test_df)

        assert len(predictions) == len(test_df)
        assert "consumption_mwh" in predictions.columns

    def test_predict_preserves_index(
        self, trained_forecaster: ProphetForecaster, sample_df: pd.DataFrame
    ) -> None:
        """Test that prediction index matches input."""
        test_df = sample_df.iloc[1000:1100]
        predictions = trained_forecaster.predict(test_df)

        pd.testing.assert_index_equal(predictions.index, test_df.index)

    def test_predict_without_model_raises(
        self, forecaster: ProphetForecaster, sample_df: pd.DataFrame
    ) -> None:
        """Test that predicting without model raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            forecaster.predict(sample_df.iloc[:10])


# ---------------------------------------------------------------------------
# Tests: Save/Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save and load methods."""

    def test_save_creates_files(
        self, trained_forecaster: ProphetForecaster, tmp_path: Path
    ) -> None:
        """Test that save creates expected files."""
        save_path = tmp_path / "prophet_model"
        trained_forecaster.save(save_path)

        assert (save_path / "prophet_model.pkl").exists()
        assert (save_path / "metadata.json").exists()

    def test_load_restores_model(
        self, trained_forecaster: ProphetForecaster, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test that load restores model correctly."""
        save_path = tmp_path / "prophet_model"
        trained_forecaster.save(save_path)

        # Create new forecaster and load
        new_forecaster = ProphetForecaster({})
        new_forecaster.load(save_path)

        assert new_forecaster._model is not None

    def test_save_load_roundtrip(
        self, trained_forecaster: ProphetForecaster, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test full save/load/predict roundtrip."""
        test_df = sample_df.iloc[1000:1100]

        # Get predictions before save
        pred_before = trained_forecaster.predict(test_df)

        # Save and load
        save_path = tmp_path / "prophet_model"
        trained_forecaster.save(save_path)

        new_forecaster = ProphetForecaster({})
        new_forecaster.load(save_path)

        # Get predictions after load
        pred_after = new_forecaster.predict(test_df)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(
            pred_before["consumption_mwh"].values,
            pred_after["consumption_mwh"].values,
        )

    def test_save_without_model_raises(
        self, forecaster: ProphetForecaster, tmp_path: Path
    ) -> None:
        """Test that saving without model raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No model to save"):
            forecaster.save(tmp_path / "model")

    def test_load_nonexistent_raises(self, forecaster: ProphetForecaster, tmp_path: Path) -> None:
        """Test that loading nonexistent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            forecaster.load(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Tests: Format Conversion
# ---------------------------------------------------------------------------


class TestToProphetFormat:
    """Tests for _to_prophet_format method."""

    def test_format_with_target(
        self, forecaster: ProphetForecaster, sample_df: pd.DataFrame
    ) -> None:
        """Test format conversion with target."""
        prophet_df = forecaster._to_prophet_format(sample_df, include_target=True)

        assert "ds" in prophet_df.columns
        assert "y" in prophet_df.columns
        assert len(prophet_df) == len(sample_df)

    def test_format_without_target(
        self, forecaster: ProphetForecaster, sample_df: pd.DataFrame
    ) -> None:
        """Test format conversion without target."""
        prophet_df = forecaster._to_prophet_format(sample_df, include_target=False)

        assert "ds" in prophet_df.columns
        assert "y" not in prophet_df.columns


# ---------------------------------------------------------------------------
# Tests: set_model
# ---------------------------------------------------------------------------


class TestSetModel:
    """Tests for set_model method."""

    def test_set_model(self, forecaster: ProphetForecaster) -> None:
        """Test setting a pre-trained model."""
        from prophet import Prophet

        mock_model = Prophet()
        forecaster.set_model(mock_model, regressor_names=["temperature"])

        assert forecaster._model is mock_model
        assert forecaster._regressor_names == ["temperature"]

    def test_set_model_without_regressors(self, forecaster: ProphetForecaster) -> None:
        """Test setting model without regressor names."""
        from prophet import Prophet

        mock_model = Prophet()
        forecaster.set_model(mock_model)

        assert forecaster._model is mock_model
        assert forecaster._regressor_names == []
