"""Tests for the EnsembleForecaster model."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_forecast.models.ensemble import EnsembleForecaster

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(n_rows: int = 48) -> pd.DataFrame:
    """Create a minimal feature DataFrame for prediction."""
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "consumption": 800.0 + rng.random(n_rows) * 400,
            "temperature_2m": rng.uniform(-5, 35, n_rows),
            "hour": idx.hour.astype(float),
        },
        index=idx,
    )


def _make_mock_catboost() -> MagicMock:
    """Create a mock CatBoost model."""
    mock = MagicMock()
    mock.predict.return_value = np.array([1000.0] * 48)
    return mock


def _make_mock_prophet() -> MagicMock:
    """Create a mock Prophet model."""
    mock = MagicMock()
    mock.predict.return_value = pd.DataFrame({"yhat": np.array([900.0] * 48)})
    return mock


def _make_mock_tft() -> MagicMock:
    """Create a mock TFT model."""
    mock = MagicMock()
    mock.predict.return_value = pd.DataFrame({"consumption_mwh": np.array([950.0] * 48)})
    return mock


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEnsembleForecasterInit:
    """Tests for EnsembleForecaster initialization."""

    def test_default_config(self) -> None:
        config: dict[str, Any] = {}
        forecaster = EnsembleForecaster(config)
        assert forecaster.weights == {"catboost": 0.45, "prophet": 0.30, "tft": 0.25}

    def test_default_active_models(self) -> None:
        config: dict[str, Any] = {}
        forecaster = EnsembleForecaster(config)
        assert forecaster.active_models == ["catboost", "prophet", "tft"]

    def test_custom_weights(self) -> None:
        config = {"weights": {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}}
        forecaster = EnsembleForecaster(config)
        assert forecaster.weights == {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}

    def test_custom_active_models(self) -> None:
        config = {"active_models": ["catboost", "prophet"]}
        forecaster = EnsembleForecaster(config)
        assert forecaster.active_models == ["catboost", "prophet"]

    def test_weights_property_returns_copy(self) -> None:
        config: dict[str, Any] = {}
        forecaster = EnsembleForecaster(config)
        weights = forecaster.weights
        weights["catboost"] = 0.99
        assert forecaster.weights["catboost"] == 0.45

    def test_active_models_property_returns_copy(self) -> None:
        config: dict[str, Any] = {}
        forecaster = EnsembleForecaster(config)
        models = forecaster.active_models
        models.append("fake")
        assert "fake" not in forecaster.active_models


# ---------------------------------------------------------------------------
# Set weights
# ---------------------------------------------------------------------------


class TestSetWeights:
    """Tests for setting ensemble weights."""

    def test_set_valid_weights(self) -> None:
        forecaster = EnsembleForecaster({})
        forecaster.set_weights({"catboost": 0.5, "prophet": 0.3, "tft": 0.2})
        assert forecaster.weights == {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}

    def test_set_invalid_weights_raises(self) -> None:
        forecaster = EnsembleForecaster({})
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            forecaster.set_weights({"catboost": 0.5, "prophet": 0.3, "tft": 0.3})

    def test_set_two_model_weights(self) -> None:
        forecaster = EnsembleForecaster({})
        forecaster.set_weights({"catboost": 0.6, "prophet": 0.4})
        assert forecaster.weights["catboost"] == 0.6
        assert forecaster.weights["prophet"] == 0.4


# ---------------------------------------------------------------------------
# Set active models
# ---------------------------------------------------------------------------


class TestSetActiveModels:
    """Tests for setting active models."""

    def test_set_valid_active_models(self) -> None:
        forecaster = EnsembleForecaster({})
        forecaster.set_active_models(["catboost", "tft"])
        assert forecaster.active_models == ["catboost", "tft"]

    def test_set_invalid_model_raises(self) -> None:
        forecaster = EnsembleForecaster({})
        with pytest.raises(ValueError, match="Unknown model"):
            forecaster.set_active_models(["catboost", "invalid"])


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


class TestTrain:
    """Tests for train method."""

    def test_train_raises_not_implemented(self) -> None:
        forecaster = EnsembleForecaster({})
        df = _make_feature_df()

        with pytest.raises(NotImplementedError, match="EnsembleTrainer"):
            forecaster.train(df)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests for prediction."""

    def test_predict_without_models_raises(self) -> None:
        forecaster = EnsembleForecaster({})
        df = _make_feature_df()

        with pytest.raises(RuntimeError, match="No active models loaded"):
            forecaster.predict(df)

    def test_predict_with_two_models(self) -> None:
        config = {"active_models": ["catboost", "prophet"]}
        forecaster = EnsembleForecaster(config)
        forecaster.set_models(
            catboost_model=_make_mock_catboost(),
            prophet_model=_make_mock_prophet(),
        )

        df = _make_feature_df()
        result = forecaster.predict(df)

        assert "consumption_mwh" in result.columns
        assert "catboost_prediction" in result.columns
        assert "prophet_prediction" in result.columns
        assert len(result) == len(df)

    def test_predict_with_all_models(self) -> None:
        forecaster = EnsembleForecaster({})
        forecaster.set_models(
            catboost_model=_make_mock_catboost(),
            prophet_model=_make_mock_prophet(),
            tft_model=_make_mock_tft(),
        )

        df = _make_feature_df()
        result = forecaster.predict(df)

        assert "consumption_mwh" in result.columns
        assert "catboost_prediction" in result.columns
        assert "prophet_prediction" in result.columns
        assert "tft_prediction" in result.columns
        assert len(result) == len(df)

    def test_predict_weighted_average_two_models(self) -> None:
        config = {
            "weights": {"catboost": 0.6, "prophet": 0.4},
            "active_models": ["catboost", "prophet"],
        }
        forecaster = EnsembleForecaster(config)

        mock_cb = MagicMock()
        mock_cb.predict.return_value = np.array([1000.0] * 48)

        mock_pr = MagicMock()
        mock_pr.predict.return_value = pd.DataFrame({"yhat": np.array([500.0] * 48)})

        forecaster.set_models(catboost_model=mock_cb, prophet_model=mock_pr)

        df = _make_feature_df()
        result = forecaster.predict(df)

        # Expected: 0.6 * 1000 + 0.4 * 500 = 800
        expected = 0.6 * 1000 + 0.4 * 500
        assert result["consumption_mwh"].iloc[0] == pytest.approx(expected)

    def test_predict_weighted_average_three_models(self) -> None:
        config = {"weights": {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}}
        forecaster = EnsembleForecaster(config)

        mock_cb = MagicMock()
        mock_cb.predict.return_value = np.array([1000.0] * 48)

        mock_pr = MagicMock()
        mock_pr.predict.return_value = pd.DataFrame({"yhat": np.array([500.0] * 48)})

        mock_tft = MagicMock()
        mock_tft.predict.return_value = pd.DataFrame({"consumption_mwh": np.array([800.0] * 48)})

        forecaster.set_models(
            catboost_model=mock_cb,
            prophet_model=mock_pr,
            tft_model=mock_tft,
        )

        df = _make_feature_df()
        result = forecaster.predict(df)

        # Expected: 0.5 * 1000 + 0.3 * 500 + 0.2 * 800 = 500 + 150 + 160 = 810
        expected = 0.5 * 1000 + 0.3 * 500 + 0.2 * 800
        assert result["consumption_mwh"].iloc[0] == pytest.approx(expected)

    def test_predict_normalizes_weights_for_subset(self) -> None:
        """Test that weights are normalized when only some active models are loaded."""
        config = {
            "weights": {"catboost": 0.5, "prophet": 0.3, "tft": 0.2},
            "active_models": ["catboost", "prophet"],  # Only 2 active
        }
        forecaster = EnsembleForecaster(config)

        mock_cb = MagicMock()
        mock_cb.predict.return_value = np.array([1000.0] * 48)

        mock_pr = MagicMock()
        mock_pr.predict.return_value = pd.DataFrame({"yhat": np.array([500.0] * 48)})

        # Only set catboost and prophet
        forecaster.set_models(catboost_model=mock_cb, prophet_model=mock_pr)

        df = _make_feature_df()
        result = forecaster.predict(df)

        # Weights normalized: 0.5 / (0.5 + 0.3) = 0.625, 0.3 / 0.8 = 0.375
        # Expected: 0.625 * 1000 + 0.375 * 500 = 625 + 187.5 = 812.5
        expected = (0.5 / 0.8) * 1000 + (0.3 / 0.8) * 500
        assert result["consumption_mwh"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Save/Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save/load functionality."""

    def test_save_creates_weights_file(self, tmp_path: Any) -> None:
        config = {"weights": {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}}
        forecaster = EnsembleForecaster(config)

        forecaster.save(tmp_path)

        weights_file = tmp_path / "ensemble_weights.json"
        assert weights_file.exists()

    def test_load_restores_weights(self, tmp_path: Any) -> None:
        # Save
        config = {
            "weights": {"catboost": 0.5, "prophet": 0.3, "tft": 0.2},
            "active_models": ["catboost", "prophet", "tft"],
        }
        forecaster1 = EnsembleForecaster(config)
        forecaster1.save(tmp_path)

        # Load
        forecaster2 = EnsembleForecaster({})
        forecaster2.load(tmp_path)

        assert forecaster2.weights == {"catboost": 0.5, "prophet": 0.3, "tft": 0.2}
        assert forecaster2.active_models == ["catboost", "prophet", "tft"]

    def test_load_models_from_files(self, tmp_path: Any) -> None:
        # Create mock model files
        import pickle

        from catboost import CatBoostRegressor
        from prophet import Prophet

        cb_path = tmp_path / "catboost.cbm"
        pr_path = tmp_path / "prophet.pkl"

        # Create minimal CatBoost model
        cb_model = CatBoostRegressor(iterations=1, verbose=0)
        cb_model.fit([[1, 2], [3, 4]], [100, 200])
        cb_model.save_model(str(cb_path))

        # Create minimal Prophet model (unfitted but pickleable)
        prophet_model = Prophet()
        with open(pr_path, "wb") as f:
            pickle.dump(prophet_model, f)

        # Test loading
        forecaster = EnsembleForecaster({})
        forecaster.load_models(catboost_path=cb_path, prophet_path=pr_path)

        assert forecaster._catboost_model is not None
        assert forecaster._prophet_model is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_predict_drops_target_column(self) -> None:
        config = {"active_models": ["catboost", "prophet"]}
        forecaster = EnsembleForecaster(config)
        forecaster.set_models(
            catboost_model=_make_mock_catboost(),
            prophet_model=_make_mock_prophet(),
        )

        df = _make_feature_df()
        assert "consumption" in df.columns

        # Should not raise even though consumption is in df
        result = forecaster.predict(df)
        assert len(result) == len(df)

    def test_set_models_works(self) -> None:
        forecaster = EnsembleForecaster({})
        forecaster.set_models(
            catboost_model=_make_mock_catboost(),
            prophet_model=_make_mock_prophet(),
            tft_model=_make_mock_tft(),
        )

        # Models should be set
        assert forecaster._catboost_model is not None
        assert forecaster._prophet_model is not None
        assert forecaster._tft_model is not None

    def test_set_models_partial(self) -> None:
        """Test setting only some models."""
        forecaster = EnsembleForecaster({})
        forecaster.set_models(catboost_model=_make_mock_catboost())

        assert forecaster._catboost_model is not None
        assert forecaster._prophet_model is None
        assert forecaster._tft_model is None
