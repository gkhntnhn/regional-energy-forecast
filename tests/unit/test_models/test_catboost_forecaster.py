"""Tests for CatBoostForecaster save/load/predict."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_forecast.models.catboost import CatBoostForecaster

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_x(n: int = 48) -> pd.DataFrame:
    """Create a feature DataFrame (no target column)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-06-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "temperature_2m": rng.uniform(-5, 35, n),
            "hour": idx.hour.astype(float),
            "day_of_week": idx.dayofweek.astype(float),
        },
        index=idx,
    ).rename_axis("datetime")


# ---------------------------------------------------------------------------
# Model property guard
# ---------------------------------------------------------------------------


class TestModelProperty:
    """Tests for model access guard."""

    def test_model_raises_when_not_loaded(self) -> None:
        forecaster = CatBoostForecaster(config={})
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = forecaster.model

    def test_set_model_attaches(self) -> None:
        forecaster = CatBoostForecaster(config={})
        mock_model = MagicMock()
        forecaster.set_model(mock_model)
        assert forecaster.model is mock_model


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests for prediction output."""

    def test_predict_returns_dataframe(self) -> None:
        forecaster = CatBoostForecaster(config={})
        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones(48)
        forecaster.set_model(mock_model)

        x = _make_feature_x(48)
        result = forecaster.predict(x)

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        assert len(result) == 48
        assert result.index.equals(x.index)

    def test_predict_raises_without_model(self) -> None:
        forecaster = CatBoostForecaster(config={})
        x = _make_feature_x(48)
        with pytest.raises(RuntimeError, match="not loaded"):
            forecaster.predict(x)


# ---------------------------------------------------------------------------
# Save / Load (mocked CatBoost I/O)
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for model persistence."""

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        forecaster = CatBoostForecaster(config={})
        mock_model = MagicMock()
        forecaster.set_model(mock_model)

        save_dir = tmp_path / "models" / "catboost"
        forecaster.save(save_dir)

        assert save_dir.exists()
        mock_model.save_model.assert_called_once_with(str(save_dir / "model.cbm"))

    def test_save_raises_without_model(self, tmp_path: Path) -> None:
        forecaster = CatBoostForecaster(config={})
        with pytest.raises(RuntimeError, match="not loaded"):
            forecaster.save(tmp_path)

    def test_load_creates_model(self, tmp_path: Path) -> None:
        forecaster = CatBoostForecaster(config={})

        # Patch CatBoostRegressor at module level
        from unittest.mock import patch

        with patch("energy_forecast.models.catboost.CatBoostRegressor") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            forecaster.load(tmp_path)

            mock_instance.load_model.assert_called_once_with(str(tmp_path / "model.cbm"))
            assert forecaster._model is mock_instance


# ---------------------------------------------------------------------------
# Simple train (without Trainer)
# ---------------------------------------------------------------------------


class TestSimpleTrain:
    """Tests for the simple train method (no Optuna)."""

    def test_simple_train_without_validation(self) -> None:
        from unittest.mock import patch

        with patch("energy_forecast.models.catboost.CatBoostRegressor") as mock_cls:
            mock_model = MagicMock()
            mock_cls.return_value = mock_model

            config: dict[str, Any] = {"target_col": "consumption", "params": {"iterations": 10}}
            forecaster = CatBoostForecaster(config=config)

            rng = np.random.default_rng(42)
            idx = pd.date_range("2024-01-01", periods=100, freq="h")
            train_df = pd.DataFrame(
                {
                    "consumption": 800 + rng.random(100) * 400,
                    "temperature": rng.uniform(-5, 35, 100),
                },
                index=idx,
            )

            forecaster.train(train_df)

            mock_model.fit.assert_called_once()
            call_args = mock_model.fit.call_args
            assert "consumption" not in call_args[0][0].columns

    def test_simple_train_with_validation(self) -> None:
        from unittest.mock import patch

        with patch("energy_forecast.models.catboost.CatBoostRegressor") as mock_cls:
            mock_model = MagicMock()
            mock_cls.return_value = mock_model

            config: dict[str, Any] = {"target_col": "consumption", "params": {}}
            forecaster = CatBoostForecaster(config=config)

            rng = np.random.default_rng(42)
            idx = pd.date_range("2024-01-01", periods=100, freq="h")
            df = pd.DataFrame(
                {
                    "consumption": 800 + rng.random(100) * 400,
                    "temperature": rng.uniform(-5, 35, 100),
                },
                index=idx,
            )

            forecaster.train(df.iloc[:80], val_df=df.iloc[80:])

            mock_model.fit.assert_called_once()
            # eval_set should be passed when val_df provided
            call_kwargs = mock_model.fit.call_args[1]
            assert "eval_set" in call_kwargs
