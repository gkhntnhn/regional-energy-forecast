"""E2E integration tests for PredictionService with mock models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.models.base import PREDICTION_COL
from energy_forecast.serving.exceptions import ModelNotLoadedError
from energy_forecast.serving.services.prediction_service import (
    PredictionService,
    PredictionServiceConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_catboost(n_features: int = 50) -> MagicMock:
    """Create a mock CatBoost model that returns fixed predictions."""
    model = MagicMock()
    model.predict.side_effect = lambda X: np.full(len(X), 1200.0)
    model.feature_names_ = [f"feat_{i}" for i in range(n_features)]
    model.get_cat_feature_indices.return_value = []
    model.get_feature_importance.return_value = np.random.default_rng(42).random(n_features)
    return model


def _make_mock_prophet() -> MagicMock:
    """Create a mock Prophet model that returns fixed predictions."""
    model = MagicMock()
    model.predict.side_effect = lambda df: pd.DataFrame(
        {"yhat": np.full(len(df), 1100.0)},
        index=df.index,
    )
    return model


def _build_service(
    tmp_path: Path,
    settings: Any,
    active_models: list[str] | None = None,
) -> PredictionService:
    """Build a PredictionService with mock models injected."""
    if active_models is None:
        active_models = ["catboost"]

    config = PredictionServiceConfig(
        models_dir=tmp_path / "models",
        catboost_path=tmp_path / "models" / "catboost" / "model.cbm",
        prophet_path=tmp_path / "models" / "prophet",
        tft_path=tmp_path / "models" / "tft",
        forecast_horizon=48,
    )
    svc = PredictionService(config=config, settings=settings)
    return svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def excel_path(tmp_path: Path) -> Path:
    """Create a minimal Excel file with 60 days of hourly consumption."""
    rng = np.random.default_rng(42)
    n_days = 60
    n_hours = n_days * 24
    dates: list[str] = []
    times: list[int] = []
    consumptions: list[float] = []

    for day in range(n_days):
        dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for hour in range(24):
            dates.append(dt.strftime("%Y-%m-%d"))
            times.append(hour)
            consumptions.append(round(800.0 + rng.random() * 400, 1))

    df = pd.DataFrame({"date": dates, "time": times, "consumption": consumptions})
    path = tmp_path / "consumption.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPredictionServiceE2E:
    """E2E integration tests for PredictionService.run_prediction()."""

    def test_run_prediction_not_loaded_raises(
        self, tmp_path: Path, settings: Any
    ) -> None:
        """Calling run_prediction before load_models raises ModelNotLoadedError."""
        svc = _build_service(tmp_path, settings)
        with pytest.raises(ModelNotLoadedError):
            svc.run_prediction(tmp_path / "dummy.xlsx")

    def test_is_ready_false_before_load(self, tmp_path: Path, settings: Any) -> None:
        """Service reports not ready before models are loaded."""
        svc = _build_service(tmp_path, settings)
        assert svc.is_ready is False

    def test_run_prediction_catboost_only(
        self, tmp_path: Path, settings: Any, excel_path: Path
    ) -> None:
        """E2E with mock CatBoost: produces 24-row T+1 output with PREDICTION_COL."""
        svc = _build_service(tmp_path, settings, active_models=["catboost"])
        mock_cb = _make_mock_catboost()

        # Manually wire up internals to bypass file-based model loading
        from energy_forecast.data.loader import DataLoader
        from energy_forecast.features.pipeline import FeaturePipeline
        from energy_forecast.models.ensemble import EnsembleForecaster

        svc._data_loader = DataLoader(settings.data_loader)
        svc._feature_pipeline = FeaturePipeline(settings)

        ensemble_config = {
            "active_models": ["catboost"],
            "weights": {"catboost": 1.0, "prophet": 0.0, "tft": 0.0},
            "target_col": "consumption",
            "prophet_regressors": [],
            "mode": "weighted_average",
        }
        ensemble = EnsembleForecaster(ensemble_config)
        ensemble._catboost_model = mock_cb
        svc._ensemble = ensemble
        svc._models_loaded = True

        # Mock external data fetchers to return empty DataFrames
        with (
            patch.object(svc, "_fetch_epias_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_generation_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_weather_data", return_value=pd.DataFrame()),
        ):
            result = svc.run_prediction(excel_path)

        # T+1 output: 24 rows (GOP)
        assert len(result) == 24
        assert PREDICTION_COL in result.columns
        assert result[PREDICTION_COL].isna().sum() == 0
        assert (result[PREDICTION_COL] > 0).all()

    def test_run_prediction_two_model_ensemble(
        self, tmp_path: Path, settings: Any, excel_path: Path
    ) -> None:
        """E2E with CatBoost + Prophet: weighted average produces valid output."""
        from energy_forecast.data.loader import DataLoader
        from energy_forecast.features.pipeline import FeaturePipeline
        from energy_forecast.models.ensemble import EnsembleForecaster

        svc = _build_service(tmp_path, settings, active_models=["catboost", "prophet"])
        svc._data_loader = DataLoader(settings.data_loader)
        svc._feature_pipeline = FeaturePipeline(settings)

        ensemble_config = {
            "active_models": ["catboost", "prophet"],
            "weights": {"catboost": 0.6, "prophet": 0.4, "tft": 0.0},
            "target_col": "consumption",
            "prophet_regressors": [r.name for r in settings.prophet.regressors],
            "mode": "weighted_average",
        }
        ensemble = EnsembleForecaster(ensemble_config)
        ensemble._catboost_model = _make_mock_catboost()
        ensemble._prophet_model = _make_mock_prophet()
        svc._ensemble = ensemble
        svc._models_loaded = True

        with (
            patch.object(svc, "_fetch_epias_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_generation_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_weather_data", return_value=pd.DataFrame()),
        ):
            result = svc.run_prediction(excel_path)

        assert len(result) == 24
        assert PREDICTION_COL in result.columns
        assert result[PREDICTION_COL].isna().sum() == 0
        # Weighted average of 1200 and 1100 with 0.6/0.4
        expected = 0.6 * 1200.0 + 0.4 * 1100.0
        np.testing.assert_allclose(result[PREDICTION_COL].values, expected, atol=1.0)

    def test_run_prediction_output_has_datetime_index(
        self, tmp_path: Path, settings: Any, excel_path: Path
    ) -> None:
        """Output DataFrame has a DatetimeIndex."""
        from energy_forecast.data.loader import DataLoader
        from energy_forecast.features.pipeline import FeaturePipeline
        from energy_forecast.models.ensemble import EnsembleForecaster

        svc = _build_service(tmp_path, settings, active_models=["catboost"])
        svc._data_loader = DataLoader(settings.data_loader)
        svc._feature_pipeline = FeaturePipeline(settings)

        ensemble_config = {
            "active_models": ["catboost"],
            "weights": {"catboost": 1.0, "prophet": 0.0, "tft": 0.0},
            "target_col": "consumption",
            "mode": "weighted_average",
        }
        ensemble = EnsembleForecaster(ensemble_config)
        ensemble._catboost_model = _make_mock_catboost()
        svc._ensemble = ensemble
        svc._models_loaded = True

        with (
            patch.object(svc, "_fetch_epias_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_generation_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_weather_data", return_value=pd.DataFrame()),
        ):
            result = svc.run_prediction(excel_path)

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_run_prediction_metadata_attrs(
        self, tmp_path: Path, settings: Any, excel_path: Path
    ) -> None:
        """Result DataFrame has latency_ms and raw_predictions in attrs."""
        from energy_forecast.data.loader import DataLoader
        from energy_forecast.features.pipeline import FeaturePipeline
        from energy_forecast.models.ensemble import EnsembleForecaster

        svc = _build_service(tmp_path, settings, active_models=["catboost"])
        svc._data_loader = DataLoader(settings.data_loader)
        svc._feature_pipeline = FeaturePipeline(settings)

        ensemble_config = {
            "active_models": ["catboost"],
            "weights": {"catboost": 1.0, "prophet": 0.0, "tft": 0.0},
            "target_col": "consumption",
            "mode": "weighted_average",
        }
        ensemble = EnsembleForecaster(ensemble_config)
        ensemble._catboost_model = _make_mock_catboost()
        svc._ensemble = ensemble
        svc._models_loaded = True

        with (
            patch.object(svc, "_fetch_epias_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_generation_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_weather_data", return_value=pd.DataFrame()),
        ):
            result = svc.run_prediction(excel_path)

        assert "latency_ms" in result.attrs
        assert isinstance(result.attrs["latency_ms"], int)
        assert result.attrs["latency_ms"] >= 0
        assert "raw_predictions" in result.attrs

    def test_run_prediction_progress_callback(
        self, tmp_path: Path, settings: Any, excel_path: Path
    ) -> None:
        """Progress callback is called during prediction."""
        from energy_forecast.data.loader import DataLoader
        from energy_forecast.features.pipeline import FeaturePipeline
        from energy_forecast.models.ensemble import EnsembleForecaster

        svc = _build_service(tmp_path, settings, active_models=["catboost"])
        svc._data_loader = DataLoader(settings.data_loader)
        svc._feature_pipeline = FeaturePipeline(settings)

        ensemble_config = {
            "active_models": ["catboost"],
            "weights": {"catboost": 1.0, "prophet": 0.0, "tft": 0.0},
            "target_col": "consumption",
            "mode": "weighted_average",
        }
        ensemble = EnsembleForecaster(ensemble_config)
        ensemble._catboost_model = _make_mock_catboost()
        svc._ensemble = ensemble
        svc._models_loaded = True

        progress_messages: list[str] = []

        with (
            patch.object(svc, "_fetch_epias_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_generation_data", return_value=pd.DataFrame()),
            patch.object(svc, "_fetch_weather_data", return_value=pd.DataFrame()),
        ):
            result = svc.run_prediction(
                excel_path, progress_callback=progress_messages.append
            )

        assert len(progress_messages) >= 3  # at least loading, features, prediction
        assert any("complete" in m.lower() for m in progress_messages)

    def test_get_model_info_after_setup(
        self, tmp_path: Path, settings: Any
    ) -> None:
        """get_model_info returns correct data when models are loaded."""
        from energy_forecast.models.ensemble import EnsembleForecaster

        svc = _build_service(tmp_path, settings)
        ensemble = EnsembleForecaster({
            "active_models": ["catboost"],
            "weights": {"catboost": 1.0, "prophet": 0.0, "tft": 0.0},
        })
        svc._ensemble = ensemble
        svc._models_loaded = True

        info = svc.get_model_info()
        assert info["loaded"] is True
        assert "catboost" in info["active_models"]
        assert info["forecast_horizon"] == 48

    def test_get_model_info_not_loaded(
        self, tmp_path: Path, settings: Any
    ) -> None:
        """get_model_info returns loaded=False before setup."""
        svc = _build_service(tmp_path, settings)
        info = svc.get_model_info()
        assert info["loaded"] is False
