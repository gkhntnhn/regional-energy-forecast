"""Tests for prediction orchestration service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast.serving.exceptions import (
    FeaturePipelineError,
    ModelNotLoadedError,
    PredictionError,
)
from energy_forecast.serving.services.prediction_service import (
    PredictionService,
    PredictionServiceConfig,
)


@pytest.fixture
def pred_config(tmp_path: Path) -> PredictionServiceConfig:
    """Minimal prediction service config with tmp paths."""
    return PredictionServiceConfig(
        models_dir=tmp_path / "models",
        catboost_path=tmp_path / "models" / "catboost" / "model.cbm",
        prophet_path=tmp_path / "models" / "prophet",
        tft_path=tmp_path / "models" / "tft",
        ensemble_dir=None,
        forecast_horizon=48,
    )


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for PredictionService."""
    settings = MagicMock()
    settings.ensemble.active_models = ["catboost", "prophet"]
    settings.ensemble.weights.catboost = 0.6
    settings.ensemble.weights.prophet = 0.4
    settings.ensemble.weights.tft = 0.0
    settings.prophet.regressors = [MagicMock(name="temperature_2m")]
    settings.env.epias_username = "test"
    settings.env.epias_password = "test"
    settings.env.smtp_server = ""
    settings.env.smtp_port = 587
    settings.env.smtp_username = ""
    settings.env.smtp_password = ""
    settings.env.sender_email = ""
    settings.openmeteo = MagicMock()
    settings.region = MagicMock()
    settings.project.timezone = "Europe/Istanbul"
    return settings


@pytest.fixture
def service(
    pred_config: PredictionServiceConfig,
    mock_settings: MagicMock,
) -> PredictionService:
    """Create a PredictionService instance."""
    return PredictionService(pred_config, mock_settings)


class TestPredictionServiceInit:
    """Tests for PredictionService initialization."""

    def test_init_not_ready(self, service: PredictionService) -> None:
        """Service is not ready before load_models."""
        assert service.is_ready is False

    def test_warnings_empty_initially(self, service: PredictionService) -> None:
        """Warnings list is empty on init."""
        assert service.warnings == []

    def test_get_model_info_not_loaded(self, service: PredictionService) -> None:
        """Model info shows not loaded when models haven't been loaded."""
        info = service.get_model_info()
        assert info == {"loaded": False}


class TestPredictionServiceLoadModels:
    """Tests for load_models method."""

    @patch("energy_forecast.serving.services.prediction_service.EnsembleForecaster")
    @patch("energy_forecast.serving.services.prediction_service.FeaturePipeline")
    @patch("energy_forecast.serving.services.prediction_service.DataLoader")
    def test_load_models_success(
        self,
        mock_loader_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_ensemble_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Successful model loading sets is_ready to True."""
        mock_ensemble = mock_ensemble_cls.return_value
        mock_ensemble.active_models = ["catboost", "prophet"]

        service.load_models()

        assert service.is_ready is True
        mock_loader_cls.assert_called_once()
        mock_pipeline_cls.assert_called_once()
        mock_ensemble_cls.assert_called_once()

    @patch("energy_forecast.serving.services.prediction_service.EnsembleForecaster")
    @patch("energy_forecast.serving.services.prediction_service.FeaturePipeline")
    @patch("energy_forecast.serving.services.prediction_service.DataLoader")
    def test_load_models_failure_raises(
        self,
        mock_loader_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_ensemble_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Failed model loading raises ModelNotLoadedError."""
        mock_loader_cls.side_effect = RuntimeError("Config error")

        with pytest.raises(ModelNotLoadedError, match="Model loading failed"):
            service.load_models()

        assert service.is_ready is False

    @patch("energy_forecast.serving.services.prediction_service.EnsembleForecaster")
    @patch("energy_forecast.serving.services.prediction_service.FeaturePipeline")
    @patch("energy_forecast.serving.services.prediction_service.DataLoader")
    def test_get_model_info_after_load(
        self,
        mock_loader_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_ensemble_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Model info returns details after successful load."""
        mock_ensemble = mock_ensemble_cls.return_value
        mock_ensemble.active_models = ["catboost", "prophet"]
        mock_ensemble.weights = {"catboost": 0.6, "prophet": 0.4}

        service.load_models()
        info = service.get_model_info()

        assert info["loaded"] is True
        assert info["active_models"] == ["catboost", "prophet"]
        assert info["forecast_horizon"] == 48


class TestPredictionServiceRunPrediction:
    """Tests for run_prediction method."""

    def _make_ready_service(self, service: PredictionService) -> None:
        """Set up service internals as if load_models() succeeded."""
        service._models_loaded = True
        service._data_loader = MagicMock()
        service._feature_pipeline = MagicMock()
        service._ensemble = MagicMock()
        service._ensemble.active_models = ["catboost"]

    def test_run_prediction_not_loaded_raises(
        self, service: PredictionService, tmp_path: Path
    ) -> None:
        """Prediction without loaded models raises ModelNotLoadedError."""
        with pytest.raises(ModelNotLoadedError, match="not loaded"):
            service.run_prediction(tmp_path / "input.xlsx")

    @patch("energy_forecast.serving.services.prediction_service.OpenMeteoClient")
    @patch("energy_forecast.serving.services.prediction_service.EpiasClient")
    def test_run_prediction_success(
        self,
        mock_epias_cls: MagicMock,
        mock_weather_cls: MagicMock,
        service: PredictionService,
        tmp_path: Path,
    ) -> None:
        """Successful prediction returns DataFrame with consumption_mwh."""
        self._make_ready_service(service)

        # Mock data loader
        idx = pd.date_range("2025-12-30", periods=72, freq="h", tz="Europe/Istanbul")
        consumption_df = pd.DataFrame({"consumption": range(24)}, index=idx[:24])
        extended_df = pd.DataFrame({"consumption": [1.0] * 72}, index=idx)

        service._data_loader.load_excel.return_value = consumption_df
        service._data_loader.extend_for_forecast.return_value = extended_df

        # Mock EPIAS client (context manager)
        mock_epias_ctx = MagicMock()
        mock_epias_ctx.fetch.return_value = pd.DataFrame(index=idx)
        mock_epias_ctx.fetch_generation.return_value = pd.DataFrame(index=idx)
        mock_epias_cls.return_value.__enter__ = MagicMock(return_value=mock_epias_ctx)
        mock_epias_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Mock weather client (context manager)
        mock_weather_ctx = MagicMock()
        mock_weather_ctx.fetch_historical.return_value = pd.DataFrame(index=idx)
        mock_weather_ctx.fetch_forecast.return_value = pd.DataFrame(index=idx)
        mock_weather_cls.return_value.__enter__ = MagicMock(return_value=mock_weather_ctx)
        mock_weather_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Mock feature pipeline
        feature_df = pd.DataFrame({"consumption_mwh": [1000.0] * 72}, index=idx)
        service._feature_pipeline.run.return_value = feature_df

        # Mock ensemble predict
        pred_idx = idx[24:]
        predictions = pd.DataFrame({"consumption_mwh": [1100.0] * 48}, index=pred_idx)
        service._ensemble.predict.return_value = predictions

        result = service.run_prediction(tmp_path / "input.xlsx")

        assert isinstance(result, pd.DataFrame)
        assert "consumption_mwh" in result.columns
        assert "latency_ms" in result.attrs

    @patch("energy_forecast.serving.services.prediction_service.OpenMeteoClient")
    @patch("energy_forecast.serving.services.prediction_service.EpiasClient")
    def test_feature_pipeline_error_raises(
        self,
        mock_epias_cls: MagicMock,
        mock_weather_cls: MagicMock,
        service: PredictionService,
        tmp_path: Path,
    ) -> None:
        """Feature pipeline failure raises FeaturePipelineError."""
        self._make_ready_service(service)

        idx = pd.date_range("2025-12-30", periods=72, freq="h", tz="Europe/Istanbul")
        service._data_loader.load_excel.return_value = pd.DataFrame(
            {"consumption": range(24)}, index=idx[:24]
        )
        service._data_loader.extend_for_forecast.return_value = pd.DataFrame(
            {"consumption": [1.0] * 72}, index=idx
        )

        mock_epias_ctx = MagicMock()
        mock_epias_ctx.fetch.return_value = pd.DataFrame(index=idx)
        mock_epias_ctx.fetch_generation.return_value = pd.DataFrame(index=idx)
        mock_epias_cls.return_value.__enter__ = MagicMock(return_value=mock_epias_ctx)
        mock_epias_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_weather_ctx = MagicMock()
        mock_weather_ctx.fetch_historical.return_value = pd.DataFrame(index=idx)
        mock_weather_ctx.fetch_forecast.return_value = pd.DataFrame(index=idx)
        mock_weather_cls.return_value.__enter__ = MagicMock(return_value=mock_weather_ctx)
        mock_weather_cls.return_value.__exit__ = MagicMock(return_value=False)

        service._feature_pipeline.run.side_effect = ValueError("Bad features")

        with pytest.raises(FeaturePipelineError, match="Feature pipeline failed"):
            service.run_prediction(tmp_path / "input.xlsx")

    @patch("energy_forecast.serving.services.prediction_service.OpenMeteoClient")
    @patch("energy_forecast.serving.services.prediction_service.EpiasClient")
    def test_unexpected_error_raises_prediction_error(
        self,
        mock_epias_cls: MagicMock,
        mock_weather_cls: MagicMock,
        service: PredictionService,
        tmp_path: Path,
    ) -> None:
        """Unexpected errors wrapped in PredictionError."""
        self._make_ready_service(service)

        service._data_loader.load_excel.side_effect = RuntimeError("Disk full")

        with pytest.raises(PredictionError, match="Prediction failed"):
            service.run_prediction(tmp_path / "input.xlsx")

    def test_progress_callback_called(
        self, service: PredictionService, tmp_path: Path
    ) -> None:
        """Progress callback receives status messages."""
        self._make_ready_service(service)

        service._data_loader.load_excel.side_effect = RuntimeError("test")
        callback = MagicMock()

        with pytest.raises(PredictionError):
            service.run_prediction(tmp_path / "input.xlsx", progress_callback=callback)

        # At least the first step message should have been sent
        callback.assert_called()


class TestPredictionServiceFetchHelpers:
    """Tests for EPIAS/weather fetch helper methods."""

    def _make_ready_service(self, service: PredictionService) -> None:
        """Set up service internals."""
        service._models_loaded = True
        service._data_loader = MagicMock()
        service._feature_pipeline = MagicMock()
        service._ensemble = MagicMock()

    @patch("energy_forecast.serving.services.prediction_service.EpiasClient")
    def test_epias_fetch_failure_returns_empty_df(
        self,
        mock_epias_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """EPIAS fetch failure returns empty DataFrame and logs warning."""
        self._make_ready_service(service)

        mock_ctx = MagicMock()
        mock_ctx.fetch.side_effect = ConnectionError("Network error")
        mock_epias_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_epias_cls.return_value.__exit__ = MagicMock(return_value=False)

        idx = pd.date_range("2025-01-01", periods=48, freq="h")
        df = pd.DataFrame({"consumption": range(48)}, index=idx)

        result = service._fetch_epias_data(df)
        assert len(result) == 48
        assert len(service.warnings) == 1
        assert "EPIAS fetch failed" in service.warnings[0]

    @patch("energy_forecast.serving.services.prediction_service.EpiasClient")
    def test_epias_auth_error_propagates(
        self,
        mock_epias_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """EPIAS auth error is NOT caught — it propagates."""
        self._make_ready_service(service)

        from energy_forecast.data.exceptions import EpiasAuthError

        mock_ctx = MagicMock()
        mock_ctx.fetch.side_effect = EpiasAuthError("Bad credentials")
        mock_epias_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_epias_cls.return_value.__exit__ = MagicMock(return_value=False)

        idx = pd.date_range("2025-01-01", periods=48, freq="h")
        df = pd.DataFrame({"consumption": range(48)}, index=idx)

        with pytest.raises(EpiasAuthError):
            service._fetch_epias_data(df)

    @patch("energy_forecast.serving.services.prediction_service.EpiasClient")
    def test_generation_fetch_failure_returns_empty_df(
        self,
        mock_epias_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Generation fetch failure returns empty DataFrame."""
        self._make_ready_service(service)

        mock_ctx = MagicMock()
        mock_ctx.fetch_generation.side_effect = ConnectionError("Timeout")
        mock_epias_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_epias_cls.return_value.__exit__ = MagicMock(return_value=False)

        idx = pd.date_range("2025-01-01", periods=48, freq="h")
        df = pd.DataFrame({"consumption": range(48)}, index=idx)

        result = service._fetch_generation_data(df)
        assert len(result) == 48
        assert "Generation fetch failed" in service.warnings[0]

    @patch("energy_forecast.serving.services.prediction_service.OpenMeteoClient")
    def test_weather_fetch_combines_historical_and_forecast(
        self,
        mock_weather_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Weather fetch combines historical and forecast data."""
        self._make_ready_service(service)

        idx = pd.date_range("2025-01-01", periods=48, freq="h")
        hist_df = pd.DataFrame({"temperature_2m": [10.0] * 24}, index=idx[:24])
        fc_df = pd.DataFrame({"temperature_2m": [15.0] * 24}, index=idx[24:])

        mock_ctx = MagicMock()
        mock_ctx.fetch_historical.return_value = hist_df
        mock_ctx.fetch_forecast.return_value = fc_df
        mock_weather_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_weather_cls.return_value.__exit__ = MagicMock(return_value=False)

        df = pd.DataFrame({"consumption": range(48)}, index=idx)
        result = service._fetch_weather_data(df)

        assert len(result) == 48
        assert result.iloc[0]["temperature_2m"] == 10.0
        assert result.iloc[-1]["temperature_2m"] == 15.0

    @patch("energy_forecast.serving.services.prediction_service.OpenMeteoClient")
    def test_weather_fetch_only_historical(
        self,
        mock_weather_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Weather returns only historical when forecast fails."""
        self._make_ready_service(service)

        idx = pd.date_range("2025-01-01", periods=48, freq="h")
        hist_df = pd.DataFrame({"temperature_2m": [10.0] * 48}, index=idx)

        mock_ctx = MagicMock()
        mock_ctx.fetch_historical.return_value = hist_df
        mock_ctx.fetch_forecast.side_effect = ConnectionError("API down")
        mock_weather_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_weather_cls.return_value.__exit__ = MagicMock(return_value=False)

        df = pd.DataFrame({"consumption": range(48)}, index=idx)
        result = service._fetch_weather_data(df)

        assert len(result) == 48

    @patch("energy_forecast.serving.services.prediction_service.OpenMeteoClient")
    def test_weather_both_fail_returns_empty(
        self,
        mock_weather_cls: MagicMock,
        service: PredictionService,
    ) -> None:
        """Both weather fetches fail returns empty DataFrame."""
        self._make_ready_service(service)

        mock_ctx = MagicMock()
        mock_ctx.fetch_historical.side_effect = ConnectionError("fail")
        mock_ctx.fetch_forecast.side_effect = ConnectionError("fail")
        mock_weather_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_weather_cls.return_value.__exit__ = MagicMock(return_value=False)

        idx = pd.date_range("2025-01-01", periods=48, freq="h")
        df = pd.DataFrame({"consumption": range(48)}, index=idx)
        result = service._fetch_weather_data(df)

        assert len(result) == 48


class TestPrepareOutput:
    """Tests for _prepare_output method."""

    def test_output_filters_to_t_plus_1(
        self, service: PredictionService
    ) -> None:
        """Output should only contain T+1 day rows."""
        # last_data_point = T-1 23:00 (Dec 30 23:00)
        # T+1 starts at Jan 1 00:00
        last_data_point = pd.Timestamp("2025-12-30 23:00", tz="Europe/Istanbul")
        idx = pd.date_range("2025-12-31", periods=48, freq="h", tz="Europe/Istanbul")
        predictions = pd.DataFrame({"consumption_mwh": [1000.0] * 48}, index=idx)

        result = service._prepare_output(predictions, last_data_point)

        # Only T+1 day (Jan 1 00:00-23:00) = 24 rows
        assert len(result) == 24
        assert result.index.min() == pd.Timestamp("2026-01-01 00:00", tz="Europe/Istanbul")
