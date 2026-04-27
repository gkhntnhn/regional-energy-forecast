"""Phase-helper unit tests for run_prediction decomposition (item 236)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.serving.exceptions import (
    FeaturePipelineError,
    PredictionError,
)
from energy_forecast.serving.services.prediction_service import (
    PredictionService,
    PredictionServiceConfig,
)


@pytest.fixture
def service(tmp_path: Path) -> PredictionService:
    """A minimal PredictionService — phase helpers don't need load_models()."""
    config = PredictionServiceConfig(
        models_dir=tmp_path / "models",
        catboost_path=tmp_path / "models" / "catboost.cbm",
        tft_path=tmp_path / "models" / "tft",
        ensemble_dir=None,
        forecast_horizon=48,
    )
    settings = MagicMock()
    return PredictionService(config, settings)


def _hourly_df(start: str, periods: int, value_col: str = "consumption") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="h")
    return pd.DataFrame({value_col: np.arange(periods, dtype=float)}, index=idx)


class TestLoadConsumptionAndExtend:
    """Phase 1: Excel load + extend."""

    def test_returns_three_tuple(self, service: PredictionService, tmp_path: Path) -> None:
        consumption = _hourly_df("2025-01-01", periods=24)
        extended = _hourly_df("2025-01-01", periods=72)  # 24 + 48
        loader = MagicMock()
        loader.load_excel.return_value = consumption
        loader.extend_for_forecast.return_value = extended
        service._data_loader = loader

        result = service._load_consumption_and_extend(tmp_path / "in.xlsx")

        cdf, last_ts, ext = result
        assert cdf is consumption
        assert last_ts == consumption.index.max()
        assert ext is extended
        loader.extend_for_forecast.assert_called_once_with(consumption, horizon_hours=48)


class TestFetchMarketData:
    """Phase 2: EPIAS market + generation + meta."""

    def test_meta_builds_when_epias_non_empty(self, service: PredictionService) -> None:
        extended = _hourly_df("2025-01-01", periods=72)
        epias = pd.DataFrame(
            {"rtc": [100.0, 110.0, 120.0]},
            index=pd.date_range("2024-12-30", periods=3, freq="h"),
        )
        with patch.object(service, "_fetch_epias_data", return_value=epias), \
             patch.object(service, "_fetch_generation_data", return_value=pd.DataFrame()):
            merged, meta = service._fetch_market_data(extended)

        assert "data_range" in meta
        assert meta["row_count"] == 3
        assert meta["last_values"] == {"rtc": 120.0}
        assert meta["nan_summary"] == {"rtc": 0}
        assert "rtc" in merged.columns

    def test_meta_empty_when_epias_empty(self, service: PredictionService) -> None:
        extended = _hourly_df("2025-01-01", periods=72)
        with patch.object(service, "_fetch_epias_data", return_value=pd.DataFrame()), \
             patch.object(service, "_fetch_generation_data", return_value=pd.DataFrame()):
            merged, meta = service._fetch_market_data(extended)

        assert meta == {}
        assert len(merged) == len(extended)

    def test_generation_joined_when_present(self, service: PredictionService) -> None:
        extended = _hourly_df("2025-01-01", periods=24)
        gen = pd.DataFrame(
            {"gen_total": np.arange(24, dtype=float) * 100},
            index=pd.date_range("2025-01-01", periods=24, freq="h"),
        )
        with patch.object(service, "_fetch_epias_data", return_value=pd.DataFrame()), \
             patch.object(service, "_fetch_generation_data", return_value=gen):
            merged, _ = service._fetch_market_data(extended)
        assert "gen_total" in merged.columns

    def test_nan_summary_counts_correctly(self, service: PredictionService) -> None:
        extended = _hourly_df("2025-01-01", periods=72)
        epias = pd.DataFrame(
            {"rtc": [100.0, np.nan, 120.0], "dam": [np.nan, np.nan, 50.0]},
            index=pd.date_range("2024-12-30", periods=3, freq="h"),
        )
        with patch.object(service, "_fetch_epias_data", return_value=epias), \
             patch.object(service, "_fetch_generation_data", return_value=pd.DataFrame()):
            _, meta = service._fetch_market_data(extended)
        assert meta["nan_summary"] == {"rtc": 1, "dam": 2}


class TestFetchAndApplyWeather:
    """Phase 3: Weather fetch + ffill (numeric + categorical only)."""

    def test_weather_columns_ffilled(self, service: PredictionService) -> None:
        extended = _hourly_df("2025-01-01", periods=4)
        merged = extended.copy()
        weather = pd.DataFrame(
            {"temperature_2m": [20.0, np.nan, np.nan, 22.0]},
            index=extended.index,
        )
        with patch.object(service, "_fetch_weather_data", return_value=weather):
            result_df, weather_out = service._fetch_and_apply_weather(merged, extended)
        assert weather_out is weather
        # ffill propagates 20 forward through the gap
        assert result_df["temperature_2m"].iloc[1] == 20.0
        assert result_df["temperature_2m"].iloc[2] == 20.0

    def test_categorical_weather_ffilled(self, service: PredictionService) -> None:
        extended = _hourly_df("2025-01-01", periods=4)
        merged = extended.copy()
        weather = pd.DataFrame(
            {"weather_code": [1.0, np.nan, np.nan, 2.0]},
            index=extended.index,
        )
        with patch.object(service, "_fetch_weather_data", return_value=weather):
            result_df, _ = service._fetch_and_apply_weather(merged, extended)
        assert result_df["weather_code"].iloc[1] == 1.0
        assert result_df["weather_code"].iloc[2] == 1.0

    def test_consumption_column_not_ffilled(self, service: PredictionService) -> None:
        """Critical leakage guard: consumption never gets ffilled."""
        idx = pd.date_range("2025-01-01", periods=4, freq="h")
        merged = pd.DataFrame({"consumption": [100.0, np.nan, np.nan, 200.0]}, index=idx)
        extended = merged.copy()
        weather = pd.DataFrame({"temperature_2m": [20.0, 21.0, 22.0, 23.0]}, index=idx)
        with patch.object(service, "_fetch_weather_data", return_value=weather):
            result_df, _ = service._fetch_and_apply_weather(merged, extended)
        # consumption NaN preserved
        assert pd.isna(result_df["consumption"].iloc[1])
        assert pd.isna(result_df["consumption"].iloc[2])


class TestRunFeaturePipelineWithHolidays:
    """Phase 4: Feature pipeline (DB holidays branch)."""

    def test_passes_through_when_no_db_session(self, service: PredictionService) -> None:
        merged = _hourly_df("2025-01-01", periods=10)
        out = _hourly_df("2025-01-01", periods=10)
        out["feature_a"] = 1.0
        pipeline = MagicMock()
        pipeline.run.return_value = out
        service._feature_pipeline = pipeline
        service._sync_session_factory = None

        result = service._run_feature_pipeline_with_holidays(merged)

        assert result is out
        assert service._last_feature_count == len(out.columns)
        pipeline.run.assert_called_once_with(merged)

    def test_wraps_pipeline_failure_in_feature_pipeline_error(
        self, service: PredictionService,
    ) -> None:
        merged = _hourly_df("2025-01-01", periods=10)
        pipeline = MagicMock()
        pipeline.run.side_effect = ValueError("schema mismatch")
        service._feature_pipeline = pipeline
        service._sync_session_factory = None

        with pytest.raises(FeaturePipelineError, match="schema mismatch"):
            service._run_feature_pipeline_with_holidays(merged)


class TestExtractForecastSplit:
    """Phase 5: Forecast/historical split by last_timestamp."""

    def test_returns_three_dataframes(self, service: PredictionService) -> None:
        idx = pd.date_range("2025-01-01", periods=72, freq="h")
        features = pd.DataFrame({"x": np.arange(72)}, index=idx)
        last_ts = idx[23]  # 24h history, 48h forecast

        fc, hist, mask = service._extract_forecast_split(features, last_ts)

        assert len(fc) == 48
        assert len(hist) == 24
        assert mask.sum() == 48

    def test_raises_when_no_forecast_rows(self, service: PredictionService) -> None:
        idx = pd.date_range("2025-01-01", periods=24, freq="h")
        features = pd.DataFrame({"x": np.arange(24)}, index=idx)
        last_ts = idx[-1]  # nothing after last_ts

        with pytest.raises(PredictionError, match="No forecast rows"):
            service._extract_forecast_split(features, last_ts)

    def test_warns_on_unexpected_horizon(
        self, service: PredictionService,
    ) -> None:
        idx = pd.date_range("2025-01-01", periods=30, freq="h")
        features = pd.DataFrame({"x": np.arange(30)}, index=idx)
        last_ts = idx[23]  # only 6 forecast rows, expected 48

        # Should not raise, just warn (check no PredictionError)
        fc, _hist, _ = service._extract_forecast_split(features, last_ts)
        assert len(fc) == 6


class TestGeneratePredictions:
    """Phase 6: Ensemble predict + raw copy."""

    def test_returns_display_and_raw_copies(self, service: PredictionService) -> None:
        forecast = _hourly_df("2025-01-02", periods=48)
        history = _hourly_df("2025-01-01", periods=24)
        out = pd.DataFrame(
            {"consumption_mwh": np.arange(48, dtype=float)},
            index=forecast.index,
        )
        ensemble = MagicMock()
        ensemble.predict.return_value = out
        service._ensemble = ensemble

        display, raw = service._generate_predictions(forecast, history)

        ensemble.predict.assert_called_once_with(forecast, history=history)
        assert display.equals(out)
        assert raw.equals(out)
        # raw is a copy, not a reference (so DB persistence won't be mutated)
        assert raw is not display


class TestBuildResponse:
    """Phase 7: Output formatting + metadata attrs."""

    def test_attaches_all_metadata(self, service: PredictionService) -> None:
        idx = pd.date_range("2025-01-02", periods=48, freq="h")
        predictions = pd.DataFrame({"consumption_mwh": np.arange(48.0)}, index=idx)
        raw = predictions.copy()
        weather = pd.DataFrame({"wth_temp": np.arange(48.0)}, index=idx)
        features = pd.DataFrame({"x": np.arange(48)}, index=idx)
        forecast_mask = np.array([True] * 48)
        last_ts = pd.Timestamp("2025-01-01 23:00")

        with patch.object(service, "_prepare_output", return_value=predictions.copy()):
            result = service._build_response(
                predictions, raw, last_ts, latency_ms=123.4,
                weather_df=weather, epias_meta={"row_count": 100},
                features_df=features, forecast_mask=forecast_mask,
            )

        assert result.attrs["latency_ms"] == 123
        assert result.attrs["weather_data"] is weather
        assert result.attrs["epias_snapshot"] == {"row_count": 100}
        assert result.attrs["features_df"] is features
        assert result.attrs["raw_predictions"] is raw
        # forecast_mask attached as-is (any boolean array shape)
        assert (result.attrs["forecast_mask"] == forecast_mask).all()
