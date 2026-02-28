"""Integration test: real YAML config → feature pipeline → validate output.

This tests the full feature pipeline with production YAML configs (not defaults),
including generation features, and validates the output against leakage rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import load_config
from energy_forecast.features.pipeline import FeaturePipeline


@pytest.fixture()
def configs_dir() -> Path:
    """Return the project configs directory."""
    return Path(__file__).parent.parent.parent / "configs"


@pytest.fixture()
def settings(configs_dir: Path):  # noqa: ANN201
    """Load real Settings from project YAML files."""
    return load_config(configs_dir)


@pytest.fixture()
def full_df(settings) -> pd.DataFrame:  # noqa: ANN001
    """Realistic 720-row DataFrame with consumption + weather + EPIAS + generation."""
    rng = np.random.default_rng(42)
    n = 720  # 30 days
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    data: dict[str, Any] = {}

    # Consumption
    data["consumption"] = 800.0 + rng.random(n) * 400

    # Weather (11 variables)
    data["temperature_2m"] = rng.uniform(-5, 35, n)
    data["relative_humidity_2m"] = rng.uniform(20, 95, n)
    data["dew_point_2m"] = rng.uniform(-10, 20, n)
    data["apparent_temperature"] = rng.uniform(-10, 40, n)
    data["precipitation"] = rng.exponential(1.0, n)
    data["snow_depth"] = rng.uniform(0, 5, n)
    data["weather_code"] = rng.choice([0, 1, 2, 3, 45, 61, 71, 95], n).astype(float)
    data["surface_pressure"] = rng.uniform(990, 1030, n)
    data["wind_speed_10m"] = rng.uniform(0, 60, n)
    data["wind_direction_10m"] = rng.uniform(0, 360, n)
    data["shortwave_radiation"] = rng.uniform(0, 800, n)

    # EPIAS market (5 variables)
    for var in settings.features.epias.variables:
        data[var] = 500.0 + rng.random(n) * 1000

    # EPIAS generation (17 variables)
    for var in settings.features.epias.generation.variables:
        if var == "gen_total":
            data[var] = 2000.0 + rng.random(n) * 1000
        else:
            data[var] = 100.0 + rng.random(n) * 500

    return pd.DataFrame(data, index=idx).rename_axis("datetime")


class TestPipelineE2E:
    """End-to-end feature pipeline tests with real configs."""

    def test_pipeline_runs_with_real_config(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """Full pipeline executes without error using production YAML config."""
        pipeline = FeaturePipeline(settings)
        result = pipeline.run(full_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(full_df)

    def test_output_has_many_features(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """Pipeline generates 100+ features from raw input."""
        pipeline = FeaturePipeline(settings)
        result = pipeline.run(full_df)
        # Raw input has ~34 columns; output should be much larger
        assert result.shape[1] > 100, (
            f"Expected 100+ features, got {result.shape[1]}"
        )

    def test_no_raw_epias_in_output(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """Raw EPIAS market + generation columns are dropped."""
        pipeline = FeaturePipeline(settings)
        result = pipeline.run(full_df)

        raw_market = settings.features.epias.variables
        raw_gen = settings.features.epias.generation.variables
        for col in raw_market + raw_gen:
            assert col not in result.columns, f"Raw column {col} not dropped"

    def test_no_leakage_in_consumption_epias_lags(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """Consumption and EPIAS lag features have periods >= 48.

        Weather/solar lags are exempt — they are known future inputs.
        """
        pipeline = FeaturePipeline(settings)
        result = pipeline.run(full_df)

        # Only check consumption and EPIAS/generation lags (not weather/solar)
        leakage_prefixes = ("consumption_", "FDPP_", "Real_Time_", "DAM_", "Bilateral_",
                            "Load_Forecast_", "gen_")
        lag_cols = [
            c for c in result.columns
            if "_lag_" in c and c.startswith(leakage_prefixes)
        ]
        assert len(lag_cols) > 0, "No consumption/EPIAS lag features found"
        for col in lag_cols:
            period_str = col.rsplit("_", maxsplit=1)[-1]
            try:
                period = int(period_str)
            except ValueError:
                continue
            assert period >= 48, f"Lag {col} has period {period} < 48 — leakage risk!"

    def test_generation_composite_features(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """renewable_ratio_lag_48 and thermal_ratio_lag_48 are in output."""
        pipeline = FeaturePipeline(settings)
        result = pipeline.run(full_df)
        assert "renewable_ratio_lag_48" in result.columns
        assert "thermal_ratio_lag_48" in result.columns

    def test_historical_forecast_split(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """Simulates the historical/forecast split used in production.

        Last 48 rows = forecast (consumption NaN), rest = historical.
        Validates both splits have features.
        """
        # Set last 48 rows consumption to NaN (simulating forecast)
        df = full_df.copy()
        df.loc[df.index[-48:], "consumption"] = np.nan

        pipeline = FeaturePipeline(settings)
        result = pipeline.run(df)

        historical = result.iloc[:-48]
        forecast = result.iloc[-48:]

        assert len(historical) == len(df) - 48
        assert len(forecast) == 48

        # Historical should have consumption-derived features
        cons_cols = [c for c in result.columns if c.startswith("consumption_lag_")]
        assert len(cons_cols) > 0

        # Forecast consumption lags should NOT be NaN (derived from past data)
        for col in cons_cols:
            if forecast[col].notna().any():
                break
        else:
            pytest.fail("All consumption lag features are NaN in forecast — lag too short?")

    def test_datetime_index_preserved(
        self,
        settings,  # noqa: ANN001
        full_df: pd.DataFrame,
    ) -> None:
        """DatetimeIndex is preserved through the pipeline."""
        pipeline = FeaturePipeline(settings)
        result = pipeline.run(full_df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "datetime"
