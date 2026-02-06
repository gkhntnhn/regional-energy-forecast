"""Unit tests for Pandera data schemas."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from energy_forecast.data.schemas import (
    ConsumptionSchema,
    EpiasSchema,
    RawExcelSchema,
    WeatherSchema,
)


class TestRawExcelSchema:
    """Tests for RawExcelSchema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid raw Excel data passes schema validation."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01"],
                "time": [0, 1],
                "consumption": [1000.0, 1100.0],
            }
        )
        validated = RawExcelSchema.validate(df)
        assert len(validated) == 2

    def test_invalid_time_raises(self) -> None:
        """Time value outside 0-23 raises SchemaError."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "time": [25],
                "consumption": [1000.0],
            }
        )
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_negative_consumption_raises(self) -> None:
        """Negative consumption raises SchemaError."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "time": [0],
                "consumption": [-10.0],
            }
        )
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)


class TestConsumptionSchema:
    """Tests for ConsumptionSchema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid consumption data with DatetimeIndex passes."""
        idx = pd.date_range("2024-01-01", periods=24, freq="h", name="datetime")
        df = pd.DataFrame({"consumption": range(24)}, index=idx, dtype=float)
        validated = ConsumptionSchema.validate(df)
        assert len(validated) == 24


class TestEpiasSchema:
    """Tests for EpiasSchema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid EPIAS data with all 5 columns passes."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame(
            {
                "FDPP": [100.0, 200.0, 300.0],
                "Real_Time_Consumption": [100.0, 200.0, 300.0],
                "DAM_Purchase": [100.0, 200.0, 300.0],
                "Bilateral_Agreement_Purchase": [100.0, 200.0, 300.0],
                "Load_Forecast": [100.0, 200.0, 300.0],
            },
            index=idx,
        )
        validated = EpiasSchema.validate(df)
        assert len(validated) == 3


class TestWeatherSchema:
    """Tests for WeatherSchema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid weather data with 11 variables passes."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        columns = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "snow_depth",
            "weather_code",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
        ]
        df = pd.DataFrame(
            {col: [1.0, 2.0, 3.0] for col in columns},
            index=idx,
        )
        validated = WeatherSchema.validate(df)
        assert len(validated) == 3
