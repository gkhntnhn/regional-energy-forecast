"""Unit tests for Pandera data schemas."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

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


class TestRawExcelSchemaNegative:
    """Negative tests for RawExcelSchema validation."""

    def test_missing_date_column_raises(self) -> None:
        """Missing 'date' column raises SchemaError."""
        df = pd.DataFrame({"time": [0], "consumption": [1000.0]})
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_time_negative_raises(self) -> None:
        """time=-1 is below ge=0 boundary."""
        df = pd.DataFrame(
            {"date": ["2024-01-01"], "time": [-1], "consumption": [1000.0]}
        )
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_time_24_raises(self) -> None:
        """time=24 is above le=23 boundary."""
        df = pd.DataFrame(
            {"date": ["2024-01-01"], "time": [24], "consumption": [1000.0]}
        )
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_consumption_exceeds_max_raises(self) -> None:
        """consumption > 10000 violates le=10000 constraint."""
        df = pd.DataFrame(
            {"date": ["2024-01-01"], "time": [0], "consumption": [10001.0]}
        )
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_non_numeric_consumption_raises(self) -> None:
        """Non-numeric consumption that cannot be coerced raises SchemaError."""
        df = pd.DataFrame(
            {"date": ["2024-01-01"], "time": [0], "consumption": ["not_a_number"]}
        )
        with pytest.raises((SchemaError, SchemaErrors)):
            RawExcelSchema.validate(df)


class TestConsumptionSchema:
    """Tests for ConsumptionSchema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid consumption data with DatetimeIndex passes."""
        idx = pd.date_range("2024-01-01", periods=24, freq="h", name="datetime")
        df = pd.DataFrame({"consumption": range(24)}, index=idx, dtype=float)
        validated = ConsumptionSchema.validate(df)
        assert len(validated) == 24


class TestConsumptionSchemaNegative:
    """Negative tests for ConsumptionSchema validation."""

    def test_range_index_raises(self) -> None:
        """RangeIndex instead of DatetimeIndex raises SchemaError."""
        df = pd.DataFrame({"consumption": [100.0, 200.0, 300.0]})
        with pytest.raises((SchemaError, SchemaErrors)):
            ConsumptionSchema.validate(df)

    def test_missing_consumption_column_raises(self) -> None:
        """Missing 'consumption' column raises SchemaError."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]}, index=idx)
        with pytest.raises(SchemaError):
            ConsumptionSchema.validate(df)

    def test_negative_consumption_raises(self) -> None:
        """Negative consumption violates ge=0.0 constraint."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame({"consumption": [-1.0, 200.0, 300.0]}, index=idx)
        with pytest.raises(SchemaError):
            ConsumptionSchema.validate(df)

    def test_empty_dataframe_raises(self) -> None:
        """Empty DataFrame with correct schema still validates (0 rows)."""
        idx = pd.DatetimeIndex([], name="datetime")
        df = pd.DataFrame({"consumption": pd.Series([], dtype=float)}, index=idx)
        # Empty DataFrame with correct structure passes Pandera (no rows to violate)
        validated = ConsumptionSchema.validate(df)
        assert len(validated) == 0


class TestEpiasSchema:
    """Tests for EpiasSchema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid EPIAS data with all 4 active columns passes."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame(
            {
                "Real_Time_Consumption": [100.0, 200.0, 300.0],
                "DAM_Purchase": [100.0, 200.0, 300.0],
                "Bilateral_Agreement_Purchase": [100.0, 200.0, 300.0],
                "Load_Forecast": [100.0, 200.0, 300.0],
            },
            index=idx,
        )
        validated = EpiasSchema.validate(df)
        assert len(validated) == 3


class TestEpiasSchemaNegative:
    """Negative tests for EpiasSchema validation."""

    def test_range_index_raises(self) -> None:
        """RangeIndex instead of DatetimeIndex raises SchemaError."""
        df = pd.DataFrame(
            {
                "Real_Time_Consumption": [100.0],
                "DAM_Purchase": [100.0],
                "Bilateral_Agreement_Purchase": [100.0],
                "Load_Forecast": [100.0],
            }
        )
        with pytest.raises((SchemaError, SchemaErrors)):
            EpiasSchema.validate(df)

    def test_missing_required_column_raises(self) -> None:
        """Missing required column (DAM_Purchase) raises SchemaError."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame(
            {
                "Real_Time_Consumption": [100.0, 200.0, 300.0],
                # DAM_Purchase missing
                "Bilateral_Agreement_Purchase": [100.0, 200.0, 300.0],
                "Load_Forecast": [100.0, 200.0, 300.0],
            },
            index=idx,
        )
        with pytest.raises(SchemaError):
            EpiasSchema.validate(df)

    def test_extra_columns_allowed(self) -> None:
        """Extra columns pass because strict=False (FDPP backward compat)."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame(
            {
                "Real_Time_Consumption": [100.0, 200.0, 300.0],
                "DAM_Purchase": [100.0, 200.0, 300.0],
                "Bilateral_Agreement_Purchase": [100.0, 200.0, 300.0],
                "Load_Forecast": [100.0, 200.0, 300.0],
                "FDPP_legacy": [50.0, 60.0, 70.0],
            },
            index=idx,
        )
        validated = EpiasSchema.validate(df)
        assert len(validated) == 3

    def test_empty_dataframe_raises(self) -> None:
        """Empty DataFrame missing required columns raises SchemaError."""
        idx = pd.DatetimeIndex([], name="datetime")
        df = pd.DataFrame(index=idx)
        with pytest.raises(SchemaError):
            EpiasSchema.validate(df)


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


class TestWeatherSchemaNegative:
    """Negative tests for WeatherSchema validation."""

    def test_range_index_raises(self) -> None:
        """RangeIndex instead of DatetimeIndex raises SchemaError."""
        df = pd.DataFrame({"temperature_2m": [10.0, 20.0]})
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_missing_critical_column_raises(self) -> None:
        """Missing temperature_2m column raises SchemaError."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        # Only provide a subset — temperature_2m intentionally missing
        df = pd.DataFrame(
            {
                "relative_humidity_2m": [50.0, 60.0, 70.0],
                "wind_speed_10m": [5.0, 10.0, 15.0],
            },
            index=idx,
        )
        with pytest.raises(SchemaError):
            WeatherSchema.validate(df)

    def test_all_columns_missing_raises(self) -> None:
        """DataFrame with only index and no weather columns raises SchemaError."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame(index=idx)
        with pytest.raises(SchemaError):
            WeatherSchema.validate(df)

    def test_nullable_columns_accept_nan(self) -> None:
        """All weather columns are nullable — NaN values should pass."""
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
            {col: [float("nan")] * 3 for col in columns},
            index=idx,
        )
        validated = WeatherSchema.validate(df)
        assert len(validated) == 3
