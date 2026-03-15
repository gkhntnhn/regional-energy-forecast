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
        df = pd.DataFrame({"date": ["2024-01-01"], "time": [-1], "consumption": [1000.0]})
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_time_24_raises(self) -> None:
        """time=24 is above le=23 boundary."""
        df = pd.DataFrame({"date": ["2024-01-01"], "time": [24], "consumption": [1000.0]})
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_consumption_exceeds_max_raises(self) -> None:
        """consumption > 10000 violates le=10000 constraint."""
        df = pd.DataFrame({"date": ["2024-01-01"], "time": [0], "consumption": [10001.0]})
        with pytest.raises(SchemaError):
            RawExcelSchema.validate(df)

    def test_non_numeric_consumption_raises(self) -> None:
        """Non-numeric consumption that cannot be coerced raises SchemaError."""
        df = pd.DataFrame({"date": ["2024-01-01"], "time": [0], "consumption": ["not_a_number"]})
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
        # Realistic weather values within physical bounds
        data = {
            "temperature_2m": [10.0, 12.0, 14.0],
            "relative_humidity_2m": [60.0, 65.0, 70.0],
            "dew_point_2m": [5.0, 6.0, 7.0],
            "apparent_temperature": [8.0, 10.0, 12.0],
            "precipitation": [0.0, 0.5, 1.0],
            "snow_depth": [0.0, 0.0, 0.0],
            "weather_code": [0.0, 1.0, 3.0],
            "surface_pressure": [1013.0, 1012.0, 1011.0],
            "wind_speed_10m": [5.0, 8.0, 12.0],
            "wind_direction_10m": [180.0, 200.0, 220.0],
            "shortwave_radiation": [100.0, 200.0, 300.0],
        }
        df = pd.DataFrame(data, index=idx)
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

    def test_temperature_above_60_raises(self) -> None:
        """temperature_2m > 60 violates le=60 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["temperature_2m"] = [61.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_temperature_below_minus_50_raises(self) -> None:
        """temperature_2m < -50 violates ge=-50 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["temperature_2m"] = [-51.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_pressure_below_870_raises(self) -> None:
        """surface_pressure < 870 violates ge=870 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["surface_pressure"] = [869.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_pressure_above_1085_raises(self) -> None:
        """surface_pressure > 1085 violates le=1085 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["surface_pressure"] = [1086.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_humidity_above_100_raises(self) -> None:
        """relative_humidity_2m > 100 violates le=100 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["relative_humidity_2m"] = [101.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_negative_precipitation_raises(self) -> None:
        """precipitation < 0 violates ge=0 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["precipitation"] = [-0.1]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_wind_direction_above_360_raises(self) -> None:
        """wind_direction_10m > 360 violates le=360 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["wind_direction_10m"] = [361.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)

    def test_weather_code_above_99_raises(self) -> None:
        """weather_code > 99 violates le=99 constraint."""
        idx = pd.date_range("2024-01-01", periods=1, freq="h", name="datetime")
        df = _make_valid_weather_df(idx)
        df["weather_code"] = [100.0]
        with pytest.raises((SchemaError, SchemaErrors)):
            WeatherSchema.validate(df)


# ---------------------------------------------------------------------------
# ConsumptionSchema additional negative tests
# ---------------------------------------------------------------------------


class TestConsumptionSchemaDtype:
    """Tests for ConsumptionSchema dtype enforcement."""

    def test_string_consumption_raises(self) -> None:
        """String consumption column that cannot be coerced raises SchemaError."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame({"consumption": ["not_a_number", "abc", "xyz"]}, index=idx)
        with pytest.raises((SchemaError, SchemaErrors)):
            ConsumptionSchema.validate(df)


# ---------------------------------------------------------------------------
# EpiasSchema additional negative tests
# ---------------------------------------------------------------------------


class TestEpiasSchemaDtype:
    """Additional negative tests for EpiasSchema."""

    def test_string_column_raises(self) -> None:
        """Non-numeric Real_Time_Consumption raises SchemaError."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", name="datetime")
        df = pd.DataFrame(
            {
                "Real_Time_Consumption": ["abc", "def", "ghi"],
                "DAM_Purchase": [100.0, 200.0, 300.0],
                "Bilateral_Agreement_Purchase": [100.0, 200.0, 300.0],
                "Load_Forecast": [100.0, 200.0, 300.0],
            },
            index=idx,
        )
        with pytest.raises((SchemaError, SchemaErrors)):
            EpiasSchema.validate(df)


# ---------------------------------------------------------------------------
# Helper to build valid weather DataFrame
# ---------------------------------------------------------------------------


def _make_valid_weather_df(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a valid weather DataFrame for mutation testing."""
    n = len(idx)
    return pd.DataFrame(
        {
            "temperature_2m": [15.0] * n,
            "relative_humidity_2m": [60.0] * n,
            "dew_point_2m": [5.0] * n,
            "apparent_temperature": [13.0] * n,
            "precipitation": [0.0] * n,
            "snow_depth": [0.0] * n,
            "weather_code": [1.0] * n,
            "surface_pressure": [1013.0] * n,
            "wind_speed_10m": [10.0] * n,
            "wind_direction_10m": [180.0] * n,
            "shortwave_radiation": [200.0] * n,
        },
        index=idx,
    )
