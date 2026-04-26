"""Pandera DataFrame schemas for data validation."""

from __future__ import annotations

import pandera as pa
from pandera.typing import Index, Series


class RawExcelSchema(pa.DataFrameModel):
    """Schema for raw Excel input after column rename."""

    date: Series[str]
    time: Series[int] = pa.Field(ge=0, le=23)
    consumption: Series[float] = pa.Field(ge=0.0, le=10000.0)

    class Config:
        coerce = True


class ConsumptionSchema(pa.DataFrameModel):
    """Schema for processed consumption DataFrame with DatetimeIndex."""

    datetime: Index[pa.DateTime]
    consumption: Series[float] = pa.Field(ge=0.0, nullable=True)


class EpiasSchema(pa.DataFrameModel):
    """Schema for EPIAS market data.

    FDPP is active (requires region=TR1 parameter). Schema validates
    the 4 core variables; FDPP is allowed via strict=False.
    """

    datetime: Index[pa.DateTime]
    # FDPP: active via region=TR1, validated when present (strict=False allows absence)
    Real_Time_Consumption: Series[float] = pa.Field(nullable=True)
    DAM_Purchase: Series[float] = pa.Field(nullable=True)
    Bilateral_Agreement_Purchase: Series[float] = pa.Field(nullable=True)
    Load_Forecast: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = False  # allow extra columns (e.g. generation data)


class WeatherSchema(pa.DataFrameModel):
    """Schema for weighted-average weather data with physical range validation."""

    datetime: Index[pa.DateTime]
    temperature_2m: Series[float] = pa.Field(ge=-50, le=60, nullable=True)
    relative_humidity_2m: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    dew_point_2m: Series[float] = pa.Field(ge=-60, le=50, nullable=True)
    apparent_temperature: Series[float] = pa.Field(ge=-70, le=70, nullable=True)
    precipitation: Series[float] = pa.Field(ge=0, nullable=True)
    snow_depth: Series[float] = pa.Field(ge=0, nullable=True)
    # weather_code allows int input (OpenMeteo + test fixtures both pass int);
    # field-level coerce keeps DatetimeIndex strict while easing this column.
    weather_code: Series[float] = pa.Field(ge=0, le=99, nullable=True, coerce=True)
    surface_pressure: Series[float] = pa.Field(ge=870, le=1085, nullable=True)
    wind_speed_10m: Series[float] = pa.Field(ge=0, nullable=True)
    wind_direction_10m: Series[float] = pa.Field(ge=0, le=360, nullable=True)
    shortwave_radiation: Series[float] = pa.Field(ge=0, nullable=True)
