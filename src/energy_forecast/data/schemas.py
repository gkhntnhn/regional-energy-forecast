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
    """Schema for EPIAS market data."""

    datetime: Index[pa.DateTime]
    FDPP: Series[float] = pa.Field(nullable=True)
    Real_Time_Consumption: Series[float] = pa.Field(nullable=True)
    DAM_Purchase: Series[float] = pa.Field(nullable=True)
    Bilateral_Agreement_Purchase: Series[float] = pa.Field(nullable=True)
    Load_Forecast: Series[float] = pa.Field(nullable=True)


class WeatherSchema(pa.DataFrameModel):
    """Schema for weighted-average weather data."""

    datetime: Index[pa.DateTime]
    temperature_2m: Series[float] = pa.Field(nullable=True)
    relative_humidity_2m: Series[float] = pa.Field(nullable=True)
    dew_point_2m: Series[float] = pa.Field(nullable=True)
    apparent_temperature: Series[float] = pa.Field(nullable=True)
    precipitation: Series[float] = pa.Field(nullable=True)
    snow_depth: Series[float] = pa.Field(nullable=True)
    weather_code: Series[float] = pa.Field(nullable=True)
    surface_pressure: Series[float] = pa.Field(nullable=True)
    wind_speed_10m: Series[float] = pa.Field(nullable=True)
    wind_direction_10m: Series[float] = pa.Field(nullable=True)
    shortwave_radiation: Series[float] = pa.Field(nullable=True)
