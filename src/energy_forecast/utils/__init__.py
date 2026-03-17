"""Shared utilities."""

from zoneinfo import ZoneInfo

from energy_forecast.utils.logging import setup_logger, suppress_training_noise

TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")

# Weather column prefixes for selective ffill/bfill (never consumption/EPIAS)
WEATHER_FILL_PREFIXES = (
    "temperature",
    "relative_humidity",
    "dew_point",
    "apparent_temperature",
    "precipitation",
    "snow_depth",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
    "shortwave_radiation",
)

__all__ = ["TZ_ISTANBUL", "WEATHER_FILL_PREFIXES", "setup_logger", "suppress_training_noise"]
