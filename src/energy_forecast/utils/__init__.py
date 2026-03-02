"""Shared utilities."""

from zoneinfo import ZoneInfo

from energy_forecast.utils.logging import setup_logger, suppress_training_noise

TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")

__all__ = ["TZ_ISTANBUL", "setup_logger", "suppress_training_noise"]
