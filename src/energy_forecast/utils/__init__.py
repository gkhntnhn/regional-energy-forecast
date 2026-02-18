"""Shared utilities."""

from zoneinfo import ZoneInfo

from energy_forecast.utils.logging import setup_logger

TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")

__all__ = ["TZ_ISTANBUL", "setup_logger"]
