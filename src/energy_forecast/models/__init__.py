"""Forecasting models."""

from energy_forecast.models.base import BaseForecaster
from energy_forecast.models.catboost import CatBoostForecaster

__all__ = ["BaseForecaster", "CatBoostForecaster"]
