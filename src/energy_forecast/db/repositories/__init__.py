"""Repository layer — CRUD operations for ORM models."""

from energy_forecast.db.repositories.analytics_repo import AnalyticsRepository
from energy_forecast.db.repositories.audit_repo import AuditRepository
from energy_forecast.db.repositories.epias_repo import EpiasRepository
from energy_forecast.db.repositories.holiday_repo import HolidayRepository
from energy_forecast.db.repositories.job_repo import JobRepository
from energy_forecast.db.repositories.prediction_repo import PredictionRepository
from energy_forecast.db.repositories.profile_repo import ProfileRepository
from energy_forecast.db.repositories.weather_cache_repo import WeatherCacheRepository
from energy_forecast.db.repositories.weather_repo import WeatherSnapshotRepository
from energy_forecast.db.sync_repos import SyncDataAccess

__all__ = [
    "AnalyticsRepository",
    "AuditRepository",
    "EpiasRepository",
    "HolidayRepository",
    "JobRepository",
    "PredictionRepository",
    "ProfileRepository",
    "SyncDataAccess",
    "WeatherCacheRepository",
    "WeatherSnapshotRepository",
]
