"""Environment, database, monitoring and API serving configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ApiConfig",
    "ApiEmailConfig",
    "ApiFilesConfig",
    "DatabaseConfig",
    "DriftDetectionConfig",
    "EnvConfig",
    "MonitoringConfig",
]


# ---------------------------------------------------------------------------
# Environment config (secrets from .env)
# ---------------------------------------------------------------------------


class EnvConfig(BaseSettings):
    """Environment variables loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    epias_username: str = ""
    epias_password: str = ""
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    sender_email: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
    aws_s3_bucket: str = ""
    aws_region: str = "eu-west-1"
    database_url: str = ""
    database_url_sync: str = ""

    # Google Drive backup
    gdrive_enabled: bool = False
    gdrive_credentials_path: str = ""
    gdrive_root_folder_id: str = ""


# ---------------------------------------------------------------------------
# Database Config
# ---------------------------------------------------------------------------


class DriftDetectionConfig(BaseModel, frozen=True):
    """Model drift detection thresholds."""

    enabled: bool = True
    mape_threshold_warning: float = Field(default=5.0, ge=0)
    mape_threshold_critical: float = Field(default=8.0, ge=0)
    mape_trend_threshold: float = Field(default=0.5, ge=0)
    bias_threshold: float = Field(default=3.0, ge=0)
    lookback_days: int = Field(default=7, ge=1)
    trend_weeks: int = Field(default=4, ge=2)
    min_samples: int = Field(default=24, ge=1)
    cooldown_hours: int = Field(default=24, ge=1)
    email_on_warning: bool = False
    admin_email: str = ""


class MonitoringConfig(BaseModel, frozen=True):
    """Monitoring configuration."""

    drift_detection: DriftDetectionConfig = Field(
        default_factory=DriftDetectionConfig
    )


class DatabaseConfig(BaseModel, frozen=True):
    """Database connection pool configuration."""

    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_timeout: int = Field(default=30, ge=5)
    pool_recycle: int = Field(default=300, ge=60)
    pool_pre_ping: bool = True
    echo: bool = False


# ---------------------------------------------------------------------------
# API Config
# ---------------------------------------------------------------------------


class ApiFilesConfig(BaseModel, frozen=True):
    """API file handling configuration."""

    upload_dir: str = "data/uploads"
    output_dir: str = "data/outputs"
    allowed_extensions: list[str] = Field(default_factory=lambda: [".xlsx", ".xls"])
    max_file_size_mb: int = Field(default=50, ge=1)
    cleanup_after_hours: int = Field(default=24, ge=1)


class ApiEmailConfig(BaseModel, frozen=True):
    """API email template configuration."""

    sender_name: str = "Energy Forecast"
    subject_template: str = "Tahmin Sonuçları - {job_id}"
    body_template: str = Field(
        default="""Merhaba,

Talep ettiğiniz 48 saatlik elektrik tüketimi tahmini ekte sunulmuştur.

İş No: {job_id}
Oluşturulma: {created_at}

İyi çalışmalar,
Energy Forecast Sistemi"""
    )


class ApiConfig(BaseModel, frozen=True):
    """API serving configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    rate_limit: str = "10/minute"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    files: ApiFilesConfig = Field(default_factory=ApiFilesConfig)
    email: ApiEmailConfig = Field(default_factory=ApiEmailConfig)
