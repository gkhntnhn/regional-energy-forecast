"""ORM models for jobs and predictions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from energy_forecast.db.base import Base

# ---------------------------------------------------------------------------
# JSONB with automatic JSON serialization for SQLite (tests use aiosqlite)
# ---------------------------------------------------------------------------


class _JSONBCompat(TypeDecorator[Any]):
    """JSONB on PostgreSQL, JSON-serialized Text on SQLite."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(Text())

    def process_bind_param(
        self, value: Any, dialect: Any
    ) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.loads(value)


class JobModel(Base):
    """Prediction job record."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )
    progress: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    excel_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_stem: Mapped[str] = mapped_column(String(100), nullable=False)
    result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Data Lineage L3
    metadata_: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        "metadata", _JSONBCompat(), nullable=True
    )
    config_snapshot: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        _JSONBCompat(), nullable=True
    )
    model_versions: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        _JSONBCompat(), nullable=True
    )
    epias_snapshot: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        _JSONBCompat(), nullable=True
    )
    excel_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Artifact path references (GDrive or local)
    historical_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    forecast_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    archive_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Email tracking
    email_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )
    email_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    email_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    email_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # Relationships
    predictions: Mapped[list[PredictionModel]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    weather_snapshots: Mapped[list[WeatherSnapshotModel]] = relationship(
        back_populates="job"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'archived')",
            name="ck_jobs_status",
        ),
        CheckConstraint(
            "email_status IN ('pending', 'sent', 'failed', 'skipped')",
            name="ck_jobs_email_status",
        ),
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_created_at", "created_at"),
    )


class PredictionModel(Base):
    """Individual hourly prediction record."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(12), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    forecast_dt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    consumption_mwh: Mapped[float] = mapped_column(nullable=False)
    period: Mapped[str] = mapped_column(String(20), nullable=False)
    model_source: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Phase 2 columns (NULL until populated)
    actual_mwh: Mapped[float | None] = mapped_column(nullable=True)
    error_pct: Mapped[float | None] = mapped_column(nullable=True)
    matched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    job: Mapped[JobModel] = relationship(back_populates="predictions")

    __table_args__ = (
        CheckConstraint(
            "period IN ('intraday', 'day_ahead')",
            name="ck_predictions_period",
        ),
        Index("idx_predictions_job_id", "job_id"),
        Index("idx_predictions_forecast_dt", "forecast_dt"),
    )


class WeatherSnapshotModel(Base):
    """Weather snapshot — forecast or actual values per hour."""

    __tablename__ = "weather_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(
        String(12), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    forecast_dt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_actual: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Weather variables (weighted averages across 4 cities)
    temperature_2m: Mapped[float | None] = mapped_column(nullable=True)
    apparent_temperature: Mapped[float | None] = mapped_column(nullable=True)
    relative_humidity_2m: Mapped[float | None] = mapped_column(nullable=True)
    dew_point_2m: Mapped[float | None] = mapped_column(nullable=True)
    precipitation: Mapped[float | None] = mapped_column(nullable=True)
    snow_depth: Mapped[float | None] = mapped_column(nullable=True)
    surface_pressure: Mapped[float | None] = mapped_column(nullable=True)
    wind_speed_10m: Mapped[float | None] = mapped_column(nullable=True)
    wind_direction_10m: Mapped[float | None] = mapped_column(nullable=True)
    shortwave_radiation: Mapped[float | None] = mapped_column(nullable=True)
    weather_code: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True
    )

    # Derived values
    wth_hdd: Mapped[float | None] = mapped_column(nullable=True)
    wth_cdd: Mapped[float | None] = mapped_column(nullable=True)

    # Relationships
    job: Mapped[JobModel | None] = relationship(
        back_populates="weather_snapshots"
    )

    __table_args__ = (
        UniqueConstraint(
            "forecast_dt", "job_id", "is_actual",
            name="uq_weather_snap_job_dt",
        ),
        Index("idx_weather_snap_job", "job_id"),
        Index("idx_weather_snap_dt", "forecast_dt"),
        # Partial unique index for actuals is PostgreSQL-only (in migration 003).
        # Not defined here because SQLAlchemy create_all() would create a
        # non-partial unique index on SQLite, breaking tests.
    )


class AuditLogModel(Base):
    """Audit log entry for API actions."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    user_email: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(45), nullable=True
    )
    details: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        _JSONBCompat(), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_created_at", "created_at"),
    )
