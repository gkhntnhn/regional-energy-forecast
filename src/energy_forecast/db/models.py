"""ORM models for jobs and predictions."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
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


class ModelRunModel(Base):
    """Training run record — lightweight log of model training results."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running"
    )

    # Metrics
    val_mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Config snapshot
    hyperparameters: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        _JSONBCompat(), nullable=True
    )
    n_trials: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_splits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feature_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Model artifact
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_promoted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_model_runs_status",
        ),
        Index("idx_model_runs_type", "model_type"),
        Index("idx_model_runs_status", "status"),
    )


# ---------------------------------------------------------------------------
# External data tables (M11 Phase 1)
# ---------------------------------------------------------------------------


class EpiasMarketModel(Base):
    """EPIAS market data — 5 variables (FDPP, RTC, DAM, Bilateral, LoadForecast)."""

    __tablename__ = "epias_market"

    dt: Mapped[datetime] = mapped_column(
        "datetime", DateTime(timezone=True), primary_key=True
    )
    fdpp: Mapped[float | None] = mapped_column(Float, nullable=True)
    rtc: Mapped[float | None] = mapped_column(Float, nullable=True)
    dam_purchase: Mapped[float | None] = mapped_column(Float, nullable=True)
    bilateral: Mapped[float | None] = mapped_column(Float, nullable=True)
    load_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_epias_market_fetched_at", "fetched_at"),)


class EpiasGenerationModel(Base):
    """EPIAS real-time generation data — 17 fuel types."""

    __tablename__ = "epias_generation"

    dt: Mapped[datetime] = mapped_column(
        "datetime", DateTime(timezone=True), primary_key=True
    )
    gen_asphaltite_coal: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_biomass: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_black_coal: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_dammed_hydro: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_fueloil: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_geothermal: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_import_coal: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_import_export: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_lignite: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_naphta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_natural_gas: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_river: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_sun: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_wasteheat: Mapped[float | None] = mapped_column(Float, nullable=True)
    gen_wind: Mapped[float | None] = mapped_column(Float, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_epias_generation_fetched_at", "fetched_at"),)


class WeatherCacheModel(Base):
    """Weather observations — per city, per source (historical/forecast)."""

    __tablename__ = "weather_cache"

    dt: Mapped[datetime] = mapped_column(
        "datetime", DateTime(timezone=True), primary_key=True
    )
    city: Mapped[str] = mapped_column(String(50), primary_key=True)
    source: Mapped[str] = mapped_column(String(20), primary_key=True)
    temperature_2m: Mapped[float | None] = mapped_column(Float, nullable=True)
    apparent_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_humidity_2m: Mapped[float | None] = mapped_column(Float, nullable=True)
    dew_point_2m: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation: Mapped[float | None] = mapped_column(Float, nullable=True)
    snow_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    surface_pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    shortwave_radiation: Mapped[float | None] = mapped_column(Float, nullable=True)
    weather_code: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_weather_cache_fetched_at", "fetched_at"),
        Index("ix_weather_cache_city", "city"),
        Index("ix_weather_cache_source", "source"),
    )


class TurkishHolidayModel(Base):
    """Turkish holiday calendar — raw parquet data (tatil_tipi derived by CalendarFE)."""

    __tablename__ = "turkish_holidays"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    # holiday_name is None for Ramadan days (tracked via is_ramadan)
    holiday_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # raw_holiday_name: original Turkish name from parquet (1:1 mapping)
    raw_holiday_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # is_ramadan: 0 = regular day, 1 = Ramadan period
    is_ramadan: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="0"
    )
    # bayram_gun_no: 0 = not a holiday, 1-4 = holiday day number
    bayram_gun_no: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="0"
    )
    # bayrama_kalan_gun: days until next bayram, -1 = not applicable
    bayrama_kalan_gun: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="-1"
    )


class ProfileCoefficientModel(Base):
    """EPİAŞ profile coefficients — 14 columns (10 base + 4 aggregate)."""

    __tablename__ = "profile_coefficients"

    dt: Mapped[datetime] = mapped_column(
        "datetime", DateTime(timezone=True), primary_key=True
    )
    # Base profiles (10) — voltage-level specific
    profile_residential_lv: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_residential_mv: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_industrial_lv: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_industrial_mv: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_commercial_lv: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_commercial_mv: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_agricultural_irrigation_lv: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    profile_agricultural_irrigation_mv: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    profile_lighting: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_government: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Aggregate profiles (4) — voltage-level totals
    profile_residential: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_industrial: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_commercial: Mapped[float | None] = mapped_column(Float, nullable=True)
    profile_agricultural_irrigation: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_profile_coefficients_fetched_at", "fetched_at"),)
