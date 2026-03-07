"""Add weather_snapshots table.

Revision ID: 003
Revises: 2051b306c0c5
Create Date: 2026-03-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: str = "2051b306c0c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "weather_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(12),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "forecast_dt", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "fetched_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "is_actual", sa.Boolean, nullable=False, server_default="false"
        ),
        # Weather variables (weighted averages)
        sa.Column("temperature_2m", sa.Float, nullable=True),
        sa.Column("apparent_temperature", sa.Float, nullable=True),
        sa.Column("relative_humidity_2m", sa.Float, nullable=True),
        sa.Column("dew_point_2m", sa.Float, nullable=True),
        sa.Column("precipitation", sa.Float, nullable=True),
        sa.Column("snow_depth", sa.Float, nullable=True),
        sa.Column("surface_pressure", sa.Float, nullable=True),
        sa.Column("wind_speed_10m", sa.Float, nullable=True),
        sa.Column("wind_direction_10m", sa.Float, nullable=True),
        sa.Column("shortwave_radiation", sa.Float, nullable=True),
        sa.Column("weather_code", sa.SmallInteger, nullable=True),
        # Derived values
        sa.Column("wth_hdd", sa.Float, nullable=True),
        sa.Column("wth_cdd", sa.Float, nullable=True),
        # Constraints
        sa.UniqueConstraint(
            "forecast_dt", "job_id", "is_actual",
            name="uq_weather_snap_job_dt",
        ),
    )
    op.create_index("idx_weather_snap_job", "weather_snapshots", ["job_id"])
    op.create_index("idx_weather_snap_dt", "weather_snapshots", ["forecast_dt"])
    # Partial unique index: one actual record per datetime
    op.create_index(
        "idx_weather_snap_actual_unique",
        "weather_snapshots",
        ["forecast_dt"],
        unique=True,
        postgresql_where=sa.text("is_actual = true"),
    )


def downgrade() -> None:
    op.drop_table("weather_snapshots")
