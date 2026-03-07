"""Initial schema — jobs + predictions tables.

Revision ID: 001
Revises: None
Create Date: 2026-03-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="pending"
        ),
        sa.Column("progress", sa.Text, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("excel_path", sa.Text, nullable=False),
        sa.Column("file_stem", sa.String(100), nullable=False),
        sa.Column("result_path", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        # Data Lineage L3
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("config_snapshot", sa.JSON, nullable=True),
        sa.Column("model_versions", sa.JSON, nullable=True),
        sa.Column("epias_snapshot", sa.JSON, nullable=True),
        sa.Column("excel_hash", sa.String(64), nullable=True),
        # Artifact paths
        sa.Column("historical_path", sa.Text, nullable=True),
        sa.Column("forecast_path", sa.Text, nullable=True),
        sa.Column("archive_path", sa.Text, nullable=True),
        # Email tracking
        sa.Column(
            "email_status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("email_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("email_error", sa.Text, nullable=True),
        sa.Column(
            "email_attempts", sa.Integer, nullable=False, server_default="0"
        ),
        # Constraints
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'archived')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(
            "email_status IN ('pending', 'sent', 'failed', 'skipped')",
            name="ck_jobs_email_status",
        ),
    )
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.create_index("idx_jobs_created_at", "jobs", ["created_at"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(12),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "forecast_dt", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("consumption_mwh", sa.Float, nullable=False),
        sa.Column("period", sa.String(20), nullable=False),
        sa.Column("model_source", sa.String(20), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # Phase 2 columns
        sa.Column("actual_mwh", sa.Float, nullable=True),
        sa.Column("error_pct", sa.Float, nullable=True),
        sa.Column("matched_at", sa.DateTime(timezone=True), nullable=True),
        # Constraints
        sa.CheckConstraint(
            "period IN ('intraday', 'day_ahead')",
            name="ck_predictions_period",
        ),
    )
    op.create_index("idx_predictions_job_id", "predictions", ["job_id"])
    op.create_index(
        "idx_predictions_forecast_dt", "predictions", ["forecast_dt"]
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("jobs")
