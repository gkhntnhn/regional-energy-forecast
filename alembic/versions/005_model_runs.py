"""Create model_runs table for training run tracking.

Revision ID: 005
Revises: 004
Create Date: 2026-03-07
"""

import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_runs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("model_type", sa.String(20), nullable=False),
        sa.Column("run_id", sa.String(50), nullable=True),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="running"
        ),
        # Metrics
        sa.Column("val_mape", sa.Float, nullable=True),
        sa.Column("test_mape", sa.Float, nullable=True),
        sa.Column("val_rmse", sa.Float, nullable=True),
        sa.Column("test_rmse", sa.Float, nullable=True),
        # Config snapshot
        sa.Column("hyperparameters", sa.dialects.postgresql.JSONB, nullable=True),
        sa.Column("n_trials", sa.Integer, nullable=True),
        sa.Column("n_splits", sa.Integer, nullable=True),
        sa.Column("feature_count", sa.Integer, nullable=True),
        # Model artifact
        sa.Column("model_path", sa.String(500), nullable=True),
        sa.Column("is_promoted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        # Timestamps
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        # Constraints
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_model_runs_status",
        ),
    )

    op.create_index("idx_model_runs_type", "model_runs", ["model_type"])
    op.create_index("idx_model_runs_status", "model_runs", ["status"])


def downgrade() -> None:
    op.drop_index("idx_model_runs_status", table_name="model_runs")
    op.drop_index("idx_model_runs_type", table_name="model_runs")
    op.drop_table("model_runs")
