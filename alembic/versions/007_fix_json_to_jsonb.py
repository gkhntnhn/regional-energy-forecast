"""Fix JSON columns to JSONB and add composite indexes.

Fixes:
  - B1: jobs table JSON columns → JSONB (better indexing, operators)
  - audit_logs.details JSON → JSONB
  - Composite indexes for common query patterns

Revision ID: 007
Revises: 006
Create Date: 2026-03-11
"""

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None

# JSON columns that should be JSONB
_JOBS_JSON_COLS = ["config_snapshot", "feature_importance", "metadata", "epias_snapshot"]
_AUDIT_JSON_COLS = ["details"]


def upgrade() -> None:
    """Convert JSON → JSONB and create composite indexes."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # --- B1: jobs table JSON → JSONB ---
    for col in _JOBS_JSON_COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE jobs ALTER COLUMN "{col}" '
                f"TYPE jsonb USING \"{col}\"::jsonb"
            )
        )

    # --- audit_logs.details JSON → JSONB ---
    for col in _AUDIT_JSON_COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE audit_logs ALTER COLUMN "{col}" '
                f"TYPE jsonb USING \"{col}\"::jsonb"
            )
        )

    # --- Composite indexes ---
    op.create_index(
        "ix_predictions_job_model",
        "predictions",
        ["job_id", "model_source"],
    )
    op.create_index(
        "ix_model_runs_type_status",
        "model_runs",
        ["model_type", "status"],
    )


def downgrade() -> None:
    """Revert JSONB → JSON and drop composite indexes."""
    bind = op.get_bind()

    # Drop indexes (dialect-agnostic)
    op.drop_index("ix_model_runs_type_status", table_name="model_runs")
    op.drop_index("ix_predictions_job_model", table_name="predictions")

    if bind.dialect.name != "postgresql":
        return

    # Revert JSONB → JSON
    for col in _JOBS_JSON_COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE jobs ALTER COLUMN "{col}" '
                f"TYPE json USING \"{col}\"::json"
            )
        )
    for col in _AUDIT_JSON_COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE audit_logs ALTER COLUMN "{col}" '
                f"TYPE json USING \"{col}\"::json"
            )
        )
