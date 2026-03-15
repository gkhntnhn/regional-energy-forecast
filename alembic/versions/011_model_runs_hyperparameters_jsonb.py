"""Fix model_runs.hyperparameters column: JSON -> JSONB.

Migration 007 converted jobs and audit_logs columns to JSONB.
Migration 009 converted jobs.model_versions to JSONB.
This migration catches model_runs.hyperparameters.

Revision ID: 011
Revises: 010
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None

_TABLE = "model_runs"
_COLS = ["hyperparameters"]


def upgrade() -> None:
    """Convert model_runs.hyperparameters JSON -> JSONB on PostgreSQL."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for col in _COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE {_TABLE} ALTER COLUMN "{col}" '
                f'TYPE jsonb USING "{col}"::jsonb'
            )
        )


def downgrade() -> None:
    """Revert model_runs.hyperparameters JSONB -> JSON on PostgreSQL."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for col in _COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE {_TABLE} ALTER COLUMN "{col}" '
                f'TYPE json USING "{col}"::json'
            )
        )
