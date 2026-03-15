"""Fix model_versions column: JSON → JSONB.

Migration 007 converted config_snapshot, metadata, and epias_snapshot to JSONB
but missed model_versions. This migration catches it up.

Revision ID: 009
Revises: 008
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None

_COLS = ["model_versions"]


def upgrade() -> None:
    """Convert model_versions JSON → JSONB on PostgreSQL."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for col in _COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE jobs ALTER COLUMN "{col}" '
                f"TYPE jsonb USING \"{col}\"::jsonb"
            )
        )


def downgrade() -> None:
    """Revert model_versions JSONB → JSON on PostgreSQL."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for col in _COLS:
        op.execute(
            sa.text(
                f'ALTER TABLE jobs ALTER COLUMN "{col}" '
                f"TYPE json USING \"{col}\"::json"
            )
        )
