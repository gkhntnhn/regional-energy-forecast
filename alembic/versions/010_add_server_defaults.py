"""Add server_default to jobs columns missing them.

Ensures SQL-level inserts (backup restore, direct SQL) get correct defaults
without relying on Python ORM layer.

Revision ID: 010
Revises: 009
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None

# (table, column, default_value)
_DEFAULTS = [
    ("jobs", "status", "pending"),
    ("jobs", "email_status", "pending"),
    ("jobs", "email_attempts", "0"),
]


def upgrade() -> None:
    """Add server_default to columns that only had Python-side defaults."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table, col, default in _DEFAULTS:
            op.execute(
                sa.text(
                    f"ALTER TABLE {table} ALTER COLUMN {col} SET DEFAULT '{default}'"
                )
            )
    # SQLite: defaults are handled by ORM model metadata (CREATE TABLE)


def downgrade() -> None:
    """Remove server_default from columns."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table, col, _default in _DEFAULTS:
            op.execute(
                sa.text(f"ALTER TABLE {table} ALTER COLUMN {col} DROP DEFAULT")
            )
