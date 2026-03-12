"""Add 'queued' to jobs status CHECK constraint.

Supports queue-based sequential processing (reject → queue migration).

Revision ID: 008
Revises: 007
Create Date: 2026-03-13
"""

from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add 'queued' to the status CHECK constraint."""
    # SQLite doesn't support ALTER CONSTRAINT, but for PostgreSQL:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.drop_constraint("ck_jobs_status", "jobs", type_="check")
        op.create_check_constraint(
            "ck_jobs_status",
            "jobs",
            "status IN ('pending', 'queued', 'running', 'completed', 'failed', 'archived')",
        )
    # SQLite: CHECK constraints are baked into CREATE TABLE,
    # handled at model level via metadata reflection / test fixtures


def downgrade() -> None:
    """Remove 'queued' from the status CHECK constraint."""
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.drop_constraint("ck_jobs_status", "jobs", type_="check")
        op.create_check_constraint(
            "ck_jobs_status",
            "jobs",
            "status IN ('pending', 'running', 'completed', 'failed', 'archived')",
        )
