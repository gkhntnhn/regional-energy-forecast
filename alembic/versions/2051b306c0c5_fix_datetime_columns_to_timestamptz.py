"""fix datetime columns to timestamptz

Revision ID: 2051b306c0c5
Revises: 001
Create Date: 2026-03-07 03:55:27.349181
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from sqlalchemy import DateTime


# revision identifiers, used by Alembic.
revision: str = "2051b306c0c5"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TZ = DateTime(timezone=True)
_NO_TZ = DateTime(timezone=False)

# All datetime columns that need TIMESTAMPTZ
_JOBS_COLS = ["created_at", "completed_at", "email_sent_at"]
_PRED_COLS = ["forecast_dt", "created_at", "matched_at"]


def upgrade() -> None:
    for col in _JOBS_COLS:
        op.alter_column(
            "jobs", col, type_=_TZ, postgresql_using=f"{col} AT TIME ZONE 'UTC'"
        )
    for col in _PRED_COLS:
        op.alter_column(
            "predictions", col, type_=_TZ, postgresql_using=f"{col} AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    for col in _JOBS_COLS:
        op.alter_column("jobs", col, type_=_NO_TZ)
    for col in _PRED_COLS:
        op.alter_column("predictions", col, type_=_NO_TZ)
