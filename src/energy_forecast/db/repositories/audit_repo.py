"""Audit log repository — write and query operations for audit_logs table."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import AuditLogModel


class AuditRepository:
    """Data access layer for audit_logs table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log(
        self,
        action: str,
        user_email: str | None = None,
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLogModel:
        """Create an audit log entry."""
        entry = AuditLogModel(
            action=action,
            user_email=user_email,
            ip_address=ip_address,
            details=details,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def get_last_action(self, action: str) -> AuditLogModel | None:
        """Get the most recent audit log entry for a given action.

        Used for drift alert cooldown — checks when the last alert was sent.
        """
        result = await self._session.execute(
            select(AuditLogModel)
            .where(AuditLogModel.action == action)
            .order_by(AuditLogModel.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
