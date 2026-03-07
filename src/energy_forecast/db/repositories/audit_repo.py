"""Audit log repository — write operations for audit_logs table."""

from __future__ import annotations

from typing import Any

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
