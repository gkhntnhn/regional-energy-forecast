"""Tests for AuditRepository."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.repositories.audit_repo import AuditRepository


@pytest.mark.asyncio
async def test_audit_log_creation(db_session: AsyncSession) -> None:
    """Audit log entry is created with correct fields."""
    repo = AuditRepository(db_session)
    entry = await repo.log(
        action="predict_request",
        user_email="user@example.com",
        ip_address="192.168.1.1",
        details={"job_id": "abc123", "file_name": "test.xlsx"},
    )
    assert entry.id is not None
    assert entry.action == "predict_request"
    assert entry.user_email == "user@example.com"
    assert entry.ip_address == "192.168.1.1"
    assert entry.details == {"job_id": "abc123", "file_name": "test.xlsx"}


@pytest.mark.asyncio
async def test_audit_log_optional_fields(db_session: AsyncSession) -> None:
    """Audit log works with minimal fields."""
    repo = AuditRepository(db_session)
    entry = await repo.log(action="job_complete")
    assert entry.action == "job_complete"
    assert entry.user_email is None
    assert entry.ip_address is None
    assert entry.details is None
