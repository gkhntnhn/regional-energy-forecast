"""Tests for AuditRepository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import AuditLogModel
from energy_forecast.db.repositories.audit_repo import AuditRepository

_TZ_ISTANBUL = timezone(timedelta(hours=3))


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


# ---------------------------------------------------------------------------
# get_last_action tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_last_action_empty_table(db_session: AsyncSession) -> None:
    """Empty table returns None."""
    repo = AuditRepository(db_session)
    result = await repo.get_last_action("drift_alert_mape")
    assert result is None


@pytest.mark.asyncio
async def test_get_last_action_single_entry(db_session: AsyncSession) -> None:
    """Single matching entry is returned."""
    repo = AuditRepository(db_session)
    await repo.log(action="drift_alert_mape", details={"mape": 8.5})
    result = await repo.get_last_action("drift_alert_mape")
    assert result is not None
    assert result.action == "drift_alert_mape"
    assert result.details == {"mape": 8.5}


@pytest.mark.asyncio
async def test_get_last_action_returns_most_recent(db_session: AsyncSession) -> None:
    """Multiple entries — returns the one with latest created_at (DESC order)."""
    now = datetime.now(tz=_TZ_ISTANBUL)

    # Insert older entry
    old_entry = AuditLogModel(
        action="drift_alert_mape",
        details={"mape": 10.0},
        created_at=now - timedelta(hours=5),
    )
    db_session.add(old_entry)

    # Insert newer entry
    new_entry = AuditLogModel(
        action="drift_alert_mape",
        details={"mape": 6.0},
        created_at=now - timedelta(hours=1),
    )
    db_session.add(new_entry)
    await db_session.flush()

    repo = AuditRepository(db_session)
    result = await repo.get_last_action("drift_alert_mape")
    assert result is not None
    assert result.details == {"mape": 6.0}


@pytest.mark.asyncio
async def test_get_last_action_filters_by_action(db_session: AsyncSession) -> None:
    """Different action types do not mix — each returns its own latest."""
    now = datetime.now(tz=_TZ_ISTANBUL)

    mape_entry = AuditLogModel(
        action="drift_alert_mape",
        details={"mape": 7.0},
        created_at=now - timedelta(hours=2),
    )
    bias_entry = AuditLogModel(
        action="drift_alert_bias",
        details={"bias": -15.0},
        created_at=now - timedelta(hours=1),
    )
    db_session.add_all([mape_entry, bias_entry])
    await db_session.flush()

    repo = AuditRepository(db_session)

    mape_result = await repo.get_last_action("drift_alert_mape")
    assert mape_result is not None
    assert mape_result.action == "drift_alert_mape"
    assert mape_result.details == {"mape": 7.0}

    bias_result = await repo.get_last_action("drift_alert_bias")
    assert bias_result is not None
    assert bias_result.action == "drift_alert_bias"
    assert bias_result.details == {"bias": -15.0}


@pytest.mark.asyncio
async def test_get_last_action_nonexistent_action(db_session: AsyncSession) -> None:
    """Querying for a non-existent action returns None even if other actions exist."""
    repo = AuditRepository(db_session)
    await repo.log(action="predict_request")
    result = await repo.get_last_action("drift_alert_mape")
    assert result is None


# ---------------------------------------------------------------------------
# Cooldown scenario tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_active_recent_alert(db_session: AsyncSession) -> None:
    """Last alert 1 hour ago — cooldown is active (< 24h threshold)."""
    now = datetime.now(tz=_TZ_ISTANBUL)
    cooldown_hours = 24

    entry = AuditLogModel(
        action="drift_alert_mape",
        details={"mape": 9.0},
        created_at=now - timedelta(hours=1),
    )
    db_session.add(entry)
    await db_session.flush()

    repo = AuditRepository(db_session)
    last = await repo.get_last_action("drift_alert_mape")
    assert last is not None

    elapsed = now - last.created_at
    assert elapsed < timedelta(hours=cooldown_hours)


@pytest.mark.asyncio
async def test_cooldown_expired_old_alert(db_session: AsyncSession) -> None:
    """Last alert 25 hours ago — cooldown has expired (> 24h threshold)."""
    now = datetime.now(tz=_TZ_ISTANBUL)
    cooldown_hours = 24

    entry = AuditLogModel(
        action="drift_alert_mape",
        details={"mape": 9.0},
        created_at=now - timedelta(hours=25),
    )
    db_session.add(entry)
    await db_session.flush()

    repo = AuditRepository(db_session)
    last = await repo.get_last_action("drift_alert_mape")
    assert last is not None

    elapsed = now - last.created_at
    assert elapsed > timedelta(hours=cooldown_hours)
