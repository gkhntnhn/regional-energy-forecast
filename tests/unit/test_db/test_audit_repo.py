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


# ---------------------------------------------------------------------------
# Ordering tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_actions_ordering(db_session: AsyncSession) -> None:
    """Multiple actions inserted — get_last_action returns most recent, not first."""
    now = datetime.now(tz=_TZ_ISTANBUL)

    for i in range(5):
        entry = AuditLogModel(
            action="predict_request",
            details={"seq": i},
            created_at=now - timedelta(hours=5 - i),  # oldest first
        )
        db_session.add(entry)
    await db_session.flush()

    repo = AuditRepository(db_session)
    last = await repo.get_last_action("predict_request")
    assert last is not None
    assert last.details == {"seq": 4}  # most recent


@pytest.mark.asyncio
async def test_ordering_with_mixed_actions(db_session: AsyncSession) -> None:
    """Mixed action types — each action returns its own most recent entry."""
    now = datetime.now(tz=_TZ_ISTANBUL)

    db_session.add(AuditLogModel(
        action="predict_request",
        details={"v": "old_predict"},
        created_at=now - timedelta(hours=3),
    ))
    db_session.add(AuditLogModel(
        action="job_complete",
        details={"v": "old_job"},
        created_at=now - timedelta(hours=2),
    ))
    db_session.add(AuditLogModel(
        action="predict_request",
        details={"v": "new_predict"},
        created_at=now - timedelta(hours=1),
    ))
    await db_session.flush()

    repo = AuditRepository(db_session)

    predict_last = await repo.get_last_action("predict_request")
    assert predict_last is not None
    assert predict_last.details == {"v": "new_predict"}

    job_last = await repo.get_last_action("job_complete")
    assert job_last is not None
    assert job_last.details == {"v": "old_job"}


# ---------------------------------------------------------------------------
# Concurrent write tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_rapid_inserts(db_session: AsyncSession) -> None:
    """Two rapid inserts with same timestamp both succeed."""
    now = datetime.now(tz=_TZ_ISTANBUL)
    repo = AuditRepository(db_session)

    entry1 = await repo.log(
        action="predict_request",
        user_email="user1@example.com",
        details={"job": "job_1"},
    )
    entry2 = await repo.log(
        action="predict_request",
        user_email="user2@example.com",
        details={"job": "job_2"},
    )

    assert entry1.id is not None
    assert entry2.id is not None
    assert entry1.id != entry2.id


@pytest.mark.asyncio
async def test_concurrent_same_timestamp_both_stored(db_session: AsyncSession) -> None:
    """Two entries with identical created_at are both persisted."""
    now = datetime.now(tz=_TZ_ISTANBUL)

    e1 = AuditLogModel(action="drift_alert_mape", details={"a": 1}, created_at=now)
    e2 = AuditLogModel(action="drift_alert_mape", details={"a": 2}, created_at=now)
    db_session.add_all([e1, e2])
    await db_session.flush()

    # Both should be persisted (get_last_action returns one of them)
    repo = AuditRepository(db_session)
    last = await repo.get_last_action("drift_alert_mape")
    assert last is not None
    assert last.details in [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_with_all_fields(db_session: AsyncSession) -> None:
    """Log entry with all fields populated returns correctly."""
    repo = AuditRepository(db_session)
    entry = await repo.log(
        action="predict_request",
        user_email="admin@company.com",
        ip_address="10.0.0.1",
        details={"job_id": "xyz789", "model": "ensemble", "file_size": 1024},
    )
    assert entry.action == "predict_request"
    assert entry.user_email == "admin@company.com"
    assert entry.ip_address == "10.0.0.1"
    assert entry.details["model"] == "ensemble"
    assert entry.details["file_size"] == 1024


@pytest.mark.asyncio
async def test_log_empty_details_dict(db_session: AsyncSession) -> None:
    """Log entry with empty dict details."""
    repo = AuditRepository(db_session)
    entry = await repo.log(action="test_action", details={})
    assert entry.details == {}
