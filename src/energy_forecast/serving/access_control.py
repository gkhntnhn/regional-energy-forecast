"""Job-access guard for IDOR mitigation (audit SEC-A4).

Today the project runs in single-tenant mode (one shared API_KEY). All jobs
implicitly belong to the API-key holder, so per-user ownership cannot be
enforced without schema work. This module provides the guard primitive that
will activate the moment a multi-tenant migration adds an
``owner_fingerprint`` column to the jobs table.

Behavior:
- If the Job has no ``owner_fingerprint`` attribute (current state) the
  guard is a no-op but emits a structured audit log line.
- If the column is added later, the guard rejects requests whose caller
  fingerprint does not match.

This keeps the call sites identical between single-tenant and multi-tenant
deployments.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import HTTPException
from loguru import logger


def compute_caller_fingerprint(token: str | None) -> str:
    """Stable opaque identifier for the bearer token.

    SHA-256 truncated to 16 hex chars — collision-resistant for audit-log
    correlation without exposing the raw token in logs.
    """
    if not token:
        return "anonymous"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest[:16]


def assert_job_access(
    job: Any,
    *,
    fingerprint: str,
    action: str,
) -> None:
    """Reject the request if the caller does not own the job.

    Args:
        job: ORM Job row or in-memory Job dataclass.
        fingerprint: Caller fingerprint from ``compute_caller_fingerprint``.
        action: Free-form audit tag (e.g. ``"status"``, ``"delete"``).

    Raises:
        HTTPException(403): Caller fingerprint does not match
            ``job.owner_fingerprint`` (only when that attribute exists and
            is non-empty).
    """
    job_id = getattr(job, "id", "<unknown>")
    owner = getattr(job, "owner_fingerprint", None)

    if owner and owner != fingerprint:
        logger.warning(
            "IDOR_BLOCKED action={} job={} caller={} owner={}",
            action, job_id, fingerprint, owner,
        )
        raise HTTPException(status_code=403, detail="Forbidden")

    logger.debug(
        "job_access action={} job={} caller={} mode={}",
        action, job_id, fingerprint,
        "owned" if owner else "single_tenant",
    )
