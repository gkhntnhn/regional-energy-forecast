"""Tests for IDOR job-access guard (item 237)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from energy_forecast.serving.access_control import (
    assert_job_access,
    compute_caller_fingerprint,
)


class TestComputeCallerFingerprint:
    """SHA-256 fingerprint of bearer token."""

    def test_stable_for_same_token(self) -> None:
        a = compute_caller_fingerprint("token-A")
        b = compute_caller_fingerprint("token-A")
        assert a == b
        assert len(a) == 16

    def test_distinct_tokens_distinct_fingerprints(self) -> None:
        a = compute_caller_fingerprint("token-A")
        b = compute_caller_fingerprint("token-B")
        assert a != b

    def test_none_token_returns_anonymous(self) -> None:
        assert compute_caller_fingerprint(None) == "anonymous"

    def test_empty_token_returns_anonymous(self) -> None:
        assert compute_caller_fingerprint("") == "anonymous"


class TestAssertJobAccess:
    """Per-job ownership check."""

    def test_no_owner_attribute_passes(self) -> None:
        job = SimpleNamespace(id="abc123")
        # Should not raise — single-tenant mode
        assert_job_access(job, fingerprint="caller-fp", action="status")

    def test_none_owner_passes(self) -> None:
        job = SimpleNamespace(id="abc123", owner_fingerprint=None)
        assert_job_access(job, fingerprint="caller-fp", action="status")

    def test_empty_owner_passes(self) -> None:
        job = SimpleNamespace(id="abc123", owner_fingerprint="")
        assert_job_access(job, fingerprint="caller-fp", action="status")

    def test_matching_owner_passes(self) -> None:
        job = SimpleNamespace(id="abc123", owner_fingerprint="caller-fp")
        assert_job_access(job, fingerprint="caller-fp", action="status")

    def test_mismatch_raises_403(self) -> None:
        job = SimpleNamespace(id="abc123", owner_fingerprint="alice-fp")
        with pytest.raises(HTTPException) as exc:
            assert_job_access(job, fingerprint="bob-fp", action="status")
        assert exc.value.status_code == 403
        assert exc.value.detail == "Forbidden"

    def test_mismatch_action_in_log(self, capsys: pytest.CaptureFixture[str]) -> None:
        from loguru import logger
        sink: list[str] = []
        sid = logger.add(lambda m: sink.append(str(m)), level="WARNING")
        try:
            job = SimpleNamespace(id="job-x", owner_fingerprint="alice")
            with pytest.raises(HTTPException):
                assert_job_access(job, fingerprint="bob", action="delete")
        finally:
            logger.remove(sid)
        assert any("IDOR_BLOCKED" in m for m in sink)
        assert any("delete" in m for m in sink)
