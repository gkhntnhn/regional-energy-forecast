"""Tests for internal scheduler endpoints (sync-epias / sync-weather).

Covers:
  - Token gating (missing / wrong / disabled / valid).
  - Endpoint payload shape on success and failure.
  - sync-epias delegates to _run_epias_db_sync (in-thread).
  - sync-weather subprocess parser (regex against loguru-formatted output).
  - Error redaction — exceptions surface only as type + truncated message.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_INTERNAL_TOKEN = "test-internal-token-9876"
TOKEN_HEADER = {"X-Internal-Token": TEST_INTERNAL_TOKEN}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client_with_token() -> TestClient:
    """TestClient with internal_token configured."""
    from energy_forecast.config import get_default_config
    from energy_forecast.serving.app import app

    app.state.api_key = "user-key"
    app.state.internal_token = TEST_INTERNAL_TOKEN
    app.state.settings = get_default_config()
    app.state.use_db = False
    app.state.db_engine = None
    app.state.session_factory = None

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def test_client_no_token() -> TestClient:
    """TestClient with internal_token absent (endpoints disabled)."""
    from energy_forecast.config import get_default_config
    from energy_forecast.serving.app import app

    app.state.api_key = "user-key"
    app.state.internal_token = ""
    app.state.settings = get_default_config()
    app.state.use_db = False
    app.state.db_engine = None
    app.state.session_factory = None

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Token gating
# ---------------------------------------------------------------------------


class TestInternalTokenGating:
    """Auth surface for /internal/* endpoints."""

    def test_missing_token_returns_401(self, test_client_with_token: TestClient) -> None:
        resp = test_client_with_token.post("/internal/sync-epias")
        assert resp.status_code == 401
        assert "internal token" in resp.json()["detail"].lower()

    def test_wrong_token_returns_401(self, test_client_with_token: TestClient) -> None:
        resp = test_client_with_token.post(
            "/internal/sync-epias",
            headers={"X-Internal-Token": "wrong-secret"},
        )
        assert resp.status_code == 401

    def test_disabled_token_returns_503(self, test_client_no_token: TestClient) -> None:
        resp = test_client_no_token.post(
            "/internal/sync-epias",
            headers=TOKEN_HEADER,
        )
        assert resp.status_code == 503
        assert "INTERNAL_TOKEN" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# sync-epias
# ---------------------------------------------------------------------------


class TestSyncEpias:
    """sync-epias endpoint payload + delegation."""

    def test_returns_payload_shape(self, test_client_with_token: TestClient) -> None:
        """Response must include all SyncResult fields."""
        with patch(
            "energy_forecast.serving.routers.internal._run_epias_db_sync",
            return_value=(7, 100, 100, "2026-04-26"),
        ):
            resp = test_client_with_token.post(
                "/internal/sync-epias?days_back=7",
                headers=TOKEN_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sync_type"] == "epias"
        assert body["days_back"] == 7
        assert body["rows_upserted"] == 200
        assert body["last_date"] == "2026-04-26"
        assert body["errors"] == []
        assert "duration_ms" in body

    def test_invalid_days_back_returns_422(self, test_client_with_token: TestClient) -> None:
        """days_back < 1 should fail Pydantic Query validation."""
        resp = test_client_with_token.post(
            "/internal/sync-epias?days_back=0",
            headers=TOKEN_HEADER,
        )
        assert resp.status_code == 422

    def test_exception_redacted_in_errors(self, test_client_with_token: TestClient) -> None:
        """Exception in _run_epias_db_sync surfaces as redacted errors[]."""

        def raise_with_secret(*_args: object, **_kwargs: object) -> None:
            msg = "DB password=hunter2 leaked\nstack frame line 2"
            raise RuntimeError(msg)

        with patch(
            "energy_forecast.serving.routers.internal._run_epias_db_sync",
            side_effect=raise_with_secret,
        ):
            resp = test_client_with_token.post(
                "/internal/sync-epias",
                headers=TOKEN_HEADER,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["errors"]) == 1
        # First-line truncation removes any "stack frame line 2" leak
        assert "stack frame line 2" not in body["errors"][0]
        assert body["errors"][0].startswith("RuntimeError:")
        assert body["rows_upserted"] == 0


# ---------------------------------------------------------------------------
# sync-weather subprocess parser
# ---------------------------------------------------------------------------


class TestSyncWeatherParser:
    """Regex parser for seed_weather.py loguru output."""

    def test_parser_extracts_aggregate_and_grid(
        self, test_client_with_token: TestClient
    ) -> None:
        """Parser extracts both aggregate and grid totals from prefixed log."""
        sample_stderr = b"""
15:50:14 | INFO    | Weather seed start
15:50:24 | INFO    |   2024 -> DB from parquet (8784 rows)
15:50:33 | INFO    |   2025 -> DB from parquet (8760 rows)
15:51:58 | INFO    |   2024 -> grid DB (131760 rows, 15 points)
15:52:35 | INFO    |   2025 -> grid DB (131400 rows, 15 points)
15:55:07 | INFO    | Grid weather seed: 263160 rows
""".strip()
        fake_proc = MagicMock(returncode=0, stdout=b"", stderr=sample_stderr)

        with patch("subprocess.run", return_value=fake_proc):
            resp = test_client_with_token.post(
                "/internal/sync-weather?days_back=2",
                headers=TOKEN_HEADER,
            )

        assert resp.status_code == 200
        body = resp.json()
        # 2 aggregate x ~8.7K + 2 grid x ~131K
        assert body["rows_upserted"] == 8784 + 8760 + 131760 + 131400
        assert body["last_date"] == "2025"  # max year token
        assert body["errors"] == []

    def test_parser_strips_ansi_color_codes(
        self, test_client_with_token: TestClient
    ) -> None:
        """ANSI color codes from loguru terminal mode must not break the regex."""
        sample = (
            b"\x1b[32m15:50:24\x1b[0m | \x1b[1mINFO   \x1b[0m |   "
            b"2024 -> DB from parquet (8784 rows)\n"
            b"\x1b[32m15:51:58\x1b[0m | \x1b[1mINFO   \x1b[0m |   "
            b"2024 -> grid DB (131760 rows, 15 points)\n"
        )
        fake_proc = MagicMock(returncode=0, stdout=b"", stderr=sample)

        with patch("subprocess.run", return_value=fake_proc):
            resp = test_client_with_token.post(
                "/internal/sync-weather",
                headers=TOKEN_HEADER,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["rows_upserted"] == 8784 + 131760

    def test_subprocess_failure_surfaces_in_errors(
        self, test_client_with_token: TestClient
    ) -> None:
        """Non-zero subprocess exit code populates errors[] without stack leak."""
        fake_proc = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"FATAL: connection refused\nDB host=secret-internal-db",
        )

        with patch("subprocess.run", return_value=fake_proc):
            resp = test_client_with_token.post(
                "/internal/sync-weather",
                headers=TOKEN_HEADER,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["errors"]) == 1
        assert body["errors"][0].startswith("RuntimeError:")
        # Subprocess output gets put into log_tail in the audit path,
        # but must not leak into the user-facing errors[] field
        assert "secret-internal-db" not in body["errors"][0]
        assert body["rows_upserted"] == 0
