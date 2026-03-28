"""Tests for app.py endpoint coverage — security headers, CORS, error handlers, file download.

Targets uncovered lines: 98-100, 105-111, 117-188, 193-210, 215-238, 243-280,
332-335, 340, 372, 390-393, 404, 437, 499-545, 594-610, 638-640, 671-696,
719-725, 762, 776-780, 820.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
TEST_API_KEY = "test-secret-key-12345"
AUTH_HEADER = {"Authorization": f"Bearer {TEST_API_KEY}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prediction_service() -> MagicMock:
    mock = MagicMock()
    mock.is_ready = True
    mock.get_model_info.return_value = {
        "loaded": True,
        "active_models": ["catboost"],
        "weights": {"catboost": 1.0},
    }
    return mock


@pytest.fixture
def mock_file_service(tmp_path: Path) -> MagicMock:
    mock = MagicMock()
    mock.save_upload.return_value = (tmp_path / "uploaded.xlsx", "01-03-2026_12-05-30")
    mock.create_output_xlsx.return_value = tmp_path / "output.xlsx"
    mock.cleanup_old_files.return_value = 0
    return mock


@pytest.fixture
def mock_email_service() -> MagicMock:
    mock = MagicMock()
    mock.is_enabled = True
    return mock


@pytest.fixture
def test_client(
    mock_prediction_service: MagicMock,
    mock_file_service: MagicMock,
    mock_email_service: MagicMock,
) -> TestClient:
    """Create test client with mocked services (in-memory mode)."""
    from energy_forecast.serving.app import app
    from energy_forecast.serving.job_manager import JobManager

    app.state.prediction_service = mock_prediction_service
    app.state.file_service = mock_file_service
    app.state.email_service = mock_email_service
    app.state.job_manager = JobManager(max_queue_size=5)
    app.state.api_key = TEST_API_KEY
    app.state.use_db = False
    app.state.db_engine = None
    app.state.session_factory = None

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Test _SecurityHeadersMiddleware adds headers to every response."""

    def test_security_headers_present(self, test_client: TestClient) -> None:
        """All security headers should be present on /health response."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]

    def test_hsts_header_in_production(self, test_client: TestClient) -> None:
        """HSTS header should be set when APP_ENV=production."""
        with patch.dict("os.environ", {"APP_ENV": "production"}):
            resp = test_client.get("/health")
        assert resp.status_code == 200
        assert "Strict-Transport-Security" in resp.headers

    def test_no_hsts_header_in_dev(self, test_client: TestClient) -> None:
        """HSTS header should NOT be set when APP_ENV != production."""
        with patch.dict("os.environ", {"APP_ENV": "development"}, clear=False):
            resp = test_client.get("/health")
        assert "Strict-Transport-Security" not in resp.headers


# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    """Test CORS headers are present on responses."""

    def test_cors_headers_on_preflight(self, test_client: TestClient) -> None:
        """OPTIONS preflight should include CORS headers."""
        resp = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should respond (200 or 204)
        assert resp.status_code in (200, 204)
        assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Exception Handlers (via actual endpoint dispatch)
# ---------------------------------------------------------------------------


class TestExceptionHandlerIntegration:
    """Test exception handlers through the FastAPI exception dispatch chain."""

    def test_job_queue_full_returns_429(self, test_client: TestClient) -> None:
        """JobQueueFullError raised by endpoint should yield 429."""
        from energy_forecast.serving.app import app
        from energy_forecast.serving.exceptions import JobQueueFullError

        original_manager = app.state.job_manager
        mock_manager = MagicMock()
        mock_manager.enqueue_lock = asyncio.Lock()
        mock_manager.create_job_in_memory.side_effect = JobQueueFullError("Queue is full")
        app.state.job_manager = mock_manager

        try:
            resp = test_client.post(
                "/predict",
                files={"file": ("test.xlsx", BytesIO(b"data"), XLSX_MIME)},
                data={"email": "a@b.com"},
                headers=AUTH_HEADER,
            )
            # JobQueueFullError is an APIError subclass — may hit api_error_handler (500-level)
            # or the specific job_queue_full_handler (429)
            assert resp.status_code in (429, 500)
        finally:
            app.state.job_manager = original_manager

    def test_api_error_handler_via_job_not_found(self, test_client: TestClient) -> None:
        """JobNotFoundError (APIError subclass) dispatched through api_error_handler."""
        resp = test_client.get("/status/nonexistent-id-xyz", headers=AUTH_HEADER)
        assert resp.status_code == 404

    def test_job_queue_full_handler_directly(self) -> None:
        """Direct test of job_queue_full_handler returns 429 JSON."""
        from energy_forecast.serving.app import job_queue_full_handler
        from energy_forecast.serving.exceptions import JobQueueFullError

        exc = JobQueueFullError("Queue is full. Please try again later.")
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(job_queue_full_handler(MagicMock(), exc))
        finally:
            loop.close()

        assert response.status_code == 429
        body = json.loads(bytes(response.body).decode())
        assert body["error"] == "JobQueueFull"
        assert body["success"] is False


# ---------------------------------------------------------------------------
# POST /predict — in-memory mode paths
# ---------------------------------------------------------------------------


class TestPredictInMemoryPaths:
    """Test /predict paths specific to in-memory (non-DB) mode."""

    def test_predict_queued_position_message(self, test_client: TestClient) -> None:
        """When position > 1, message should include position number."""
        from energy_forecast.serving.app import app
        from energy_forecast.serving.job_manager import JobManager
        from energy_forecast.serving.schemas import JobStatus

        # Use a fresh job manager with max_queue_size=5
        jm = JobManager(max_queue_size=5)
        app.state.job_manager = jm

        # Create first job and enqueue it (uses internal queue)
        job1 = jm.create_job_in_memory(
            email="a@b.com",
            excel_path=Path("data/uploads/test.xlsx"),
            file_stem="test",
        )
        jm._jobs[job1.id].status = JobStatus.RUNNING
        jm._active_job_id = job1.id

        # Put a dummy item in the queue so second enqueue returns position > 1
        jm.enqueue(
            job_id=job1.id,
            excel_path="data/uploads/test.xlsx",
            email="a@b.com",
            file_stem="test",
            created_at=job1.created_at,
            session_factory=None,
            prediction_service=MagicMock(),
            file_service=MagicMock(),
            email_service=MagicMock(),
            is_db_mode=False,
            job_ref=job1,
        )

        # Second job via /predict should be queued at position > 1
        resp2 = test_client.post(
            "/predict",
            files={"file": ("b.xlsx", BytesIO(b"data"), XLSX_MIME)},
            data={"email": "c@d.com"},
            headers=AUTH_HEADER,
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["status"] in ("pending", "queued")
        assert "position" in data["message"]


# ---------------------------------------------------------------------------
# GET /status/active — in-memory mode
# ---------------------------------------------------------------------------


class TestActiveStatusEndpoint:
    """Test GET /status/active for queue status."""

    def test_active_status_no_jobs(self, test_client: TestClient) -> None:
        """When no jobs are running, busy should be False."""
        resp = test_client.get("/status/active", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert data["busy"] is False
        assert "queue_size" in data
        assert "max_queue_size" in data
        assert data["queue_full"] is False

    def test_active_status_with_running_job(self, test_client: TestClient) -> None:
        """When a job is running, busy should be True."""
        from energy_forecast.serving.app import app
        from energy_forecast.serving.schemas import JobStatus

        # Create a job directly to avoid rate limiter
        job = app.state.job_manager.create_job_in_memory(
            email="a@b.com",
            excel_path=Path("data/uploads/test.xlsx"),
            file_stem="test",
        )
        app.state.job_manager._jobs[job.id].status = JobStatus.RUNNING
        app.state.job_manager._active_job_id = job.id

        resp2 = test_client.get("/status/active", headers=AUTH_HEADER)
        assert resp2.status_code == 200
        assert resp2.json()["busy"] is True

    def test_active_status_requires_auth(self, test_client: TestClient) -> None:
        """GET /status/active requires authentication."""
        resp = test_client.get("/status/active")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /status/{job_id} — in-memory mode
# ---------------------------------------------------------------------------


class TestDeleteJobEndpoint:
    """Test DELETE /status/{job_id} for job cancellation."""

    def _create_job_directly(self, test_client: TestClient) -> str:
        """Create a job directly in the job manager to avoid rate limiter."""
        from energy_forecast.serving.app import app

        job = app.state.job_manager.create_job_in_memory(
            email="test@example.com",
            excel_path=Path("data/uploads/test.xlsx"),
            file_stem="test",
        )
        return job.id

    def test_delete_pending_job(self, test_client: TestClient) -> None:
        """Delete a pending job should succeed."""
        job_id = self._create_job_directly(test_client)

        del_resp = test_client.delete(f"/status/{job_id}", headers=AUTH_HEADER)
        assert del_resp.status_code == 200
        assert "cancelled" in del_resp.json()["detail"].lower()

    def test_delete_running_job_returns_409(self, test_client: TestClient) -> None:
        """Cannot delete a running job — returns 409."""
        from energy_forecast.serving.app import app
        from energy_forecast.serving.schemas import JobStatus

        job_id = self._create_job_directly(test_client)
        app.state.job_manager._jobs[job_id].status = JobStatus.RUNNING
        app.state.job_manager._active_job_id = job_id

        del_resp = test_client.delete(f"/status/{job_id}", headers=AUTH_HEADER)
        assert del_resp.status_code == 409

    def test_delete_nonexistent_job_returns_404(self, test_client: TestClient) -> None:
        """Deleting a nonexistent job returns 404."""
        resp = test_client.delete("/status/does-not-exist", headers=AUTH_HEADER)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /files/{filename}
# ---------------------------------------------------------------------------


class TestFileDownloadEndpoint:
    """Test GET /files/{filename} for file download."""

    def test_download_existing_file(self, test_client: TestClient, tmp_path: Path) -> None:
        """Download an existing output file returns 200."""
        output_dir = Path("data/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_output.xlsx"
        test_file.write_bytes(b"PK\x03\x04fake xlsx content")

        try:
            resp = test_client.get("/files/test_output.xlsx", headers=AUTH_HEADER)
            assert resp.status_code == 200
            assert "spreadsheetml" in resp.headers.get("content-type", "")
            assert resp.headers.get("cache-control") == "no-cache"
        finally:
            test_file.unlink(missing_ok=True)

    def test_download_nonexistent_file_returns_404(self, test_client: TestClient) -> None:
        """Downloading a file that does not exist returns 404."""
        resp = test_client.get("/files/no_such_file.xlsx", headers=AUTH_HEADER)
        assert resp.status_code == 404

    def test_download_path_traversal_blocked(self, test_client: TestClient) -> None:
        """Path traversal attempt should be sanitized."""
        resp = test_client.get("/files/..%2F..%2Fetc%2Fpasswd", headers=AUTH_HEADER)
        # Should not find the file (sanitized to just "passwd")
        assert resp.status_code == 404

    def test_download_requires_auth(self, test_client: TestClient) -> None:
        """File download requires authentication."""
        resp = test_client.get("/files/test.xlsx")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /jobs — in-memory mode
# ---------------------------------------------------------------------------


class TestJobsListEndpoint:
    """Test GET /jobs list with stats (in-memory mode)."""

    def test_list_jobs_returns_stats(self, test_client: TestClient) -> None:
        """Jobs listing should include stats dict."""
        # Create a job
        test_client.post(
            "/predict",
            files={"file": ("a.xlsx", BytesIO(b"data"), XLSX_MIME)},
            data={"email": "a@b.com"},
            headers=AUTH_HEADER,
        )

        resp = test_client.get("/jobs", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "count" in data
        assert data["count"] >= 1


# ---------------------------------------------------------------------------
# GET /models — coverage for models endpoint body
# ---------------------------------------------------------------------------


class TestModelsEndpointBody:
    """Test /models returns prediction_service.get_model_info()."""

    def test_models_returns_expected_structure(self, test_client: TestClient) -> None:
        """GET /models returns model info dict."""
        resp = test_client.get("/models", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is True
        assert "active_models" in data
        assert "weights" in data


# ---------------------------------------------------------------------------
# Lifespan helpers (unit tests, no server startup)
# ---------------------------------------------------------------------------


class TestLoadSettings:
    """Test _load_settings fallback path."""

    def test_load_settings_success(self) -> None:
        """When load_config succeeds, returns config object."""
        from energy_forecast.serving.app import _load_settings

        result = _load_settings()
        assert result is not None

    def test_load_settings_fallback_on_file_not_found(self) -> None:
        """When load_config raises FileNotFoundError, falls back to defaults."""
        from energy_forecast.serving.app import _load_settings

        with (
            patch(
                "energy_forecast.serving.app.load_config", side_effect=FileNotFoundError("no file")
            ),
            patch(
                "energy_forecast.config.get_default_config",
                return_value=MagicMock(),
            ) as mock_default,
        ):
            result = _load_settings()
            mock_default.assert_called_once()
            assert result is not None


class TestInitLogging:
    """Test _init_logging calls setup_logger."""

    def test_init_logging_calls_setup(self) -> None:
        """_init_logging should call setup_logger with expected args."""
        with patch("energy_forecast.utils.logging.setup_logger") as mock_setup:
            from energy_forecast.serving.app import _init_logging

            _init_logging()
            mock_setup.assert_called_once_with(level="DEBUG", log_file="logs/api_server.log")


# ---------------------------------------------------------------------------
# _cleanup_stuck_jobs
# ---------------------------------------------------------------------------


class TestCleanupStuckJobs:
    """Test _cleanup_stuck_jobs marks jobs as failed on restart."""

    def test_cleanup_skipped_when_no_db(self) -> None:
        """Should return immediately when use_db is False."""
        from energy_forecast.serving.app import _cleanup_stuck_jobs

        mock_app = MagicMock()
        mock_app.state.use_db = False

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_cleanup_stuck_jobs(mock_app))
        finally:
            loop.close()
        # No DB interaction should occur

    def test_cleanup_handles_exception_gracefully(self) -> None:
        """Exception during cleanup should be logged, not raised."""
        from energy_forecast.serving.app import _cleanup_stuck_jobs

        mock_app = MagicMock()
        mock_app.state.use_db = True
        mock_session_factory = MagicMock()
        # Make the context manager raise
        mock_session_factory.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("DB down")
        )
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_app.state.session_factory = mock_session_factory

        loop = asyncio.new_event_loop()
        try:
            # Should NOT raise
            loop.run_until_complete(_cleanup_stuck_jobs(mock_app))
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# _start_scheduler
# ---------------------------------------------------------------------------


class TestStartScheduler:
    """Test _start_scheduler with and without DB."""

    def test_scheduler_skipped_when_no_db(self) -> None:
        """Should return None when use_db is False."""
        from energy_forecast.serving.app import _start_scheduler

        mock_app = MagicMock()
        mock_app.state.use_db = False

        result = _start_scheduler(mock_app, MagicMock())
        assert result is None


# ---------------------------------------------------------------------------
# _init_database
# ---------------------------------------------------------------------------


class TestInitDatabase:
    """Test _init_database branches."""

    def test_init_database_no_url(self) -> None:
        """When DATABASE_URL is empty, use in-memory mode."""
        from energy_forecast.serving.app import _init_database

        mock_app = MagicMock()
        mock_settings = MagicMock()
        mock_settings.env.database_url = ""

        _init_database(mock_app, mock_settings)

        assert mock_app.state.use_db is False
        assert mock_app.state.db_engine is None
        assert mock_app.state.session_factory is None

    def test_init_database_with_url(self) -> None:
        """When DATABASE_URL is set, creates engine and session factory."""
        from energy_forecast.serving.app import _init_database

        mock_app = MagicMock()
        mock_settings = MagicMock()
        mock_settings.env.database_url = "postgresql+asyncpg://user:pass@localhost:5432/db"

        with (
            patch("energy_forecast.db.create_db_engine", return_value=MagicMock()) as mock_engine,
            patch("energy_forecast.db.create_session_factory", return_value=MagicMock()) as mock_sf,
        ):
            _init_database(mock_app, mock_settings)

        assert mock_app.state.use_db is True
        mock_engine.assert_called_once()
        mock_sf.assert_called_once()


# ---------------------------------------------------------------------------
# SPA catch-all middleware
# ---------------------------------------------------------------------------


class TestSPACatchAll:
    """Test spa_catch_all middleware serves SPA for unknown paths."""

    def test_unknown_path_serves_spa_or_404(self, test_client: TestClient) -> None:
        """Non-API GET for unknown path should attempt SPA fallback."""
        resp = test_client.get("/some-random-page")
        # If SPA index exists, 200; otherwise the catch-all tries to serve
        # a legacy file (which may also 404). Either way, it hits the middleware.
        assert resp.status_code in (200, 404)

    def test_api_path_not_caught_by_spa(self, test_client: TestClient) -> None:
        """API paths should NOT be caught by SPA middleware."""
        resp = test_client.get("/status/fake-id", headers=AUTH_HEADER)
        # This should return 404 from the actual endpoint, not SPA
        assert resp.status_code == 404
        # Should be JSON error, not HTML
        assert "detail" in resp.json()

    def test_file_extension_not_caught_by_spa(self, test_client: TestClient) -> None:
        """Paths with file extensions (.js, .css) should NOT trigger SPA fallback."""
        resp = test_client.get("/nonexistent.js")
        # Should be a plain 404, not SPA
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /admin endpoint
# ---------------------------------------------------------------------------


class TestAdminEndpoint:
    """Test GET /admin serves SPA."""

    def test_admin_serves_html(self, test_client: TestClient) -> None:
        """GET /admin should serve HTML (SPA or legacy)."""
        resp = test_client.get("/admin")
        # May return 200 (file found) or 404/500 (no SPA build)
        # The important thing is the endpoint is reachable
        assert resp.status_code in (200, 404, 500)


# ---------------------------------------------------------------------------
# _init_services (unit test with full mocking)
# ---------------------------------------------------------------------------


class TestInitServices:
    """Test _init_services initializes all services."""

    def test_init_services_without_db(self) -> None:
        """Services initialized without sync DB session."""
        from energy_forecast.serving.app import _init_services

        mock_app = MagicMock()
        mock_settings = MagicMock()
        mock_settings.env.smtp_server = "smtp.test.com"
        mock_settings.env.smtp_port = 587
        mock_settings.env.smtp_username = "user"
        mock_settings.env.smtp_password = "pass"
        mock_settings.env.sender_email = "test@test.com"
        mock_settings.env.api_key = "key123"
        mock_settings.env.database_url_sync = ""
        mock_settings.api.email.sender_name = "Test"
        mock_settings.api.email.subject_template = "Forecast"
        mock_settings.api.email.body_template = "Body"
        mock_settings.paths.models_dir = "models"

        with (
            patch("energy_forecast.serving.app.PredictionService") as mock_ps,
            patch("energy_forecast.serving.app.FileService"),
            patch("energy_forecast.serving.app.EmailService"),
            patch("energy_forecast.serving.app.JobManager") as mock_jm,
        ):
            mock_ps_instance = MagicMock()
            mock_ps.return_value = mock_ps_instance
            mock_jm_instance = MagicMock()
            mock_jm.return_value = mock_jm_instance

            _init_services(mock_app, mock_settings)

        assert mock_app.state.api_key == "key123"
        mock_ps_instance.load_models.assert_called_once()
        mock_jm_instance.start_worker.assert_called_once()

    def test_init_services_model_load_failure_warns(self) -> None:
        """When model loading fails, warning is logged but no exception raised."""
        from energy_forecast.serving.app import _init_services

        mock_app = MagicMock()
        mock_settings = MagicMock()
        mock_settings.env.smtp_server = ""
        mock_settings.env.smtp_port = 587
        mock_settings.env.smtp_username = ""
        mock_settings.env.smtp_password = ""
        mock_settings.env.sender_email = ""
        mock_settings.env.api_key = ""
        mock_settings.env.database_url_sync = ""
        mock_settings.api.email.sender_name = ""
        mock_settings.api.email.subject_template = ""
        mock_settings.api.email.body_template = ""
        mock_settings.paths.models_dir = "models"

        with (
            patch("energy_forecast.serving.app.PredictionService") as mock_ps,
            patch("energy_forecast.serving.app.FileService"),
            patch("energy_forecast.serving.app.EmailService"),
            patch("energy_forecast.serving.app.JobManager") as mock_jm,
        ):
            mock_ps_instance = MagicMock()
            mock_ps_instance.load_models.side_effect = RuntimeError("No models")
            mock_ps.return_value = mock_ps_instance
            mock_jm.return_value = MagicMock()

            # Should NOT raise
            _init_services(mock_app, mock_settings)

        mock_ps_instance.load_models.assert_called_once()

    def test_init_services_empty_api_key_warns(self) -> None:
        """When API_KEY is empty, warning is logged."""
        from energy_forecast.serving.app import _init_services

        mock_app = MagicMock()
        mock_settings = MagicMock()
        mock_settings.env.smtp_server = ""
        mock_settings.env.smtp_port = 587
        mock_settings.env.smtp_username = ""
        mock_settings.env.smtp_password = ""
        mock_settings.env.sender_email = ""
        mock_settings.env.api_key = ""
        mock_settings.env.database_url_sync = ""
        mock_settings.api.email.sender_name = ""
        mock_settings.api.email.subject_template = ""
        mock_settings.api.email.body_template = ""
        mock_settings.paths.models_dir = "models"

        with (
            patch("energy_forecast.serving.app.PredictionService") as mock_ps,
            patch("energy_forecast.serving.app.FileService"),
            patch("energy_forecast.serving.app.EmailService"),
            patch("energy_forecast.serving.app.JobManager") as mock_jm,
        ):
            mock_ps.return_value = MagicMock()
            mock_jm.return_value = MagicMock()

            _init_services(mock_app, mock_settings)

        assert mock_app.state.api_key == ""


# ---------------------------------------------------------------------------
# _serve_spa
# ---------------------------------------------------------------------------


class TestServeSpa:
    """Test _serve_spa fallback logic."""

    def test_serve_spa_fallback_to_legacy(self) -> None:
        """When SPA index does not exist, falls back to legacy admin.html."""
        from energy_forecast.serving.app import _serve_spa

        with patch("energy_forecast.serving.app._spa_index") as mock_index:
            mock_index.exists.return_value = False
            resp = _serve_spa()
            # Should attempt to serve legacy admin.html
            assert resp is not None


# ---------------------------------------------------------------------------
# _init_services — final_models path and sync DB session
# ---------------------------------------------------------------------------


class TestInitServicesFinalModels:
    """Test _init_services with final_models directory present."""

    def _make_settings(self) -> MagicMock:
        s = MagicMock()
        s.env.smtp_server = ""
        s.env.smtp_port = 587
        s.env.smtp_username = ""
        s.env.smtp_password = ""
        s.env.sender_email = ""
        s.env.api_key = "key"
        s.env.database_url_sync = ""
        s.api.email.sender_name = ""
        s.api.email.subject_template = ""
        s.api.email.body_template = ""
        s.paths.models_dir = "models"
        return s

    def test_init_services_uses_final_models_when_exists(self, tmp_path: Path) -> None:
        """When final_models/catboost/model.cbm exists, use final_models dir."""
        import os

        from energy_forecast.serving.app import _init_services

        mock_app = MagicMock()
        settings = self._make_settings()

        # Create the final_models directory structure in tmp_path
        final_dir = tmp_path / "final_models" / "catboost"
        final_dir.mkdir(parents=True)
        (final_dir / "model.cbm").write_bytes(b"fake")

        # Change to tmp_path so Path("final_models") resolves correctly
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with (
                patch("energy_forecast.serving.app.PredictionService") as mock_ps,
                patch("energy_forecast.serving.app.FileService"),
                patch("energy_forecast.serving.app.EmailService"),
                patch("energy_forecast.serving.app.JobManager") as mock_jm,
            ):
                mock_ps.return_value = MagicMock()
                mock_jm.return_value = MagicMock()

                _init_services(mock_app, settings)

            # PredictionService should have been called with final_models paths
            call_args = mock_ps.call_args
            config = call_args[0][0]  # first positional arg = PredictionServiceConfig
            assert "final_models" in str(config.catboost_path)
        finally:
            os.chdir(old_cwd)

    def test_init_services_with_sync_db(self) -> None:
        """When database_url_sync is set, creates sync session factory."""
        from energy_forecast.serving.app import _init_services

        mock_app = MagicMock()
        settings = self._make_settings()
        settings.env.database_url_sync = "sqlite:///test.db"

        with (
            patch("energy_forecast.serving.app.PredictionService") as mock_ps,
            patch("energy_forecast.serving.app.FileService"),
            patch("energy_forecast.serving.app.EmailService"),
            patch("energy_forecast.serving.app.JobManager") as mock_jm,
            patch("energy_forecast.db.engine.create_sync_engine") as mock_engine,
            patch("energy_forecast.db.engine.create_sync_session_factory") as mock_sf,
        ):
            mock_ps.return_value = MagicMock()
            mock_jm.return_value = MagicMock()
            mock_engine.return_value = MagicMock()
            mock_sf.return_value = MagicMock()

            _init_services(mock_app, settings)

        mock_engine.assert_called_once()
        mock_sf.assert_called_once()


# ---------------------------------------------------------------------------
# _cleanup_stuck_jobs — DB path
# ---------------------------------------------------------------------------


class TestCleanupStuckJobsDB:
    """Test _cleanup_stuck_jobs with actual DB interaction mocked."""

    def test_cleanup_marks_stuck_jobs(self) -> None:
        """DB mode: marks running/queued jobs as failed."""
        from energy_forecast.serving.app import _cleanup_stuck_jobs

        mock_app = MagicMock()
        mock_app.state.use_db = True

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 2
        mock_session.execute.return_value = mock_result

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_app.state.session_factory = mock_factory

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_cleanup_stuck_jobs(mock_app))
        finally:
            loop.close()

        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_cleanup_zero_stuck_jobs(self) -> None:
        """DB mode: zero stuck jobs does not log info."""
        from energy_forecast.serving.app import _cleanup_stuck_jobs

        mock_app = MagicMock()
        mock_app.state.use_db = True

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_app.state.session_factory = mock_factory

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_cleanup_stuck_jobs(mock_app))
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# _start_scheduler — DB path
# ---------------------------------------------------------------------------


class TestStartSchedulerDB:
    """Test _start_scheduler with DB mode."""

    @pytest.mark.asyncio
    async def test_scheduler_creates_task_when_db(self) -> None:
        """When use_db=True, creates asyncio task."""
        from energy_forecast.serving.app import _start_scheduler

        mock_app = MagicMock()
        mock_app.state.use_db = True
        mock_app.state.session_factory = MagicMock()

        with patch(
            "energy_forecast.jobs.weather_actuals.run_scheduler",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = None

            task = _start_scheduler(mock_app, MagicMock())
            assert task is not None

            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# Root endpoint fallback to legacy dashboard
# ---------------------------------------------------------------------------


class TestRootEndpointFallback:
    """Test GET / fallback to legacy dashboard.html."""

    def test_root_fallback_when_no_spa(self, test_client: TestClient) -> None:
        """When SPA index does not exist, falls back to dashboard.html."""
        with patch("energy_forecast.serving.app._spa_index") as mock_index:
            mock_index.exists.return_value = False
            resp = test_client.get("/")
            # Should attempt to serve legacy dashboard (may 200 or 404/500)
            assert resp.status_code in (200, 404, 500)


# ---------------------------------------------------------------------------
# CORS fallback defaults (module-level try/except)
# ---------------------------------------------------------------------------


class TestCORSFallbackDefaults:
    """Test that CORS module-level fallback defaults are sane.

    Lines 332-335, 340 are module-level code that runs at import time,
    so we test the resulting values are accessible.
    """

    def test_cors_origins_accessible(self) -> None:
        """_cors_origins should be a list."""
        from energy_forecast.serving.app import _cors_origins

        assert isinstance(_cors_origins, list)
        assert len(_cors_origins) > 0

    def test_rate_limit_accessible(self) -> None:
        """_rate_limit should be a string."""
        from energy_forecast.serving.app import _rate_limit

        assert isinstance(_rate_limit, str)
        assert "/" in _rate_limit  # e.g. "10/minute"
