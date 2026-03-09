"""Tests for FastAPI application endpoints."""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Excel MIME type constant to keep lines short
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
TEST_API_KEY = "test-secret-key-12345"
AUTH_HEADER = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def mock_prediction_service() -> MagicMock:
    """Create mock prediction service."""
    mock = MagicMock()
    mock.is_ready = True
    mock.run_prediction.return_value = pd.DataFrame(
        {"prediction": [1000.0] * 48},
        index=pd.date_range("2025-01-01", periods=48, freq="h"),
    )
    mock.get_model_info.return_value = {
        "loaded": True,
        "active_models": ["catboost", "prophet"],
        "weights": {"catboost": 0.6, "prophet": 0.4},
    }
    return mock


@pytest.fixture
def mock_file_service(tmp_path: Path) -> MagicMock:
    """Create mock file service."""
    mock = MagicMock()
    mock.save_upload.return_value = (tmp_path / "uploaded.xlsx", "01-03-2026_12-05-30")
    mock.create_output_xlsx.return_value = tmp_path / "output.xlsx"
    mock.cleanup_old_files.return_value = 0
    return mock


@pytest.fixture
def mock_email_service() -> MagicMock:
    """Create mock email service."""
    mock = MagicMock()
    mock.is_enabled = True
    mock.send_prediction_result.return_value = True
    mock.send_error_notification.return_value = True
    return mock


@pytest.fixture
def test_client(
    mock_prediction_service: MagicMock,
    mock_file_service: MagicMock,
    mock_email_service: MagicMock,
) -> TestClient:
    """Create test client with mocked services."""
    from energy_forecast.serving.app import app
    from energy_forecast.serving.job_manager import JobManager

    # Set up app state with mocks
    app.state.prediction_service = mock_prediction_service
    app.state.file_service = mock_file_service
    app.state.email_service = mock_email_service
    app.state.job_manager = JobManager()
    app.state.api_key = TEST_API_KEY
    app.state.use_db = False
    app.state.db_engine = None
    app.state.session_factory = None

    return TestClient(app, raise_server_exceptions=False)



class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, test_client: TestClient) -> None:
        """Test health check returns OK."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"


class TestAuthentication:
    """Tests for API key authentication."""

    def test_health_no_auth_required(self, test_client: TestClient) -> None:
        """Health endpoint should work without auth."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_predict_requires_auth(self, test_client: TestClient) -> None:
        """Predict endpoint rejects requests without auth."""
        excel_content = BytesIO(b"fake excel content")
        response = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "test@example.com"},
        )
        assert response.status_code == 401

    def test_predict_rejects_wrong_key(self, test_client: TestClient) -> None:
        """Predict endpoint rejects wrong API key."""
        excel_content = BytesIO(b"fake excel content")
        response = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "test@example.com"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_jobs_requires_auth(self, test_client: TestClient) -> None:
        """Jobs endpoint rejects requests without auth."""
        response = test_client.get("/jobs")
        assert response.status_code == 401

    def test_models_requires_auth(self, test_client: TestClient) -> None:
        """Models endpoint rejects requests without auth."""
        response = test_client.get("/models")
        assert response.status_code == 401


class TestModelsEndpoint:
    """Tests for /models endpoint."""

    def test_get_models_info(self, test_client: TestClient) -> None:
        """Test getting model information."""
        response = test_client.get("/models", headers=AUTH_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is True
        assert "catboost" in data["active_models"]


class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_success(
        self,
        test_client: TestClient,
        mock_file_service: MagicMock,
    ) -> None:
        """Test successful prediction request."""
        # Create a simple Excel file content
        excel_content = BytesIO(b"fake excel content")

        response = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "test@example.com"},
            headers=AUTH_HEADER,
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "queued" in data["message"].lower()

    def test_predict_invalid_email(self, test_client: TestClient) -> None:
        """Test prediction with invalid email."""
        excel_content = BytesIO(b"fake excel content")

        response = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "not-an-email"},
            headers=AUTH_HEADER,
        )

        assert response.status_code == 422  # Validation error

    def test_predict_models_not_loaded(
        self,
        test_client: TestClient,
        mock_prediction_service: MagicMock,
    ) -> None:
        """Test prediction when models not loaded."""
        mock_prediction_service.is_ready = False

        excel_content = BytesIO(b"fake excel content")
        response = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "test@example.com"},
            headers=AUTH_HEADER,
        )

        assert response.status_code == 503

    def test_predict_job_already_running(
        self,
        test_client: TestClient,
        mock_prediction_service: MagicMock,
    ) -> None:
        """Test prediction when another job is running."""
        mock_prediction_service.is_ready = True

        # First request - should succeed
        excel_content1 = BytesIO(b"fake excel content")
        response1 = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content1, XLSX_MIME)},
            data={"email": "test1@example.com"},
            headers=AUTH_HEADER,
        )
        assert response1.status_code == 200
        job_id = response1.json()["job_id"]

        # Manually set job to running state (in-memory mode)
        from energy_forecast.serving.app import app
        from energy_forecast.serving.schemas import JobStatus

        app.state.job_manager._jobs[job_id].status = JobStatus.RUNNING
        app.state.job_manager._active_job_id = job_id

        # Second request - should fail with 429
        excel_content2 = BytesIO(b"fake excel content")
        response2 = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content2, XLSX_MIME)},
            data={"email": "test2@example.com"},
            headers=AUTH_HEADER,
        )
        assert response2.status_code == 429


class TestStatusEndpoint:
    """Tests for GET /status/{job_id} endpoint."""

    def test_get_status_pending(self, test_client: TestClient) -> None:
        """Test getting status of a job.

        Note: TestClient runs BackgroundTasks synchronously, so the job
        may complete before we check status. We accept any valid status.
        """
        # Create a job first
        excel_content = BytesIO(b"fake excel content")
        create_response = test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "test@example.com"},
            headers=AUTH_HEADER,
        )
        job_id = create_response.json()["job_id"]

        # Get status
        response = test_client.get(f"/status/{job_id}", headers=AUTH_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        # TestClient runs tasks synchronously, so job may already be completed
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_get_status_not_found(self, test_client: TestClient) -> None:
        """Test getting status of nonexistent job."""
        response = test_client.get("/status/nonexistent123", headers=AUTH_HEADER)

        assert response.status_code == 404


class TestJobsEndpoint:
    """Tests for GET /jobs endpoint."""

    def test_list_jobs_empty(self, test_client: TestClient) -> None:
        """Test listing jobs when empty."""
        # Reset job manager
        from energy_forecast.serving.app import app
        from energy_forecast.serving.job_manager import JobManager
        app.state.job_manager = JobManager()

        response = test_client.get("/jobs", headers=AUTH_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["jobs"] == []

    def test_list_jobs_with_data(self, test_client: TestClient) -> None:
        """Test listing jobs with data."""
        # Create a job
        excel_content = BytesIO(b"fake excel content")
        test_client.post(
            "/predict",
            files={"file": ("test.xlsx", excel_content, XLSX_MIME)},
            data={"email": "test@example.com"},
            headers=AUTH_HEADER,
        )

        response = test_client.get("/jobs", headers=AUTH_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["jobs"]) == 1
        # Email should be masked
        assert "***" in data["jobs"][0]["email"]


class TestApiKeyNotConfigured:
    """Tests for verify_api_key when API key is not set on server (line 84)."""

    def test_empty_api_key_returns_401(self, test_client: TestClient) -> None:
        """Test that empty server API key rejects all authenticated requests."""
        from energy_forecast.serving.app import app

        original_key = app.state.api_key
        try:
            app.state.api_key = ""
            response = test_client.get("/models", headers=AUTH_HEADER)
            assert response.status_code == 401
            assert "not configured" in response.json()["detail"].lower()
        finally:
            app.state.api_key = original_key


class TestRateLimitHandler:
    """Tests for _rate_limit_exceeded_handler (lines 103-104)."""

    def test_returns_429_json_response(self) -> None:
        """Test handler returns 429 with structured JSON body."""
        from energy_forecast.serving.app import _rate_limit_exceeded_handler

        mock_request = MagicMock()
        mock_exc = MagicMock()
        mock_exc.detail = "Rate limit exceeded"

        response = _rate_limit_exceeded_handler(mock_request, mock_exc)

        assert response.status_code == 429
        # JSONResponse body is bytes; decode and check
        import json

        body = json.loads(bytes(response.body).decode())
        assert body["success"] is False
        assert "rate limit" in body["error"].lower()
        assert body["detail"] == "Rate limit exceeded"

    def test_exc_without_detail_attribute(self) -> None:
        """Test handler works when exception has no .detail attribute."""
        from energy_forecast.serving.app import _rate_limit_exceeded_handler

        mock_request = MagicMock()
        exc = RuntimeError("too many requests")
        # RuntimeError has no .detail, so getattr should fall back to str(exc)

        response = _rate_limit_exceeded_handler(mock_request, exc)

        assert response.status_code == 429
        import json

        body = json.loads(bytes(response.body).decode())
        assert "too many requests" in body["detail"]


class TestApiErrorHandler:
    """Tests for api_error_handler (line 270)."""

    def test_returns_structured_error_response(self) -> None:
        """Test handler returns JSONResponse with correct status and body."""
        from energy_forecast.serving.app import api_error_handler
        from energy_forecast.serving.exceptions import APIError

        mock_request = MagicMock()
        exc = APIError(detail="Something went wrong", status_code=400)

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(api_error_handler(mock_request, exc))
        finally:
            loop.close()

        assert response.status_code == 400
        import json

        body = json.loads(bytes(response.body).decode())
        assert body["success"] is False
        assert body["error"] == "APIError"
        assert body["detail"] == "Something went wrong"

    def test_handles_subclass_exceptions(self) -> None:
        """Test handler works with APIError subclasses not caught by specific handlers."""
        from energy_forecast.serving.app import api_error_handler
        from energy_forecast.serving.exceptions import PredictionError

        mock_request = MagicMock()
        exc = PredictionError(detail="Pipeline failed")

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(api_error_handler(mock_request, exc))
        finally:
            loop.close()

        assert response.status_code == 500
        import json

        body = json.loads(bytes(response.body).decode())
        assert body["error"] == "PredictionError"
        assert body["detail"] == "Pipeline failed"


class TestStatusUnexpectedError:
    """Tests for get_status unexpected exception path (lines 390-392)."""

    def test_get_status_unexpected_error_returns_500(
        self, test_client: TestClient
    ) -> None:
        """Test that a non-JobNotFoundError exception returns 500."""
        from energy_forecast.serving.app import app

        # Replace job_manager with a mock that raises RuntimeError
        original_manager = app.state.job_manager
        mock_manager = MagicMock()
        mock_manager.get_job_in_memory = MagicMock(
            side_effect=RuntimeError("DB connection lost")
        )
        app.state.job_manager = mock_manager

        try:
            response = test_client.get("/status/some-job-id", headers=AUTH_HEADER)
            assert response.status_code == 500
            assert "internal server error" in response.json()["detail"].lower()
        finally:
            app.state.job_manager = original_manager


class TestRootEndpoint:
    """Tests for root endpoint / (line 442)."""

    def test_root_serves_dashboard(self, test_client: TestClient) -> None:
        """Test that GET / returns the dashboard HTML file."""
        response = test_client.get("/")

        # The static file exists, so it should return 200 with HTML content
        assert response.status_code == 200
        assert "html" in response.headers.get("content-type", "").lower()
