"""Tests for FastAPI application endpoints."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
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
    mock.save_upload.return_value = tmp_path / "uploaded.xlsx"
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

        # Manually set job to running state
        from energy_forecast.serving.app import app
        app.state.job_manager._set_running(job_id)

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
