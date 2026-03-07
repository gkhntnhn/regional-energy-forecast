"""Tests for admin API router endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from energy_forecast.db.models import (
    JobModel,
    ModelRunModel,
    PredictionModel,
)
from energy_forecast.utils import TZ_ISTANBUL

TEST_API_KEY = "test-secret-key-12345"
AUTH_HEADER = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def admin_client(db_engine: Any, db_session_factory: Any) -> TestClient:
    """Create test client with DB for admin endpoints."""
    from energy_forecast.serving.app import app
    from energy_forecast.serving.job_manager import JobManager

    mock_ps = MagicMock()
    mock_ps.is_ready = True
    mock_ps.get_model_info.return_value = {"loaded": True}
    mock_ps.get_feature_importance_top.return_value = None

    app.state.prediction_service = mock_ps
    app.state.file_service = MagicMock()
    app.state.email_service = MagicMock()
    app.state.job_manager = JobManager()
    app.state.api_key = TEST_API_KEY
    app.state.use_db = True
    app.state.db_engine = db_engine
    app.state.session_factory = db_session_factory

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def _seed_predictions(db_session_factory: Any) -> None:
    """Seed DB with sample predictions for analytics tests."""
    async with db_session_factory() as session:
        job = JobModel(
            id="test_j1",
            email="test@test.com",
            status="completed",
            excel_path="/tmp/test.xlsx",
            file_stem="test",
            created_at=datetime.now(tz=TZ_ISTANBUL),
            completed_at=datetime.now(tz=TZ_ISTANBUL),
            metadata_={
                "feature_importance_top15": [
                    {"feature": "consumption_lag_48", "importance": 25.0},
                    {"feature": "temperature_2m", "importance": 15.0},
                ]
            },
            epias_snapshot={
                "last_values": {
                    "Load_Forecast": 4500.0,
                    "Real_Time_Consumption": 4400.0,
                }
            },
        )
        session.add(job)

        for h in range(24):
            dt = datetime(2026, 3, 1, h, tzinfo=TZ_ISTANBUL)
            for model in ["ensemble", "catboost", "prophet"]:
                pred = PredictionModel(
                    job_id="test_j1",
                    forecast_dt=dt,
                    consumption_mwh=1200.0 + h * 10,
                    period="day_ahead",
                    model_source=model,
                    actual_mwh=1250.0 + h * 10,
                    error_pct=abs(1200.0 - 1250.0) / 1250.0 * 100,
                    matched_at=datetime.now(tz=TZ_ISTANBUL),
                    created_at=dt,
                )
                session.add(pred)

        run = ModelRunModel(
            model_type="catboost",
            status="completed",
            val_mape=2.5,
            test_mape=2.8,
        )
        session.add(run)
        await session.commit()


# ---------------------------------------------------------------------------
# No-DB fallback
# ---------------------------------------------------------------------------


class TestAdminNoDb:
    """Admin endpoints return empty results when DB is disabled."""

    def test_daily_mape_no_db(self) -> None:
        from energy_forecast.serving.app import app

        app.state.use_db = False
        app.state.api_key = TEST_API_KEY
        app.state.prediction_service = MagicMock(is_ready=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/admin/analytics/mape/daily", headers=AUTH_HEADER)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_job_history_no_db(self) -> None:
        from energy_forecast.serving.app import app

        app.state.use_db = False
        app.state.api_key = TEST_API_KEY
        app.state.prediction_service = MagicMock(is_ready=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/admin/jobs/history", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# With DB
# ---------------------------------------------------------------------------


class TestAdminWithDb:
    """Admin endpoints with seeded DB data."""

    @pytest.mark.usefixtures("_seed_predictions")
    def test_daily_mape(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/mape/daily?days=30", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            assert "day" in data[0]
            assert "mape" in data[0]

    @pytest.mark.usefixtures("_seed_predictions")
    def test_weekly_mape(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/mape/weekly?weeks=12", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_hourly_mape(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/mape/hourly", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_per_model_mape(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/models/mape?days=30", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_model_comparison(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/models/comparison", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_feature_trend(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/features/trend?days=30", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_epias_accuracy(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/analytics/epias/accuracy?days=30", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_job_history(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/jobs/history?page=1&size=10", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "jobs" in data

    @pytest.mark.usefixtures("_seed_predictions")
    def test_job_details(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/jobs/test_j1/details", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test_j1"
        assert "predictions" in data

    @pytest.mark.usefixtures("_seed_predictions")
    def test_model_runs(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/models/runs?model_type=catboost", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.usefixtures("_seed_predictions")
    def test_system_health(self, admin_client: TestClient) -> None:
        resp = admin_client.get(
            "/admin/system/health", headers=AUTH_HEADER
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "database" in data
        assert "models_loaded" in data


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAdminAuth:
    """Admin endpoints require authentication."""

    def test_no_auth_rejected(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/admin/analytics/mape/daily")
        assert resp.status_code == 401
