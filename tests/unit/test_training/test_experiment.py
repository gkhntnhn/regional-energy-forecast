"""Tests for the MLflow experiment tracker."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult

# ---------------------------------------------------------------------------
# Disabled tracker (noop)
# ---------------------------------------------------------------------------


class TestDisabledTracker:
    """All methods are silent no-ops when disabled."""

    def test_disabled_noop(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        # None of these should raise
        tracker.log_params({"a": 1})
        tracker.log_metrics({"mape": 5.0})
        tracker.log_model(object())
        tracker.log_feature_importance({"feat_a": 0.5})

    def test_disabled_start_run_yields_none(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        with tracker.start_run("test") as run_id:
            assert run_id is None

    def test_disabled_log_split_metrics(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        m = MetricsResult(mape=5.0, mae=10.0, rmse=12.0, r2=0.9, smape=5.0, wmape=5.0, mbe=1.0)
        tracker.log_split_metrics(0, m, m, m)


# ---------------------------------------------------------------------------
# Enabled tracker (mocked MLflow)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mlflow() -> MagicMock:
    """Create a mock mlflow module."""
    mock = MagicMock()
    mock.start_run.return_value.__enter__ = MagicMock(
        return_value=MagicMock(info=MagicMock(run_id="test-run-123"))
    )
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


class TestEnabledTracker:
    """Test that enabled tracker calls mlflow correctly."""

    def test_start_run_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        with tracker.start_run("trial_01") as run_id:
            assert run_id == "test-run-123"

    def test_log_params_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        params: dict[str, Any] = {"depth": 6, "lr": 0.05}
        tracker.log_params(params)
        mock_mlflow.log_params.assert_called_once_with(params)

    def test_log_metrics_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        metrics = {"mape": 5.2, "mae": 10.0}
        tracker.log_metrics(metrics, step=1)
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=1)

    def test_log_model_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        model = MagicMock()
        tracker.log_model(model, artifact_path="catboost")
        mock_mlflow.catboost.log_model.assert_called_once_with(
            model, artifact_path="catboost"
        )

    def test_log_feature_importance_when_enabled(
        self, mock_mlflow: MagicMock
    ) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        importance = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        tracker.log_feature_importance(importance, top_n=2)
        assert mock_mlflow.log_metric.call_count == 2

    def test_log_split_metrics_when_enabled(
        self, mock_mlflow: MagicMock
    ) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        m = MetricsResult(
            mape=5.0, mae=10.0, rmse=12.0, r2=0.9, smape=5.0, wmape=5.0, mbe=1.0
        )
        tracker.log_split_metrics(0, m, m, m)
        mock_mlflow.log_metrics.assert_called_once()
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "split_00_val_mape" in logged
        assert "split_00_test_mae" in logged
