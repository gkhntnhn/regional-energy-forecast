"""Tests for the MLflow experiment tracker."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

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


class TestInit:
    """Test ExperimentTracker initialization."""

    @patch("energy_forecast.training.experiment.mlflow", create=True)
    def test_init_enabled_calls_mlflow(self, mock_mlflow: MagicMock) -> None:
        """Enabled tracker sets tracking URI and experiment."""
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = ExperimentTracker(
                experiment_name="test-exp",
                tracking_uri="http://test:5000",
                enabled=True,
            )
        assert tracker._enabled is True
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://test:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")


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
        mock_mlflow.catboost.log_model.assert_called_once_with(model, name="catboost")

    def test_log_feature_importance_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        importance = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        tracker.log_feature_importance(importance, top_n=2)
        assert mock_mlflow.log_metric.call_count == 2

    def test_log_split_metrics_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        m = MetricsResult(mape=5.0, mae=10.0, rmse=12.0, r2=0.9, smape=5.0, wmape=5.0, mbe=1.0)
        tracker.log_split_metrics(0, m, m, m)
        mock_mlflow.log_metrics.assert_called_once()
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "split_00_val_mape" in logged
        assert "split_00_test_mae" in logged

    @patch("pickle.dump")
    def test_log_prophet_model_when_enabled(
        self, mock_dump: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        model = MagicMock()
        tracker.log_prophet_model(model, artifact_path="prophet")
        mock_dump.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()

    def test_log_tft_model_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        model = MagicMock()
        tracker.log_tft_model(model, artifact_path="tft")
        model.save.assert_called_once()
        mock_mlflow.log_artifacts.assert_called_once()

    def test_log_ensemble_weights_when_enabled(self, mock_mlflow: MagicMock) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        weights = {"catboost": 0.6, "prophet": 0.4}
        tracker.log_ensemble_weights(weights)
        assert mock_mlflow.log_metric.call_count == 2

    def test_disabled_log_prophet_model(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker.log_prophet_model(object())

    def test_disabled_log_tft_model(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker.log_tft_model(object())

    def test_disabled_log_ensemble_weights(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker.log_ensemble_weights({"catboost": 0.5})

    def test_log_training_meta_when_enabled(self, mock_mlflow: MagicMock) -> None:
        """log_training_meta splits numeric vs non-numeric into metrics vs params."""
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        meta: dict[str, Any] = {
            "training_time_seconds": 123.4,
            "data_rows": 48000,
            "python_version": "3.11.14",
        }
        tracker.log_training_meta(meta)
        # training_time_seconds -> metrics, rest -> params
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "training_time_seconds" in logged_metrics
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert "data_rows" in logged_params

    def test_log_config_snapshot_when_enabled(self, mock_mlflow: MagicMock) -> None:
        """log_config_snapshot writes YAML and logs as artifact."""
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        config_dict: dict[str, Any] = {"n_trials": 50, "model": "catboost"}
        tracker.log_config_snapshot(config_dict)
        mock_mlflow.log_artifact.assert_called_once()

    def test_log_predictions_summary_when_enabled(self, mock_mlflow: MagicMock) -> None:
        """log_predictions_summary logs residual statistics."""
        import numpy as np

        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        tracker.log_predictions_summary(y_true, y_pred, prefix="test")
        mock_mlflow.log_metrics.assert_called_once()
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "test_pred_mean" in logged
        assert "test_residual_std" in logged

    def test_log_artifact_when_enabled(self, mock_mlflow: MagicMock) -> None:
        """log_artifact calls mlflow.log_artifact."""
        tracker = ExperimentTracker(enabled=False)
        tracker._enabled = True
        tracker._mlflow = mock_mlflow
        tracker.log_artifact("/tmp/file.txt", artifact_path="outputs")
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/file.txt", "outputs")

    def test_disabled_log_training_meta(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker.log_training_meta({"data_rows": 48000})

    def test_disabled_log_config_snapshot(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker.log_config_snapshot({"n_trials": 50})

    def test_disabled_log_predictions_summary(self) -> None:
        import numpy as np

        tracker = ExperimentTracker(enabled=False)
        tracker.log_predictions_summary(np.array([1.0]), np.array([1.0]))

    def test_disabled_log_artifact(self) -> None:
        tracker = ExperimentTracker(enabled=False)
        tracker.log_artifact("/tmp/file.txt")
