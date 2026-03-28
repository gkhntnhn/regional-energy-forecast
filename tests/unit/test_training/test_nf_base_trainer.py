"""Unit tests for NeuralForecastTrainer abstract base class.

Tests the shared orchestration logic (TSCV, Optuna, MLflow, OOF cache)
via a minimal concrete subclass (MockNFTrainer).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import (
    CrossValidationConfig,
    ModelSearchConfig,
)
from energy_forecast.training.metrics import MetricsResult
from energy_forecast.training.nf_base_trainer import (
    NeuralForecastTrainer,
    NFPipelineResult,
    NFTrainingResult,
)
from energy_forecast.training.results import SplitResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(mape: float = 2.5) -> MetricsResult:
    """Create a MetricsResult with plausible values."""
    return MetricsResult(mape=mape, mae=50.0, rmse=70.0, r2=0.95, smape=2.4, wmape=2.3, mbe=-1.0)


def _make_split_result(idx: int = 0, val_mape: float = 2.5, test_mape: float = 3.0) -> SplitResult:
    """Create a SplitResult with controllable MAPEs."""
    return SplitResult(
        split_idx=idx,
        train_metrics=_make_metrics(1.5),
        val_metrics=_make_metrics(val_mape),
        test_metrics=_make_metrics(test_mape),
        val_month="2024-06",
        test_month="2024-07",
        val_predictions=np.array([100.0, 110.0]),
        val_actuals=np.array([102.0, 108.0]),
        test_predictions=np.array([105.0, 115.0]),
        test_actuals=np.array([103.0, 112.0]),
    )


def _make_training_result(
    n_splits: int = 2,
    val_mape: float = 2.5,
    test_mape: float = 3.0,
) -> NFTrainingResult:
    """Create an NFTrainingResult with n_splits split results."""
    splits = [_make_split_result(i, val_mape, test_mape) for i in range(n_splits)]
    return NFTrainingResult(
        split_results=splits,
        avg_val_mape=val_mape,
        avg_test_mape=test_mape,
        std_val_mape=0.1,
    )


# ---------------------------------------------------------------------------
# Concrete mock subclass
# ---------------------------------------------------------------------------


class MockNFTrainer(NeuralForecastTrainer):
    """Concrete subclass implementing all abstract methods for testing."""

    @property
    def _model_name(self) -> str:
        return "mock_nf"

    @property
    def _model_config(self) -> Any:
        return self._settings.mock_nf

    @property
    def _hp_search_config(self) -> Any:
        return self._hp_config.mock_nf

    def _build_nf_config(self, params: dict[str, Any]) -> Any:
        return {"built_from": params}

    def _create_forecaster(self, config: Any) -> Any:
        return MagicMock(name="MockForecaster")

    def _get_futr_exog_list(self) -> list[str]:
        return ["hour_sin", "hour_cos"]

    def _get_hist_exog_list(self) -> list[str]:
        return ["temperature_2m"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model_config() -> MagicMock:
    """Minimal model config matching NF trainer expectations."""
    cfg = MagicMock()
    cfg.optimization.n_jobs = 1
    cfg.optimization.optuna_splits = 2
    cfg.optimization.val_size_hours = 720
    cfg.training.random_seed = 42
    cfg.best_params = {}
    cfg.model_dump.return_value = {"mock": True}
    return cfg


@pytest.fixture
def mock_hp_config() -> MagicMock:
    """Mock hyperparameter config that mimics HyperparameterConfig."""
    hp = MagicMock()
    hp.cross_validation = CrossValidationConfig(n_splits=2, val_months=1, test_months=1)
    hp.target_col = "consumption"
    hp.skip_validation_after_optuna = False
    hp.mock_nf = ModelSearchConfig(n_trials=2, search_space={})
    return hp


@pytest.fixture
def mock_settings(mock_model_config: MagicMock, mock_hp_config: MagicMock) -> MagicMock:
    """Settings with mock_nf model config."""
    settings = MagicMock()
    settings.mock_nf = mock_model_config
    settings.hyperparameters = mock_hp_config
    settings.paths.models_dir = "models"
    return settings


@pytest.fixture
def trainer(mock_settings: MagicMock) -> MockNFTrainer:
    """Create a MockNFTrainer with disabled tracker."""
    return MockNFTrainer(mock_settings)


# ---------------------------------------------------------------------------
# Tests: _train_split
# ---------------------------------------------------------------------------


class TestTrainSplit:
    """Tests for _train_split (lines 184-261)."""

    def test_train_split_returns_split_result(self, trainer: MockNFTrainer) -> None:
        """_train_split returns SplitResult with correct metrics."""
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        train_df = pd.DataFrame({"consumption": np.full(100, 1000.0)}, index=idx[:100])
        val_df = pd.DataFrame({"consumption": np.full(50, 1050.0)}, index=idx[100:150])
        test_df = pd.DataFrame({"consumption": np.full(50, 1100.0)}, index=idx[150:200])

        from energy_forecast.training.splitter import SplitInfo

        split_info = SplitInfo(
            split_idx=0,
            train_start=idx[0],
            train_end=idx[99],
            val_start=idx[100],
            val_end=idx[149],
            test_start=idx[150],
            test_end=idx[199],
        )

        # Mock forecaster's train/predict/rolling_predict
        mock_forecaster = MagicMock()
        pred_df = pd.DataFrame(
            {"consumption_mwh": np.full(48, 1010.0)},
            index=idx[:48],
        )
        val_pred_df = pd.DataFrame(
            {"consumption_mwh": np.full(50, 1055.0)},
            index=idx[100:150],
        )
        test_pred_df = pd.DataFrame(
            {"consumption_mwh": np.full(50, 1095.0)},
            index=idx[150:200],
        )
        mock_forecaster.predict.return_value = pred_df
        mock_forecaster.rolling_predict.side_effect = [val_pred_df, test_pred_df]

        with patch.object(trainer, "_create_forecaster", return_value=mock_forecaster):
            result = trainer._train_split(split_info, train_df, val_df, test_df, params={})

        assert isinstance(result, SplitResult)
        assert result.split_idx == 0
        assert result.val_metrics.mape >= 0
        assert result.test_metrics.mape >= 0
        assert result.val_predictions is not None
        assert result.test_predictions is not None

    def test_train_split_cleans_up_model(self, trainer: MockNFTrainer) -> None:
        """_train_split deletes model and calls cuda cache cleanup."""
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        train_df = pd.DataFrame({"consumption": np.full(100, 1000.0)}, index=idx[:100])
        val_df = pd.DataFrame({"consumption": np.full(50, 1050.0)}, index=idx[100:150])
        test_df = pd.DataFrame({"consumption": np.full(50, 1100.0)}, index=idx[150:200])

        from energy_forecast.training.splitter import SplitInfo

        split_info = SplitInfo(
            split_idx=0,
            train_start=idx[0],
            train_end=idx[99],
            val_start=idx[100],
            val_end=idx[149],
            test_start=idx[150],
            test_end=idx[199],
        )

        mock_forecaster = MagicMock()
        pred_df = pd.DataFrame({"consumption_mwh": np.full(48, 1010.0)}, index=idx[:48])
        val_pred_df = pd.DataFrame({"consumption_mwh": np.full(50, 1055.0)}, index=idx[100:150])
        test_pred_df = pd.DataFrame({"consumption_mwh": np.full(50, 1095.0)}, index=idx[150:200])
        mock_forecaster.predict.return_value = pred_df
        mock_forecaster.rolling_predict.side_effect = [val_pred_df, test_pred_df]

        with (
            patch.object(trainer, "_create_forecaster", return_value=mock_forecaster),
            patch.object(trainer, "_maybe_empty_cuda_cache") as mock_cuda,
        ):
            trainer._train_split(split_info, train_df, val_df, test_df, params={})

        mock_cuda.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _maybe_empty_cuda_cache (lines 150-153)
# ---------------------------------------------------------------------------


class TestMaybeEmptyCudaCache:
    """Tests for CUDA cache cleanup."""

    def test_no_cuda_available(self, trainer: MockNFTrainer) -> None:
        """No-op when CUDA is not available."""
        with patch("energy_forecast.training.nf_base_trainer.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            trainer._maybe_empty_cuda_cache()
            mock_torch.cuda.empty_cache.assert_not_called()

    def test_cuda_available_single_job(self, trainer: MockNFTrainer) -> None:
        """Empties cache when CUDA available and n_jobs=1."""
        trainer._settings.mock_nf.optimization.n_jobs = 1
        with patch("energy_forecast.training.nf_base_trainer.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            trainer._maybe_empty_cuda_cache()
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_cuda_available_multi_job(self, trainer: MockNFTrainer) -> None:
        """Skips cache clear when n_jobs > 1."""
        trainer._settings.mock_nf.optimization.n_jobs = 4
        with patch("energy_forecast.training.nf_base_trainer.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            trainer._maybe_empty_cuda_cache()
            mock_torch.cuda.empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _create_objective (lines 307-376)
# ---------------------------------------------------------------------------


class TestCreateObjective:
    """Tests for Optuna objective creation (lines 307-376)."""

    def test_no_splits_raises(self, trainer: MockNFTrainer) -> None:
        """ValueError when no CV splits available."""
        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter([])),
            pytest.raises(ValueError, match="No CV splits"),
        ):
            trainer._create_objective(pd.DataFrame())

    def test_objective_returns_callable(self, trainer: MockNFTrainer) -> None:
        """_create_objective returns (callable, dict)."""
        from energy_forecast.training.splitter import SplitInfo

        info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-03-31"),
            val_start=pd.Timestamp("2024-04-01"),
            val_end=pd.Timestamp("2024-04-30"),
            test_start=pd.Timestamp("2024-05-01"),
            test_end=pd.Timestamp("2024-05-31"),
        )
        mock_split = (info, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        with patch.object(trainer._splitter, "iter_splits", return_value=iter([mock_split])):
            objective, trial_results = trainer._create_objective(pd.DataFrame())

        assert callable(objective)
        assert isinstance(trial_results, dict)

    def test_selects_subset_when_optuna_splits_smaller(self, trainer: MockNFTrainer) -> None:
        """When optuna_splits < total splits, selects evenly spaced subset (lines 330-331)."""
        from energy_forecast.training.splitter import SplitInfo

        splits = []
        for i in range(6):
            info = SplitInfo(
                split_idx=i,
                train_start=pd.Timestamp("2024-01-01"),
                train_end=pd.Timestamp("2024-03-31"),
                val_start=pd.Timestamp("2024-04-01"),
                val_end=pd.Timestamp("2024-04-30"),
                test_start=pd.Timestamp("2024-05-01"),
                test_end=pd.Timestamp("2024-05-31"),
            )
            splits.append((info, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))

        # Set optuna_splits=2 so it must select a subset of 6
        trainer._settings.mock_nf.optimization.optuna_splits = 2

        with patch.object(trainer._splitter, "iter_splits", return_value=iter(splits)):
            objective, _ = trainer._create_objective(pd.DataFrame())

        assert callable(objective)

    def test_objective_caches_trial_results(self, trainer: MockNFTrainer) -> None:
        """Objective stores split results in trial_results dict (lines 369-374)."""
        from energy_forecast.training.splitter import SplitInfo

        info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-03-31"),
            val_start=pd.Timestamp("2024-04-01"),
            val_end=pd.Timestamp("2024-04-30"),
            test_start=pd.Timestamp("2024-05-01"),
            test_end=pd.Timestamp("2024-05-31"),
        )
        mock_split = (info, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        sr = _make_split_result(0)

        # Keep _train_split patched for both _create_objective AND objective() call
        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter([mock_split])),
            patch.object(trainer, "_train_split", return_value=sr),
            patch(
                "energy_forecast.training.nf_base_trainer.suggest_params",
                return_value={"lr": 0.01},
            ),
        ):
            objective, trial_results = trainer._create_objective(pd.DataFrame())

            # Run objective with a mock trial
            mock_trial = MagicMock()
            mock_trial.number = 7
            val = objective(mock_trial)

        assert val == sr.val_metrics.mape
        assert 7 in trial_results
        assert len(trial_results[7]) == 1

    def test_objective_returns_inf_on_split_failure(self, trainer: MockNFTrainer) -> None:
        """Objective returns inf when a split training fails (line 367)."""
        from energy_forecast.training.splitter import SplitInfo

        info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-03-31"),
            val_start=pd.Timestamp("2024-04-01"),
            val_end=pd.Timestamp("2024-04-30"),
            test_start=pd.Timestamp("2024-05-01"),
            test_end=pd.Timestamp("2024-05-31"),
        )
        mock_split = (info, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter([mock_split])),
            patch.object(trainer, "_train_split", side_effect=RuntimeError("boom")),
            patch(
                "energy_forecast.training.nf_base_trainer.suggest_params",
                return_value={},
            ),
        ):
            objective, _ = trainer._create_objective(pd.DataFrame())

            mock_trial = MagicMock()
            mock_trial.number = 0
            val = objective(mock_trial)

        assert val == float("inf")

    def test_objective_reraises_trial_pruned(self, trainer: MockNFTrainer) -> None:
        """TrialPruned is re-raised, not caught as general exception (line 363-364)."""
        from optuna import TrialPruned

        from energy_forecast.training.splitter import SplitInfo

        info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-03-31"),
            val_start=pd.Timestamp("2024-04-01"),
            val_end=pd.Timestamp("2024-04-30"),
            test_start=pd.Timestamp("2024-05-01"),
            test_end=pd.Timestamp("2024-05-31"),
        )
        mock_split = (info, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        with (
            patch.object(trainer._splitter, "iter_splits", return_value=iter([mock_split])),
            patch.object(trainer, "_train_split", side_effect=TrialPruned()),
            patch(
                "energy_forecast.training.nf_base_trainer.suggest_params",
                return_value={},
            ),
        ):
            objective, _ = trainer._create_objective(pd.DataFrame())

            mock_trial = MagicMock()
            mock_trial.number = 0
            with pytest.raises(TrialPruned):
                objective(mock_trial)


# ---------------------------------------------------------------------------
# Tests: _run_fixed (lines 561-599)
# ---------------------------------------------------------------------------


class TestRunFixed:
    """Tests for _run_fixed method (lines 561-599)."""

    def test_run_fixed_uses_best_params(self, trainer: MockNFTrainer) -> None:
        """_run_fixed reads best_params from config and trains all splits."""
        trainer._settings.mock_nf.best_params = {"lr": 0.01, "n_block": 2}

        best_result = _make_training_result()
        mock_model = MagicMock()

        with (
            patch.object(trainer, "_train_all_splits", return_value=best_result),
            patch.object(trainer, "train_final", return_value=mock_model),
            patch.object(trainer, "_save_oof_cache"),
        ):
            result = trainer._run_fixed(pd.DataFrame())

        assert isinstance(result, NFPipelineResult)
        assert result.study is None
        assert result.best_params == {"lr": 0.01, "n_block": 2}
        assert result.final_model is mock_model
        assert result.training_time_seconds >= 0

    def test_run_fixed_saves_model(self, trainer: MockNFTrainer, tmp_path: Any) -> None:
        """_run_fixed saves model to disk."""
        trainer._settings.mock_nf.best_params = {"lr": 0.01}
        trainer._settings.paths.models_dir = str(tmp_path)

        best_result = _make_training_result()
        mock_model = MagicMock()

        with (
            patch.object(trainer, "_train_all_splits", return_value=best_result),
            patch.object(trainer, "train_final", return_value=mock_model),
            patch.object(trainer, "_save_oof_cache"),
        ):
            trainer._run_fixed(pd.DataFrame())

        mock_model.save.assert_called_once()

    def test_run_fixed_calls_oof_cache(self, trainer: MockNFTrainer) -> None:
        """_run_fixed saves OOF cache for ensemble (line 591)."""
        trainer._settings.mock_nf.best_params = {"lr": 0.01}

        best_result = _make_training_result()
        mock_model = MagicMock()

        with (
            patch.object(trainer, "_train_all_splits", return_value=best_result),
            patch.object(trainer, "train_final", return_value=mock_model),
            patch.object(trainer, "_save_oof_cache") as mock_oof,
        ):
            trainer._run_fixed(pd.DataFrame())

        mock_oof.assert_called_once_with(best_result)


# ---------------------------------------------------------------------------
# Tests: run() dispatcher (lines 546-559)
# ---------------------------------------------------------------------------


class TestRunDispatch:
    """Tests for run() auto-dispatch (lines 546-559)."""

    def test_no_best_params_runs_optuna(self, trainer: MockNFTrainer) -> None:
        """Empty best_params dispatches to _run_optuna."""
        trainer._settings.mock_nf.best_params = {}

        with patch.object(trainer, "_run_optuna", return_value=MagicMock()) as mock_opt:
            trainer.run(pd.DataFrame())
            mock_opt.assert_called_once()

    def test_best_params_runs_fixed(self, trainer: MockNFTrainer) -> None:
        """Populated best_params dispatches to _run_fixed."""
        trainer._settings.mock_nf.best_params = {"lr": 0.01}

        with patch.object(trainer, "_run_fixed", return_value=MagicMock()) as mock_fix:
            trainer.run(pd.DataFrame())
            mock_fix.assert_called_once()

    def test_force_hpo_ignores_best_params(self, mock_settings: MagicMock) -> None:
        """force_hpo=True dispatches to _run_optuna even with best_params."""
        mock_settings.mock_nf.best_params = {"lr": 0.01}
        t = MockNFTrainer(mock_settings, force_hpo=True)

        with patch.object(t, "_run_optuna", return_value=MagicMock()) as mock_opt:
            t.run(pd.DataFrame())
            mock_opt.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _save_oof_cache (lines 529-542)
# ---------------------------------------------------------------------------


class TestSaveOofCache:
    """Tests for _save_oof_cache non-fatal behavior (lines 541-542)."""

    def test_oof_cache_failure_is_nonfatal(self, trainer: MockNFTrainer) -> None:
        """OOF cache save failure logs warning, does not raise."""
        best_result = _make_training_result()

        with patch(
            "energy_forecast.training.oof_cache.compute_config_hash",
            side_effect=RuntimeError("disk full"),
        ):
            # Should not raise
            trainer._save_oof_cache(best_result)

    def test_oof_cache_calls_save(self, trainer: MockNFTrainer) -> None:
        """Successful OOF cache save invokes save_oof_cache."""
        best_result = _make_training_result()

        with (
            patch(
                "energy_forecast.training.oof_cache.compute_config_hash",
                return_value="abc123",
            ),
            patch(
                "energy_forecast.training.oof_cache.save_oof_cache",
            ) as mock_save,
        ):
            trainer._save_oof_cache(best_result)

        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _run_optuna full path (lines 601-643)
# ---------------------------------------------------------------------------


class TestRunOptuna:
    """Tests for _run_optuna pipeline."""

    def test_run_optuna_returns_pipeline_result(self, trainer: MockNFTrainer) -> None:
        """_run_optuna returns NFPipelineResult with study and model."""
        mock_study = MagicMock()
        mock_study.best_params = {"lr": 0.005}
        mock_study.best_value = 2.1
        mock_study.best_trial.number = 0
        mock_study.best_trial.user_attrs = {"avg_test_mape": 2.8}

        best_result = _make_training_result()
        mock_model = MagicMock()

        with (
            patch.object(trainer, "optimize", return_value=(mock_study, best_result)),
            patch.object(trainer, "train_final", return_value=mock_model),
            patch.object(trainer, "_save_oof_cache"),
        ):
            result = trainer._run_optuna(pd.DataFrame())

        assert isinstance(result, NFPipelineResult)
        assert result.study is mock_study
        assert result.best_params == {"lr": 0.005}
        assert result.final_model is mock_model


# ---------------------------------------------------------------------------
# Tests: train_final (lines 454-483)
# ---------------------------------------------------------------------------


class TestTrainFinal:
    """Tests for train_final method."""

    def test_train_final_with_val_split(self, trainer: MockNFTrainer) -> None:
        """Large dataset gets train/val split for early stopping."""
        n = 2000
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({"consumption": np.full(n, 1000.0)}, index=idx)

        mock_model = MagicMock()
        with patch.object(trainer, "_create_forecaster", return_value=mock_model):
            result = trainer.train_final(df, {"lr": 0.01})

        assert result is mock_model
        # train was called with val_df (not None)
        call_args = mock_model.train.call_args
        assert call_args[0][1] is not None  # val_df positional arg

    def test_train_final_small_data_no_val(self, trainer: MockNFTrainer) -> None:
        """Small dataset trains without validation split."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({"consumption": np.full(n, 1000.0)}, index=idx)

        # val_size_hours=720, 100 < 720*2 → no val split
        mock_model = MagicMock()
        with patch.object(trainer, "_create_forecaster", return_value=mock_model):
            result = trainer.train_final(df, {"lr": 0.01})

        assert result is mock_model
        call_args = mock_model.train.call_args
        assert call_args[0][1] is None  # val_df is None


# ---------------------------------------------------------------------------
# Tests: _log_cv_metrics / _log_common_meta
# ---------------------------------------------------------------------------


class TestMLflowLogging:
    """Tests for MLflow logging helpers."""

    def test_log_cv_metrics_calls_tracker(self, trainer: MockNFTrainer) -> None:
        """_log_cv_metrics logs aggregate and per-split metrics."""
        trainer._tracker = MagicMock()
        best_result = _make_training_result(n_splits=3)

        trainer._log_cv_metrics(best_result)

        trainer._tracker.log_metrics.assert_called_once()
        assert trainer._tracker.log_split_metrics.call_count == 3

    def test_log_cv_metrics_empty_splits(self, trainer: MockNFTrainer) -> None:
        """_log_cv_metrics handles empty split_results gracefully."""
        trainer._tracker = MagicMock()
        best_result = NFTrainingResult(
            split_results=[],
            avg_val_mape=2.5,
            avg_test_mape=3.0,
            std_val_mape=0.1,
        )

        trainer._log_cv_metrics(best_result)

        trainer._tracker.log_metrics.assert_called_once()
        # std_test_mape should be 0.0 when no splits
        call_kwargs = trainer._tracker.log_metrics.call_args[0][0]
        assert call_kwargs["std_test_mape"] == 0.0

    def test_log_common_meta_includes_covariates(self, trainer: MockNFTrainer) -> None:
        """_log_common_meta logs futr/hist exog lists."""
        trainer._tracker = MagicMock()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        trainer._log_common_meta(df, extra={"mode": "fixed"})

        trainer._tracker.log_training_meta.assert_called_once()
        trainer._tracker.log_config_snapshot.assert_called_once()
        trainer._tracker.log_params.assert_called_once()
        logged_params = trainer._tracker.log_params.call_args[0][0]
        assert logged_params["futr_exog_list"] == "hour_sin,hour_cos"
        assert logged_params["hist_exog_list"] == "temperature_2m"
