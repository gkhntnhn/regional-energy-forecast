"""Unit tests for MultiSeedTSMixerxTrainer (R12 production deploy path)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.training.metrics import MetricsResult
from energy_forecast.training.multi_seed_trainer import (
    DEFAULT_SEEDS,
    MultiSeedResult,
    MultiSeedTSMixerxTrainer,
)
from energy_forecast.training.nf_base_trainer import NFTrainingResult
from energy_forecast.training.results import SplitResult


def _make_metrics(mape: float = 1.7) -> MetricsResult:
    return MetricsResult(mape=mape, mae=10.0, rmse=15.0, r2=0.9,
                         smape=mape, wmape=mape, mbe=0.0)


def _make_split(
    idx: int,
    val_pred: "np.ndarray[tuple[int, ...], np.dtype[np.float64]]",
    val_actual: "np.ndarray[tuple[int, ...], np.dtype[np.float64]]",
    test_pred: "np.ndarray[tuple[int, ...], np.dtype[np.float64]]",
    test_actual: "np.ndarray[tuple[int, ...], np.dtype[np.float64]]",
) -> SplitResult:
    return SplitResult(
        split_idx=idx,
        train_metrics=_make_metrics(1.5),
        val_metrics=_make_metrics(1.7),
        test_metrics=_make_metrics(1.8),
        val_month=f"2025-0{idx+1}",
        test_month=f"2025-0{idx+2}",
        val_predictions=val_pred,
        val_actuals=val_actual,
        test_predictions=test_pred,
        test_actuals=test_actual,
    )


def _make_settings(tmp_path: Path) -> MagicMock:
    settings = MagicMock()
    settings.paths.models_dir = str(tmp_path)
    settings.hyperparameters.target_col = "consumption"
    return settings


class TestModuleSurface:
    """Module-level constants and dataclass shape."""

    def test_default_seeds_5_unique_ints(self) -> None:
        assert len(DEFAULT_SEEDS) == 5
        assert len(set(DEFAULT_SEEDS)) == 5
        assert all(isinstance(s, int) for s in DEFAULT_SEEDS)
        assert 42 in DEFAULT_SEEDS

    def test_multi_seed_result_frozen(self, tmp_path: Path) -> None:
        result = MultiSeedResult(
            seeds=[1, 2],
            seed_val_mapes=[1.7, 1.8],
            seed_test_mapes=[1.75, 1.85],
            ensemble_val_mape=1.65,
            ensemble_test_mape=1.70,
            naive_avg_val_mape=1.75,
            naive_avg_test_mape=1.80,
            seed_models_dir=tmp_path,
            training_time_seconds=42.0,
        )
        with pytest.raises((AttributeError, Exception)):
            result.seeds = [99]  # type: ignore[misc]
        assert result.seed_predictions == {}


class TestConstructor:
    """MultiSeedTSMixerxTrainer initialization."""

    def test_defaults_use_default_seeds(self, tmp_path: Path) -> None:
        trainer = MultiSeedTSMixerxTrainer(_make_settings(tmp_path))
        assert trainer._seeds == DEFAULT_SEEDS
        assert trainer._deterministic is True
        assert trainer._target_col == "consumption"

    def test_custom_seeds(self, tmp_path: Path) -> None:
        trainer = MultiSeedTSMixerxTrainer(
            _make_settings(tmp_path), seeds=[7, 13, 19]
        )
        assert trainer._seeds == [7, 13, 19]

    def test_deterministic_off(self, tmp_path: Path) -> None:
        trainer = MultiSeedTSMixerxTrainer(
            _make_settings(tmp_path), deterministic=False
        )
        assert trainer._deterministic is False


class TestEnableDeterminism:
    """_enable_determinism torch wiring."""

    def test_off_skips_torch_call(self, tmp_path: Path) -> None:
        trainer = MultiSeedTSMixerxTrainer(
            _make_settings(tmp_path), deterministic=False
        )
        with patch("torch.use_deterministic_algorithms") as mock_call:
            trainer._enable_determinism()
        mock_call.assert_not_called()

    def test_on_calls_torch(self, tmp_path: Path) -> None:
        trainer = MultiSeedTSMixerxTrainer(
            _make_settings(tmp_path), deterministic=True
        )
        with patch("torch.use_deterministic_algorithms") as mock_call:
            trainer._enable_determinism()
        mock_call.assert_called_once_with(True, warn_only=True)

    def test_on_sets_cublas_workspace(self, tmp_path: Path) -> None:
        import os
        trainer = MultiSeedTSMixerxTrainer(
            _make_settings(tmp_path), deterministic=True
        )
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        with patch("torch.use_deterministic_algorithms"):
            trainer._enable_determinism()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


class TestRun:
    """End-to-end run() with mocked TSMixerxTrainer."""

    @pytest.fixture
    def fake_cv_result(self) -> NFTrainingResult:
        rng = np.random.default_rng(0)
        s0 = _make_split(
            0,
            val_pred=rng.normal(1000, 50, 24).astype(np.float64),
            val_actual=rng.normal(1000, 50, 24).astype(np.float64),
            test_pred=rng.normal(1000, 50, 24).astype(np.float64),
            test_actual=rng.normal(1000, 50, 24).astype(np.float64),
        )
        return NFTrainingResult(
            split_results=[s0],
            avg_val_mape=1.7,
            avg_test_mape=1.75,
            std_val_mape=0.05,
        )

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=200, freq="h")
        return pd.DataFrame({"consumption": np.arange(200, dtype=float)}, index=idx)

    def test_run_two_seeds_creates_seed_dirs(
        self, tmp_path: Path, fake_cv_result: NFTrainingResult, df: pd.DataFrame
    ) -> None:
        settings = _make_settings(tmp_path)
        trainer = MultiSeedTSMixerxTrainer(
            settings, seeds=[42, 123], deterministic=False
        )

        mock_inner = MagicMock()
        mock_inner._train_all_splits.return_value = fake_cv_result
        mock_final = MagicMock()
        mock_inner.train_final.return_value = mock_final

        with patch(
            "energy_forecast.training.multi_seed_trainer.TSMixerxTrainer",
            return_value=mock_inner,
        ):
            result = trainer.run(df, best_params={"lr": 0.001})

        assert isinstance(result, MultiSeedResult)
        assert result.seeds == [42, 123]
        assert len(result.seed_val_mapes) == 2
        assert len(result.seed_test_mapes) == 2
        # seed dirs created
        assert (tmp_path / "tsmixerx_multi_seed" / "seed_42").exists()
        assert (tmp_path / "tsmixerx_multi_seed" / "seed_123").exists()
        # final_model.save called per seed
        assert mock_final.save.call_count == 2

    def test_run_overrides_seed_in_params(
        self, tmp_path: Path, fake_cv_result: NFTrainingResult, df: pd.DataFrame
    ) -> None:
        settings = _make_settings(tmp_path)
        trainer = MultiSeedTSMixerxTrainer(
            settings, seeds=[7, 13], deterministic=False
        )

        mock_inner = MagicMock()
        mock_inner._train_all_splits.return_value = fake_cv_result
        mock_inner.train_final.return_value = MagicMock()

        with patch(
            "energy_forecast.training.multi_seed_trainer.TSMixerxTrainer",
            return_value=mock_inner,
        ):
            trainer.run(df, best_params={"random_seed": 999, "lr": 0.001})

        call_args_list = mock_inner._train_all_splits.call_args_list
        seeds_passed = [c.args[1]["random_seed"] for c in call_args_list]
        assert seeds_passed == [7, 13]

    def test_run_ensemble_metrics_differ_from_naive(
        self, tmp_path: Path, df: pd.DataFrame
    ) -> None:
        rng = np.random.default_rng(42)
        actual = rng.normal(1000, 100, 24).astype(np.float64)

        def make_cv(pred_offset: float) -> NFTrainingResult:
            return NFTrainingResult(
                split_results=[_make_split(
                    0,
                    val_pred=actual + pred_offset,
                    val_actual=actual,
                    test_pred=actual + pred_offset,
                    test_actual=actual,
                )],
                avg_val_mape=abs(pred_offset),
                avg_test_mape=abs(pred_offset),
                std_val_mape=0.0,
            )

        settings = _make_settings(tmp_path)
        trainer = MultiSeedTSMixerxTrainer(
            settings, seeds=[1, 2], deterministic=False
        )

        mock_a, mock_b = MagicMock(), MagicMock()
        mock_a._train_all_splits.return_value = make_cv(50.0)
        mock_b._train_all_splits.return_value = make_cv(-50.0)
        mock_a.train_final.return_value = MagicMock()
        mock_b.train_final.return_value = MagicMock()

        with patch(
            "energy_forecast.training.multi_seed_trainer.TSMixerxTrainer",
            side_effect=[mock_a, mock_b],
        ):
            result = trainer.run(df, best_params={})

        # Naive avg of |50|, |-50| = 50; ensemble (avg pred) MAPE near 0 (Jensen)
        assert result.naive_avg_val_mape == pytest.approx(50.0, rel=0.5)
        assert result.ensemble_val_mape < result.naive_avg_val_mape
        assert result.training_time_seconds >= 0.0

    def test_run_populates_seed_predictions(
        self, tmp_path: Path, fake_cv_result: NFTrainingResult, df: pd.DataFrame
    ) -> None:
        settings = _make_settings(tmp_path)
        trainer = MultiSeedTSMixerxTrainer(
            settings, seeds=[42], deterministic=False
        )

        mock_inner = MagicMock()
        mock_inner._train_all_splits.return_value = fake_cv_result
        mock_inner.train_final.return_value = MagicMock()

        with patch(
            "energy_forecast.training.multi_seed_trainer.TSMixerxTrainer",
            return_value=mock_inner,
        ):
            result = trainer.run(df, best_params={})

        assert 42 in result.seed_predictions
        assert "val_pred" in result.seed_predictions[42]
        assert "val_actual" in result.seed_predictions[42]
        assert "test_pred" in result.seed_predictions[42]
        assert "test_actual" in result.seed_predictions[42]
