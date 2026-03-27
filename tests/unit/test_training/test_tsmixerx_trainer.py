"""Unit tests for TSMixerxTrainer fixed mode dispatch."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast.config import (
    CrossValidationConfig,
    HyperparameterConfig,
    ModelSearchConfig,
    TSMixerxArchitectureConfig,
    TSMixerxConfig,
    TSMixerxCovariatesConfig,
    TSMixerxTrainingConfig,
)


def _make_tsmixerx_config(*, best_params: dict[str, Any] | None = None) -> TSMixerxConfig:
    """Create minimal TSMixerx config for fast tests."""
    return TSMixerxConfig(
        architecture=TSMixerxArchitectureConfig(n_block=1, ff_dim=16),
        training=TSMixerxTrainingConfig(
            prediction_length=24,
            max_steps=10,
            windows_batch_size=32,
            accelerator="cpu",
            num_workers=0,
        ),
        covariates=TSMixerxCovariatesConfig(
            futr_exog=["hour_sin"],
            hist_exog=[],
        ),
        best_params=best_params or {},
    )


class TestFixedModeDispatch:
    """Tests for run() auto-dispatch between fixed and Optuna modes."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings with real HP config for TSMixerx."""
        settings = MagicMock()
        settings.tsmixerx = _make_tsmixerx_config()
        settings.hyperparameters = HyperparameterConfig(
            tsmixerx=ModelSearchConfig(n_trials=1, search_space={}),
            cross_validation=CrossValidationConfig(n_splits=2, val_months=1, test_months=1),
            target_col="consumption",
        )
        settings.paths.models_dir = "models"
        return settings

    def test_empty_best_params_runs_optuna(self, mock_settings: MagicMock) -> None:
        """Empty best_params → _run_optuna."""
        from energy_forecast.training.tsmixerx_trainer import TSMixerxTrainer

        mock_settings.tsmixerx = _make_tsmixerx_config(best_params={})
        trainer = TSMixerxTrainer(mock_settings)
        with patch.object(trainer, "_run_optuna") as mock_optuna:
            mock_optuna.return_value = MagicMock()
            trainer.run(pd.DataFrame())
            mock_optuna.assert_called_once()

    def test_filled_best_params_runs_fixed(self, mock_settings: MagicMock) -> None:
        """Filled best_params → _run_fixed."""
        from energy_forecast.training.tsmixerx_trainer import TSMixerxTrainer

        mock_settings.tsmixerx = _make_tsmixerx_config(
            best_params={"n_block": 2, "ff_dim": 64, "learning_rate": 0.005},
        )
        trainer = TSMixerxTrainer(mock_settings)
        with patch.object(trainer, "_run_fixed") as mock_fixed:
            mock_fixed.return_value = MagicMock()
            trainer.run(pd.DataFrame())
            mock_fixed.assert_called_once()

    def test_force_hpo_ignores_best_params(self, mock_settings: MagicMock) -> None:
        """--force-hpo → _run_optuna even when best_params exists."""
        from energy_forecast.training.tsmixerx_trainer import TSMixerxTrainer

        mock_settings.tsmixerx = _make_tsmixerx_config(
            best_params={"n_block": 2, "learning_rate": 0.005},
        )
        trainer = TSMixerxTrainer(mock_settings, force_hpo=True)
        with patch.object(trainer, "_run_optuna") as mock_optuna:
            mock_optuna.return_value = MagicMock()
            trainer.run(pd.DataFrame())
            mock_optuna.assert_called_once()

    def test_force_hpo_default_false(self, mock_settings: MagicMock) -> None:
        """force_hpo defaults to False."""
        from energy_forecast.training.tsmixerx_trainer import TSMixerxTrainer

        trainer = TSMixerxTrainer(mock_settings)
        assert trainer._force_hpo is False
