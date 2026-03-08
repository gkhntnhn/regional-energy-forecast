"""Tests for dynamic Optuna search space parameter suggestion."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from energy_forecast.config import SearchParamConfig
from energy_forecast.training.search import suggest_params


@pytest.fixture
def mock_trial() -> MagicMock:
    """Create a mock Optuna trial with preset return values."""
    trial = MagicMock()
    trial.suggest_int.return_value = 5
    trial.suggest_float.return_value = 0.05
    trial.suggest_categorical.return_value = "RMSE"
    return trial


class TestSuggestInt:
    """Tests for integer parameter suggestion."""

    def test_suggest_int(self, mock_trial: MagicMock) -> None:
        """Int config calls trial.suggest_int with correct args."""
        space: dict[str, SearchParamConfig] = {
            "depth": SearchParamConfig(type="int", low=4, high=7),
        }
        result = suggest_params(mock_trial, space)

        mock_trial.suggest_int.assert_called_once_with(
            "depth",
            4,
            7,
            step=1,
            log=False,
        )
        assert result == {"depth": 5}


class TestSuggestFloat:
    """Tests for float parameter suggestion."""

    def test_suggest_float(self, mock_trial: MagicMock) -> None:
        """Float config calls trial.suggest_float with correct args."""
        space: dict[str, SearchParamConfig] = {
            "learning_rate": SearchParamConfig(type="float", low=0.01, high=0.1),
        }
        result = suggest_params(mock_trial, space)

        mock_trial.suggest_float.assert_called_once_with(
            name="learning_rate",
            low=0.01,
            high=0.1,
            log=False,
        )
        assert result == {"learning_rate": 0.05}

    def test_suggest_float_log(self, mock_trial: MagicMock) -> None:
        """Float config with log=True passes log flag."""
        space: dict[str, SearchParamConfig] = {
            "lr": SearchParamConfig(type="float", low=0.001, high=0.1, log=True),
        }
        suggest_params(mock_trial, space)

        mock_trial.suggest_float.assert_called_once_with(
            name="lr",
            low=0.001,
            high=0.1,
            log=True,
        )

    def test_suggest_float_step(self, mock_trial: MagicMock) -> None:
        """Float config with step passes step kwarg."""
        space: dict[str, SearchParamConfig] = {
            "subsample": SearchParamConfig(
                type="float",
                low=0.5,
                high=1.0,
                step=0.01,
            ),
        }
        suggest_params(mock_trial, space)

        mock_trial.suggest_float.assert_called_once_with(
            name="subsample",
            low=0.5,
            high=1.0,
            log=False,
            step=0.01,
        )


class TestSuggestCategorical:
    """Tests for categorical parameter suggestion."""

    def test_suggest_categorical(self, mock_trial: MagicMock) -> None:
        """Categorical config calls trial.suggest_categorical with choices."""
        choices: list[Any] = ["RMSE", "MAE"]
        space: dict[str, SearchParamConfig] = {
            "loss_function": SearchParamConfig(
                type="categorical",
                choices=choices,
            ),
        }
        result = suggest_params(mock_trial, space)

        mock_trial.suggest_categorical.assert_called_once_with(
            "loss_function",
            ["RMSE", "MAE"],
        )
        assert result == {"loss_function": "RMSE"}


class TestEdgeCases:
    """Tests for edge cases and mixed parameter types."""

    def test_empty_search_space(self, mock_trial: MagicMock) -> None:
        """Empty search space returns empty params dict."""
        result = suggest_params(mock_trial, {})

        assert result == {}
        mock_trial.suggest_int.assert_not_called()
        mock_trial.suggest_float.assert_not_called()
        mock_trial.suggest_categorical.assert_not_called()

    def test_mixed_types(self, mock_trial: MagicMock) -> None:
        """Mixed int + float + categorical calls all three suggest methods."""
        choices: list[Any] = ["RMSE", "MAE"]
        space: dict[str, SearchParamConfig] = {
            "depth": SearchParamConfig(type="int", low=4, high=7),
            "learning_rate": SearchParamConfig(type="float", low=0.01, high=0.1),
            "loss_function": SearchParamConfig(type="categorical", choices=choices),
        }
        result = suggest_params(mock_trial, space)

        assert len(result) == 3
        mock_trial.suggest_int.assert_called_once()
        mock_trial.suggest_float.assert_called_once()
        mock_trial.suggest_categorical.assert_called_once()
        assert result["depth"] == 5
        assert result["learning_rate"] == 0.05
        assert result["loss_function"] == "RMSE"
