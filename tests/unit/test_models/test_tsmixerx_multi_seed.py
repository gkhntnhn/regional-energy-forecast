"""Unit tests for MultiSeedTSMixerxForecaster (R12 Jensen ensemble)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import (
    TSMixerxArchitectureConfig,
    TSMixerxConfig,
    TSMixerxCovariatesConfig,
    TSMixerxTrainingConfig,
)
from energy_forecast.models.base import PREDICTION_COL
from energy_forecast.models.tsmixerx_multi_seed import MultiSeedTSMixerxForecaster

SEEDS = [42, 123, 456, 789, 2026]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def canonical_config() -> TSMixerxConfig:
    """Minimal config used as the 'canonical' architecture across mock seeds."""
    return TSMixerxConfig(
        architecture=TSMixerxArchitectureConfig(
            n_block=4,
            ff_dim=96,
            dropout=0.2,
            input_size=168,
            revin=False,
        ),
        training=TSMixerxTrainingConfig(
            prediction_length=48,
            max_steps=10,
            windows_batch_size=32,
            step_size=12,
            learning_rate=0.001,
            early_stop_patience_steps=-1,
            val_check_steps=5,
            random_seed=42,
            accelerator="cpu",
            num_workers=0,
            scaler_type="standard",
        ),
        covariates=TSMixerxCovariatesConfig(
            futr_exog=["apparent_temperature"],
            hist_exog=["consumption_lag_48"],
        ),
    )


def _make_seed_dir(base: Path, seed: int) -> Path:
    """Create an empty seed_{seed}/ directory with a metadata.json file."""
    seed_dir = base / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": {
            "n_block": 4,
            "ff_dim": 96,
            "dropout": 0.2,
            "input_size": 168,
            "revin": False,
        },
        "training": {
            "input_size": 168,
            "prediction_length": 48,
            "num_workers": 0,
            "scaler_type": "standard",
        },
        "covariates": {
            "futr_exog": ["apparent_temperature"],
            "hist_exog": ["consumption_lag_48"],
        },
        "ckpt_hashes": {},
    }
    (seed_dir / "metadata.json").write_text(json.dumps(metadata))
    return seed_dir


@pytest.fixture
def multi_seed_dir(tmp_path: Path) -> Path:
    """Fake multi-seed directory with 5 empty seed_*/ subdirectories."""
    base = tmp_path / "multi_seed"
    for seed in SEEDS:
        _make_seed_dir(base, seed)
    return base


def _make_mock_forecaster(
    config: TSMixerxConfig,
    prediction_values: np.ndarray[Any, Any] | None = None,
) -> MagicMock:
    """Build a MagicMock that quacks like TSMixerxForecaster."""
    forecaster = MagicMock()
    forecaster._tsmixerx_config = config
    if prediction_values is not None:
        # predict() returns a DataFrame with PREDICTION_COL column
        forecaster.predict.return_value = pd.DataFrame(
            {PREDICTION_COL: prediction_values},
            index=pd.date_range("2025-01-01", periods=len(prediction_values), freq="h"),
        )
    return forecaster


def _patch_inner_from_checkpoint(
    forecasters_by_name: dict[str, MagicMock],
) -> Any:
    """Patch TSMixerxForecaster.from_checkpoint to return MagicMock per seed_dir.name."""

    def side_effect(path: Path | str) -> MagicMock:
        path = Path(path)
        name = path.name
        if name in forecasters_by_name:
            return forecasters_by_name[name]
        raise FileNotFoundError(f"No mock for {name}")

    return patch(
        "energy_forecast.models.tsmixerx_multi_seed.TSMixerxForecaster.from_checkpoint",
        side_effect=side_effect,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestFromCheckpointDiscovery:
    """Directory discovery + graceful degrade during load."""

    def test_discovers_and_loads_all_seeds(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        mocks = {
            f"seed_{s}": _make_mock_forecaster(canonical_config) for s in SEEDS
        }
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        assert forecaster.n_seeds_loaded == 5
        assert forecaster.n_seeds_requested == 5
        assert forecaster.is_fitted
        assert forecaster.is_degraded is False
        assert len(forecaster._seeds_loaded) == 5

    def test_raises_on_no_seed_subdirs(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No seed_"):
            MultiSeedTSMixerxForecaster.from_checkpoint(empty)

    def test_graceful_degrade_on_partial_failure(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        """3/5 seeds load, 2 fail — ensemble still initialises."""
        mocks = {
            "seed_42": _make_mock_forecaster(canonical_config),
            "seed_123": _make_mock_forecaster(canonical_config),
            "seed_456": _make_mock_forecaster(canonical_config),
        }
        # seed_789 and seed_2026 will raise FileNotFoundError from our side_effect
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        assert forecaster.n_seeds_loaded == 3
        assert forecaster.n_seeds_requested == 5
        assert forecaster.is_degraded is True
        assert set(forecaster._seeds_loaded) == {"seed_42", "seed_123", "seed_456"}

    def test_raises_when_all_seeds_fail(self, multi_seed_dir: Path) -> None:
        with (
            _patch_inner_from_checkpoint({}),
            pytest.raises(RuntimeError, match=r"All .* seeds failed to load"),
        ):
            MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

    def test_detects_config_mismatch_across_seeds(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        """input_size divergence must cause a load-time RuntimeError."""
        mismatched_arch = TSMixerxArchitectureConfig(
            n_block=4,
            ff_dim=96,
            dropout=0.2,
            input_size=336,  # DIFFERENT
            revin=False,
        )
        mismatched = canonical_config.model_copy(update={"architecture": mismatched_arch})

        mocks = {
            "seed_42": _make_mock_forecaster(canonical_config),
            "seed_123": _make_mock_forecaster(canonical_config),
            "seed_456": _make_mock_forecaster(mismatched),  # mismatch
            "seed_789": _make_mock_forecaster(canonical_config),
            "seed_2026": _make_mock_forecaster(canonical_config),
        }
        # Degrade: 1 config error surfaces as a per-seed failure (not load-all abort),
        # since we catch exceptions; remaining 4 still load successfully.
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        assert forecaster.n_seeds_loaded == 4
        assert "seed_456" not in forecaster._seeds_loaded
        assert forecaster.is_degraded is True

    def test_reads_optional_top_level_metadata(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        top_meta = {"ensemble_test_mape": 1.6494, "seeds": SEEDS}
        (multi_seed_dir / "metadata.json").write_text(json.dumps(top_meta))

        mocks = {f"seed_{s}": _make_mock_forecaster(canonical_config) for s in SEEDS}
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        assert forecaster._top_metadata["ensemble_test_mape"] == pytest.approx(1.6494)


class TestPredict:
    """Jensen ensemble prediction correctness + failure modes."""

    def test_predict_jensen_math(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        """Predictions from 3 fake seeds must be averaged element-wise."""
        preds_42 = np.array([100.0, 200.0, 300.0])
        preds_123 = np.array([110.0, 210.0, 310.0])
        preds_456 = np.array([120.0, 220.0, 320.0])
        mocks = {
            "seed_42": _make_mock_forecaster(canonical_config, preds_42),
            "seed_123": _make_mock_forecaster(canonical_config, preds_123),
            "seed_456": _make_mock_forecaster(canonical_config, preds_456),
            "seed_789": _make_mock_forecaster(canonical_config, preds_42),
            "seed_2026": _make_mock_forecaster(canonical_config, preds_42),
        }
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        # Override loaded set to only the 3 we care about for a clean math check
        forecaster._forecasters = [mocks[s] for s in ("seed_42", "seed_123", "seed_456")]
        forecaster._seeds_loaded = ["seed_42", "seed_123", "seed_456"]

        features = pd.DataFrame(
            {"dummy": [0, 0, 0]},
            index=pd.date_range("2025-01-01", periods=3, freq="h"),
        )
        result = forecaster.predict(features)

        expected = np.mean([preds_42, preds_123, preds_456], axis=0)
        np.testing.assert_allclose(result[PREDICTION_COL].to_numpy(), expected)

    def test_predict_raises_when_not_fitted(
        self,
        canonical_config: TSMixerxConfig,
    ) -> None:
        forecaster = MultiSeedTSMixerxForecaster(canonical_config)
        features = pd.DataFrame(
            {"dummy": [0]}, index=pd.date_range("2025-01-01", periods=1, freq="h")
        )
        with pytest.raises(RuntimeError, match="no loaded seeds"):
            forecaster.predict(features)

    def test_predict_graceful_degrade_at_runtime(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        """A runtime predict failure on one seed should skip but continue."""
        good = np.array([1000.0, 1100.0, 1200.0])
        good_forecaster = _make_mock_forecaster(canonical_config, good)
        bad_forecaster = _make_mock_forecaster(canonical_config)
        bad_forecaster.predict.side_effect = RuntimeError("GPU OOM")

        mocks: dict[str, MagicMock] = {
            "seed_42": good_forecaster,
            "seed_123": bad_forecaster,
            "seed_456": good_forecaster,
            "seed_789": good_forecaster,
            "seed_2026": good_forecaster,
        }
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        features = pd.DataFrame(
            {"dummy": [0, 0, 0]},
            index=pd.date_range("2025-01-01", periods=3, freq="h"),
        )
        result = forecaster.predict(features)
        # 4 seeds succeeded -> mean equals `good` (since 4 identical mocks)
        np.testing.assert_allclose(result[PREDICTION_COL].to_numpy(), good)

    def test_predict_raises_when_all_seeds_fail_at_runtime(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        bad = _make_mock_forecaster(canonical_config)
        bad.predict.side_effect = RuntimeError("catastrophic")
        mocks = {f"seed_{s}": bad for s in SEEDS}
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        features = pd.DataFrame(
            {"dummy": [0]},
            index=pd.date_range("2025-01-01", periods=1, freq="h"),
        )
        with pytest.raises(RuntimeError, match="All per-seed predictions failed"):
            forecaster.predict(features)


class TestPredictTimeTelemetry:
    """get_seed_info reflects actual predict() outcome (audit K1 fix)."""

    def test_last_request_counters_initially_none(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        mocks = {f"seed_{s}": _make_mock_forecaster(canonical_config) for s in SEEDS}
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        info = forecaster.get_seed_info()
        assert info["last_request_n_succeeded"] is None
        assert info["last_request_failed_seeds"] == []

    def test_last_request_reflects_partial_failure(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        """5 seeds loaded, 2 fail at predict time — counters tell the truth."""
        good = np.array([1000.0, 1100.0, 1200.0])
        good_fc = _make_mock_forecaster(canonical_config, good)
        bad_fc = _make_mock_forecaster(canonical_config)
        bad_fc.predict.side_effect = RuntimeError("GPU OOM")

        mocks: dict[str, MagicMock] = {
            "seed_42": good_fc,
            "seed_123": bad_fc,
            "seed_456": good_fc,
            "seed_789": bad_fc,
            "seed_2026": good_fc,
        }
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        features = pd.DataFrame(
            {"dummy": [0, 0, 0]},
            index=pd.date_range("2025-01-01", periods=3, freq="h"),
        )
        forecaster.predict(features)

        info = forecaster.get_seed_info()
        assert info["n_loaded"] == 5  # load-time count unchanged
        assert info["last_request_n_succeeded"] == 3  # 3/5 succeeded
        assert set(info["last_request_failed_seeds"]) == {"seed_123", "seed_789"}


class TestObservability:
    """get_seed_info + get_per_seed_predictions."""

    def test_get_seed_info_full_load(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        mocks = {f"seed_{s}": _make_mock_forecaster(canonical_config) for s in SEEDS}
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        info = forecaster.get_seed_info()
        assert info["ensemble_type"] == "multi_seed_jensen"
        assert info["n_requested"] == 5
        assert info["n_loaded"] == 5
        assert info["is_degraded"] is False
        assert len(info["seeds_loaded"]) == 5

    def test_get_per_seed_predictions_returns_mapped_dict(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        preds = {
            "seed_42": np.array([1.0, 2.0]),
            "seed_123": np.array([3.0, 4.0]),
            "seed_456": np.array([5.0, 6.0]),
            "seed_789": np.array([7.0, 8.0]),
            "seed_2026": np.array([9.0, 10.0]),
        }
        mocks = {s: _make_mock_forecaster(canonical_config, v) for s, v in preds.items()}
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        features = pd.DataFrame(
            {"dummy": [0, 0]}, index=pd.date_range("2025-01-01", periods=2, freq="h")
        )
        result = forecaster.get_per_seed_predictions(features)
        assert set(result.keys()) == set(preds.keys())
        for seed_name, expected in preds.items():
            np.testing.assert_allclose(result[seed_name], expected)


class TestUnsupportedOperations:
    """train / save / load must raise NotImplementedError."""

    def test_train_not_supported(self, canonical_config: TSMixerxConfig) -> None:
        forecaster = MultiSeedTSMixerxForecaster(canonical_config)
        df = pd.DataFrame({"y": [1.0]}, index=pd.date_range("2025-01-01", periods=1, freq="h"))
        with pytest.raises(NotImplementedError, match="MultiSeedTSMixerxTrainer"):
            forecaster.train(df)

    def test_save_not_supported(self, canonical_config: TSMixerxConfig, tmp_path: Path) -> None:
        forecaster = MultiSeedTSMixerxForecaster(canonical_config)
        with pytest.raises(NotImplementedError, match="Per-seed checkpoints are saved"):
            forecaster.save(tmp_path)

    def test_load_not_supported(self, canonical_config: TSMixerxConfig, tmp_path: Path) -> None:
        forecaster = MultiSeedTSMixerxForecaster(canonical_config)
        with pytest.raises(NotImplementedError, match="from_checkpoint"):
            forecaster.load(tmp_path)


class TestProxyConfig:
    """Ensure _tsmixerx_config proxies the canonical architecture.

    EnsembleForecaster._get_base_predictions reads
    ``self._tsmixerx_model._tsmixerx_config.architecture.input_size`` — this
    path must resolve for MultiSeedTSMixerxForecaster too.
    """

    def test_input_size_proxy_matches_canonical(
        self,
        multi_seed_dir: Path,
        canonical_config: TSMixerxConfig,
    ) -> None:
        mocks = {f"seed_{s}": _make_mock_forecaster(canonical_config) for s in SEEDS}
        with _patch_inner_from_checkpoint(mocks):
            forecaster = MultiSeedTSMixerxForecaster.from_checkpoint(multi_seed_dir)

        assert forecaster._tsmixerx_config.architecture.input_size == 168
        assert forecaster._tsmixerx_config.architecture.n_block == 4
        assert forecaster._tsmixerx_config.architecture.ff_dim == 96
