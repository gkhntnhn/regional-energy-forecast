"""Unit tests for ConsumptionFeatureEngineer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import ConsumptionConfig
from energy_forecast.features.consumption import ConsumptionFeatureEngineer


@pytest.fixture()
def consumption_df() -> pd.DataFrame:
    """720-row DataFrame with consumption column and DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=720, freq="h")
    return pd.DataFrame(
        {"consumption": 800.0 + rng.random(720) * 400},
        index=idx,
    ).rename_axis("datetime")


@pytest.fixture()
def config() -> dict[str, object]:
    """Consumption feature config from defaults."""
    return ConsumptionConfig().model_dump()


@pytest.fixture()
def result(
    consumption_df: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    """Transformed DataFrame with all consumption features."""
    eng = ConsumptionFeatureEngineer(config=config)
    return eng.transform(consumption_df)


class TestLagFeatures:
    """Tests for consumption lag features."""

    def test_lag_features_created(self, result: pd.DataFrame) -> None:
        """Lag columns for all configured periods exist."""
        expected = [
            "consumption_lag_48",
            "consumption_lag_72",
            "consumption_lag_96",
            "consumption_lag_168",
            "consumption_lag_336",
            "consumption_lag_720",
        ]
        for col in expected:
            assert col in result.columns, f"Missing lag column: {col}"

    def test_lag_values_correct(
        self,
        consumption_df: pd.DataFrame,
        result: pd.DataFrame,
    ) -> None:
        """lag_48 equals consumption.shift(48) on non-NaN rows."""
        expected = consumption_df["consumption"].shift(48)
        actual = result["consumption_lag_48"]
        mask = actual.notna()
        pd.testing.assert_series_equal(
            actual[mask],
            expected[mask],
            check_names=False,
        )

    def test_lag_min_lag_enforced(self, result: pd.DataFrame) -> None:
        """All lag columns have period >= 48."""
        lag_cols = [c for c in result.columns if c.startswith("consumption_lag_")]
        for col in lag_cols:
            period = int(col.split("_")[-1])
            assert period >= 48, f"Lag {col} has period {period} < 48"


class TestWindowFeatures:
    """Tests for rolling window features."""

    def test_window_features_created(self, result: pd.DataFrame) -> None:
        """Rolling window columns for window=24 with all functions exist."""
        expected = [
            "consumption_window_24_mean",
            "consumption_window_24_std",
            "consumption_window_24_min",
            "consumption_window_24_max",
        ]
        for col in expected:
            assert col in result.columns, f"Missing window column: {col}"

    def test_window_leakage_safe(self, result: pd.DataFrame) -> None:
        """First min_lag + window - 1 rows of window features are NaN.

        WindowFeatures with periods=48 shifts data by 48 before rolling.
        For window=24: first 48 + 24 - 1 = 71 rows must be NaN.
        """
        min_lag = 48
        window = 24
        nan_boundary = min_lag + window - 1
        col = "consumption_window_24_mean"
        assert result[col].iloc[:nan_boundary].isna().all(), (
            f"Expected first {nan_boundary} rows to be NaN for {col}"
        )
        assert result[col].iloc[nan_boundary:].notna().any(), (
            f"Expected some valid values after row {nan_boundary}"
        )

    def test_window_all_sizes(self, result: pd.DataFrame) -> None:
        """Window columns exist for each configured window size."""
        windows = [24, 48, 168, 336, 720]
        for w in windows:
            col = f"consumption_window_{w}_mean"
            assert col in result.columns, f"Missing window column: {col}"


class TestExpandingFeatures:
    """Tests for expanding window features."""

    def test_expanding_created(self, result: pd.DataFrame) -> None:
        """Expanding mean and std columns exist."""
        assert "consumption_expanding_mean" in result.columns
        assert "consumption_expanding_std" in result.columns


class TestEwmaFeatures:
    """Tests for EWMA features."""

    def test_ewma_created(self, result: pd.DataFrame) -> None:
        """EWMA columns for all configured spans exist."""
        expected = [
            "consumption_ewma_24",
            "consumption_ewma_48",
            "consumption_ewma_168",
        ]
        for col in expected:
            assert col in result.columns, f"Missing EWMA column: {col}"


class TestMomentumFeatures:
    """Tests for momentum features."""

    def test_momentum_created(self, result: pd.DataFrame) -> None:
        """Momentum and pct_change columns for configured periods exist."""
        expected = [
            "consumption_momentum_24",
            "consumption_momentum_168",
            "consumption_pct_change_24",
            "consumption_pct_change_168",
        ]
        for col in expected:
            assert col in result.columns, f"Missing momentum column: {col}"


class TestQuantileFeatures:
    """Tests for quantile features."""

    def test_quantile_created(self, result: pd.DataFrame) -> None:
        """Quantile columns for all configured quantiles exist."""
        expected = [
            "consumption_q25_168",
            "consumption_q50_168",
            "consumption_q75_168",
        ]
        for col in expected:
            assert col in result.columns, f"Missing quantile column: {col}"


class TestNanAndCount:
    """Tests for NaN behavior and overall feature count."""

    def test_nan_in_early_rows(self, result: pd.DataFrame) -> None:
        """First 48 rows of all lag features are NaN (min_lag=48)."""
        lag_cols = [c for c in result.columns if c.startswith("consumption_lag_")]
        for col in lag_cols:
            assert result[col].iloc[:48].isna().all(), f"Expected first 48 rows to be NaN for {col}"

    def test_feature_count(self, result: pd.DataFrame) -> None:
        """Total feature columns >= 38.

        Breakdown: 6 lags + 20 rolling + 2 expanding + 3 ewma
        + 4 momentum + 3 quantile = 38.
        """
        new_cols = [c for c in result.columns if c != "consumption"]
        assert len(new_cols) >= 38, f"Expected >= 38 feature columns, got {len(new_cols)}"
