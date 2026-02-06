"""Unit tests for custom sklearn-compatible transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features.custom import (
    DegreeDayFeatures,
    EwmaFeatures,
    MomentumFeatures,
    QuantileFeatures,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a 720-row DataFrame with DatetimeIndex and consumption column."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=720, freq="h", name="datetime")
    return pd.DataFrame(
        {"consumption": rng.uniform(800, 1200, size=720)},
        index=index,
    )


@pytest.fixture()
def temperature_df() -> pd.DataFrame:
    """Create a 720-row DataFrame with DatetimeIndex and temperature_2m column."""
    index = pd.date_range("2024-01-01", periods=720, freq="h", name="datetime")
    return pd.DataFrame(
        {"temperature_2m": np.full(720, 20.0)},
        index=index,
    )


class TestEwmaFeatures:
    """Tests for EwmaFeatures transformer."""

    def test_ewma_shift_applied(self, sample_df: pd.DataFrame) -> None:
        """EWMA result is shifted by periods; first `periods` values are NaN."""
        transformer = EwmaFeatures(variables=["consumption"], spans=[12], periods=48)
        result = transformer.fit_transform(sample_df)
        col = "consumption_ewma_12"
        assert col in result.columns
        assert result[col].iloc[:48].isna().all()
        assert result[col].iloc[48:].notna().any()

    def test_ewma_spans_from_config(self, sample_df: pd.DataFrame) -> None:
        """Correct number of EWMA columns: len(variables) * len(spans)."""
        spans = [6, 12, 24]
        transformer = EwmaFeatures(variables=["consumption"], spans=spans, periods=48)
        result = transformer.fit_transform(sample_df)
        ewma_cols = [c for c in result.columns if "_ewma_" in c]
        assert len(ewma_cols) == 1 * len(spans)

    def test_ewma_column_naming(self, sample_df: pd.DataFrame) -> None:
        """Columns are named {var}_ewma_{span}."""
        spans = [6, 24]
        transformer = EwmaFeatures(variables=["consumption"], spans=spans, periods=48)
        result = transformer.fit_transform(sample_df)
        assert "consumption_ewma_6" in result.columns
        assert "consumption_ewma_24" in result.columns


class TestMomentumFeatures:
    """Tests for MomentumFeatures transformer."""

    def test_momentum_calculation(self, sample_df: pd.DataFrame) -> None:
        """momentum_24 equals shift(48) - shift(72)."""
        transformer = MomentumFeatures(
            variables=["consumption"],
            min_lag=48,
            momentum_periods=[24],
        )
        result = transformer.fit_transform(sample_df)
        col = "consumption_momentum_24"
        assert col in result.columns

        series = sample_df["consumption"]
        expected = series.shift(48) - series.shift(48 + 24)
        # Compare where both are not NaN
        mask = expected.notna() & result[col].notna()
        np.testing.assert_allclose(
            result.loc[mask, col].to_numpy(),
            expected[mask].to_numpy(),
            rtol=1e-10,
        )

    def test_pct_change_calculation(self, sample_df: pd.DataFrame) -> None:
        """pct_change = momentum / older * 100 (with epsilon for stability)."""
        transformer = MomentumFeatures(
            variables=["consumption"],
            min_lag=48,
            momentum_periods=[24],
        )
        result = transformer.fit_transform(sample_df)
        pct_col = "consumption_pct_change_24"
        mom_col = "consumption_momentum_24"
        assert pct_col in result.columns

        series = sample_df["consumption"]
        older = series.shift(48 + 24)
        expected_pct = result[mom_col] / (older + 1e-9) * 100
        mask = expected_pct.notna() & result[pct_col].notna()
        np.testing.assert_allclose(
            result.loc[mask, pct_col].to_numpy(),
            expected_pct[mask].to_numpy(),
            rtol=1e-10,
        )


class TestQuantileFeatures:
    """Tests for QuantileFeatures transformer."""

    def test_quantile_shift_applied(self, sample_df: pd.DataFrame) -> None:
        """First periods + window - 1 values are NaN due to shift + rolling."""
        transformer = QuantileFeatures(
            variables=["consumption"],
            quantiles=[0.50],
            window=168,
            periods=48,
        )
        result = transformer.fit_transform(sample_df)
        col = "consumption_q50_168"
        assert col in result.columns
        # shift(48) makes first 48 NaN, then rolling(168) needs 168 valid values
        # so first 48 + 168 - 1 = 215 values are NaN
        nan_count = result[col].iloc[: 48 + 168 - 1].isna().sum()
        assert nan_count == 48 + 168 - 1

    def test_quantile_ordering(self, sample_df: pd.DataFrame) -> None:
        """q25 <= q50 <= q75 where values are not NaN."""
        transformer = QuantileFeatures(
            variables=["consumption"],
            quantiles=[0.25, 0.50, 0.75],
            window=168,
            periods=48,
        )
        result = transformer.fit_transform(sample_df)
        q25 = result["consumption_q25_168"]
        q50 = result["consumption_q50_168"]
        q75 = result["consumption_q75_168"]
        mask = q25.notna() & q50.notna() & q75.notna()
        assert (q25[mask] <= q50[mask]).all()
        assert (q50[mask] <= q75[mask]).all()


class TestDegreeDayFeatures:
    """Tests for DegreeDayFeatures transformer."""

    def test_degree_day_hdd(self) -> None:
        """T=10, base=18 produces HDD=8."""
        index = pd.date_range("2024-01-01", periods=24, freq="h", name="datetime")
        df = pd.DataFrame({"temperature_2m": np.full(24, 10.0)}, index=index)
        transformer = DegreeDayFeatures(
            temp_variable="temperature_2m", hdd_base=18.0, cdd_base=24.0
        )
        result = transformer.fit_transform(df)
        np.testing.assert_allclose(result["wth_hdd"].to_numpy(), 8.0)

    def test_degree_day_cdd(self) -> None:
        """T=30, base=24 produces CDD=6."""
        index = pd.date_range("2024-01-01", periods=24, freq="h", name="datetime")
        df = pd.DataFrame({"temperature_2m": np.full(24, 30.0)}, index=index)
        transformer = DegreeDayFeatures(
            temp_variable="temperature_2m", hdd_base=18.0, cdd_base=24.0
        )
        result = transformer.fit_transform(df)
        np.testing.assert_allclose(result["wth_cdd"].to_numpy(), 6.0)

    def test_degree_day_zero(self, temperature_df: pd.DataFrame) -> None:
        """T=20 is between bases: HDD=0 and CDD=0."""
        transformer = DegreeDayFeatures(
            temp_variable="temperature_2m", hdd_base=18.0, cdd_base=24.0
        )
        result = transformer.fit_transform(temperature_df)
        np.testing.assert_allclose(result["wth_hdd"].to_numpy(), 0.0)
        np.testing.assert_allclose(result["wth_cdd"].to_numpy(), 0.0)
