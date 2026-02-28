"""Unit tests for EpiasFeatureEngineer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config.settings import EpiasConfig, GenerationConfig
from energy_forecast.features.epias import EpiasFeatureEngineer

EPIAS_VARIABLES: list[str] = [
    "FDPP",
    "Real_Time_Consumption",
    "DAM_Purchase",
    "Bilateral_Agreement_Purchase",
    "Load_Forecast",
]

GENERATION_VARIABLES: list[str] = [
    "gen_natural_gas",
    "gen_wind",
    "gen_sun",
    "gen_total",
    "gen_lignite",
    "gen_dammed_hydro",
    "gen_import_coal",
    "gen_river",
]


@pytest.fixture()
def epias_config() -> dict[str, Any]:
    """Default EPIAS feature config dict."""
    return EpiasConfig().model_dump()


@pytest.fixture()
def engineer(epias_config: dict[str, Any]) -> EpiasFeatureEngineer:
    """EpiasFeatureEngineer with default config."""
    return EpiasFeatureEngineer(epias_config)


@pytest.fixture()
def epias_df() -> pd.DataFrame:
    """720-row DataFrame with 5 EPIAS columns and DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=720, freq="h")
    data: dict[str, Any] = {}
    for var in EPIAS_VARIABLES:
        data[var] = 500.0 + rng.random(720) * 1000
    return pd.DataFrame(data, index=idx).rename_axis("datetime")


class TestEpiasFeatureEngineer:
    """Tests for EpiasFeatureEngineer."""

    def test_lag_features_created(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """Lag feature FDPP_lag_48 is created."""
        result = engineer.fit_transform(epias_df)
        assert "FDPP_lag_48" in result.columns

    def test_lag_min_lag_enforced(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """All lag feature periods are >= 48."""
        result = engineer.fit_transform(epias_df)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        assert len(lag_cols) > 0
        for col in lag_cols:
            # Column format: {var}_lag_{period}
            period_str = col.rsplit("_", maxsplit=1)[-1]
            period = int(period_str)
            assert period >= 48, f"Lag {col} has period {period} < 48"

    def test_window_leakage_safe(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """Rolling window features have NaN in early rows due to shift."""
        result = engineer.fit_transform(epias_df)
        # Find any rolling window column
        window_cols = [c for c in result.columns if "_window_" in c]
        assert len(window_cols) > 0
        # First rows should be NaN due to periods=48 shift
        first_value = result[window_cols[0]].iloc[0]
        assert pd.isna(first_value), "First row of window feature should be NaN"

    def test_expanding_created(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """Expanding mean feature is created for FDPP."""
        result = engineer.fit_transform(epias_df)
        expanding_cols = [c for c in result.columns if "expanding" in c.lower()]
        assert len(expanding_cols) > 0

    def test_raw_dropped(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """Raw EPIAS columns are dropped when drop_raw=True (default)."""
        result = engineer.fit_transform(epias_df)
        for var in EPIAS_VARIABLES:
            assert var not in result.columns, f"Raw column {var} should be dropped"

    def test_raw_kept_disabled(self, epias_df: pd.DataFrame) -> None:
        """Raw EPIAS columns are kept when drop_raw=False."""
        config = EpiasConfig().model_dump()
        config["drop_raw"] = False
        eng = EpiasFeatureEngineer(config)
        result = eng.fit_transform(epias_df)
        for var in EPIAS_VARIABLES:
            assert var in result.columns, f"Raw column {var} should be kept"

    def test_missing_variable_graceful(
        self,
        engineer: EpiasFeatureEngineer,
    ) -> None:
        """Missing column is skipped without error."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=720, freq="h")
        # Only include 2 of 5 variables
        df = pd.DataFrame(
            {
                "FDPP": 500.0 + rng.random(720) * 1000,
                "Load_Forecast": 500.0 + rng.random(720) * 1000,
            },
            index=idx,
        ).rename_axis("datetime")

        result = engineer.fit_transform(df)
        # Should have lag features for the 2 present variables
        assert "FDPP_lag_48" in result.columns
        assert "Load_Forecast_lag_48" in result.columns
        # Should not have lag features for missing variables
        assert "DAM_Purchase_lag_48" not in result.columns

    def test_all_five_variables(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """Lag features are created for all 5 EPIAS variables."""
        result = engineer.fit_transform(epias_df)
        for var in EPIAS_VARIABLES:
            col = f"{var}_lag_48"
            assert col in result.columns, f"Missing lag feature: {col}"

    def test_nan_early_rows(
        self,
        engineer: EpiasFeatureEngineer,
        epias_df: pd.DataFrame,
    ) -> None:
        """Lag features are NaN in the first 48 rows."""
        result = engineer.fit_transform(epias_df)
        lag_col = "FDPP_lag_48"
        assert lag_col in result.columns
        # First 48 rows should all be NaN
        assert result[lag_col].iloc[:48].isna().all()
        # Row 48 should have a value
        assert pd.notna(result[lag_col].iloc[48])


# ---------------------------------------------------------------------------
# Generation feature tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def gen_config() -> dict[str, Any]:
    """EPIAS config with generation section enabled (composites on)."""
    cfg = EpiasConfig().model_dump()
    gen = GenerationConfig(
        variables=list(GENERATION_VARIABLES),
        composites=GenerationConfig.model_fields["composites"]
        .default_factory()  # type: ignore[union-attr]
        .model_copy(update={"enabled": True}),
    ).model_dump()
    cfg["generation"] = gen
    return cfg


@pytest.fixture()
def gen_engineer(gen_config: dict[str, Any]) -> EpiasFeatureEngineer:
    """EpiasFeatureEngineer with generation config."""
    return EpiasFeatureEngineer(gen_config)


@pytest.fixture()
def gen_df() -> pd.DataFrame:
    """720-row DataFrame with generation columns + market columns."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=720, freq="h")
    data: dict[str, Any] = {}
    # Market columns (needed to avoid warnings)
    for var in EPIAS_VARIABLES:
        data[var] = 500.0 + rng.random(720) * 1000
    # Generation columns
    for var in GENERATION_VARIABLES:
        data[var] = 100.0 + rng.random(720) * 500
    # Ensure gen_total > sum of parts to get valid ratios
    data["gen_total"] = 2000.0 + rng.random(720) * 1000
    return pd.DataFrame(data, index=idx).rename_axis("datetime")


class TestGenerationFeatures:
    """Tests for generation feature engineering (17 fuel types)."""

    def test_generation_lag_features_created(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """gen_natural_gas_lag_48 and gen_wind_lag_48 are created."""
        result = gen_engineer.fit_transform(gen_df)
        assert "gen_natural_gas_lag_48" in result.columns
        assert "gen_wind_lag_48" in result.columns
        assert "gen_total_lag_48" in result.columns

    def test_generation_lag_168_created(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """Generation weekly lag (168h) is created."""
        result = gen_engineer.fit_transform(gen_df)
        assert "gen_natural_gas_lag_168" in result.columns

    def test_generation_lag_min_lag_enforced(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """All generation lag periods are >= 48."""
        result = gen_engineer.fit_transform(gen_df)
        gen_lag_cols = [c for c in result.columns if c.startswith("gen_") and "_lag_" in c]
        assert len(gen_lag_cols) > 0
        for col in gen_lag_cols:
            period_str = col.rsplit("_", maxsplit=1)[-1]
            period = int(period_str)
            assert period >= 48, f"Generation lag {col} has period {period} < 48"

    def test_generation_rolling_features_created(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """Rolling window features are created for generation variables."""
        result = gen_engineer.fit_transform(gen_df)
        window_cols = [c for c in result.columns if c.startswith("gen_") and "_window_" in c]
        assert len(window_cols) > 0

    def test_generation_expanding_features_created(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """Expanding mean features are created for generation variables."""
        result = gen_engineer.fit_transform(gen_df)
        expanding_cols = [
            c for c in result.columns if c.startswith("gen_") and "expanding" in c.lower()
        ]
        assert len(expanding_cols) > 0

    def test_generation_raw_dropped(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """Raw generation columns are dropped when drop_raw=True (default)."""
        result = gen_engineer.fit_transform(gen_df)
        for var in GENERATION_VARIABLES:
            assert var not in result.columns, f"Raw generation column {var} should be dropped"

    def test_generation_raw_kept_disabled(
        self,
        gen_config: dict[str, Any],
        gen_df: pd.DataFrame,
    ) -> None:
        """Raw generation columns kept when drop_raw=False."""
        gen_config["generation"]["drop_raw"] = False
        eng = EpiasFeatureEngineer(gen_config)
        result = eng.fit_transform(gen_df)
        for var in GENERATION_VARIABLES:
            assert var in result.columns, f"Raw generation column {var} should be kept"

    def test_composite_renewable_ratio(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """renewable_ratio_lag_48 is computed from generation composites."""
        result = gen_engineer.fit_transform(gen_df)
        assert "renewable_ratio_lag_48" in result.columns
        # Values should be between 0 and 1 (where not NaN)
        valid = result["renewable_ratio_lag_48"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_composite_thermal_ratio(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """thermal_ratio_lag_48 is computed from generation composites."""
        result = gen_engineer.fit_transform(gen_df)
        assert "thermal_ratio_lag_48" in result.columns
        valid = result["thermal_ratio_lag_48"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_composite_nan_first_48_rows(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """Composite ratio features have NaN in first 48 rows (from lagged inputs)."""
        result = gen_engineer.fit_transform(gen_df)
        assert result["renewable_ratio_lag_48"].iloc[:48].isna().all()
        assert pd.notna(result["renewable_ratio_lag_48"].iloc[48])

    def test_all_generation_variables_get_lags(
        self,
        gen_engineer: EpiasFeatureEngineer,
        gen_df: pd.DataFrame,
    ) -> None:
        """All 8 configured generation variables get lag_48 features."""
        result = gen_engineer.fit_transform(gen_df)
        for var in GENERATION_VARIABLES:
            col = f"{var}_lag_48"
            assert col in result.columns, f"Missing generation lag feature: {col}"
