"""EPIAS market data feature engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd
from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)
from loguru import logger

from energy_forecast.features.base import BaseFeatureEngineer


class EpiasFeatureEngineer(BaseFeatureEngineer):
    """Generates lag, rolling, and expanding features from EPIAS data.

    CRITICAL: All lags use min_lag=48. Raw EPIAS values are DROPPED
    after feature creation — only derived features remain.

    Args:
        config: EPIAS feature configuration dict.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate EPIAS-derived features (market + generation).

        Args:
            X: DataFrame with raw EPIAS columns and DatetimeIndex.

        Returns:
            DataFrame with derived features (raw values dropped if configured).
        """
        df = X.copy()

        # --- Market features ---
        raw_vars: list[str] = self.config["variables"]
        available_vars = [v for v in raw_vars if v in df.columns]
        if available_vars:
            lag_cfg: dict[str, Any] = self.config["lags"]
            min_lag: int = lag_cfg["min_lag"]
            df = self._add_lags(df, available_vars, lag_cfg["values"])
            df = self._add_rolling(df, available_vars, min_lag, self.config["rolling"])
            df = self._add_expanding(df, available_vars, min_lag, self.config["expanding"])
            if self.config.get("drop_raw", True):
                df = df.drop(columns=available_vars, errors="ignore")
        else:
            logger.warning("No EPIAS market columns found in DataFrame, skipping")

        # --- Generation features ---
        gen_cfg: dict[str, Any] | None = self.config.get("generation")
        if gen_cfg:
            gen_vars: list[str] = gen_cfg["variables"]
            available_gen = [v for v in gen_vars if v in df.columns]
            if available_gen:
                gen_lag_cfg: dict[str, Any] = gen_cfg["lags"]
                gen_min_lag: int = gen_lag_cfg["min_lag"]
                df = self._add_lags(df, available_gen, gen_lag_cfg["values"])
                df = self._add_rolling(
                    df, available_gen, gen_min_lag, gen_cfg["rolling"]
                )
                df = self._add_expanding(
                    df, available_gen, gen_min_lag, gen_cfg["expanding"]
                )
                # Composite ratios BEFORE dropping raw lag columns
                df = self._add_generation_composites(df, gen_cfg)
                if gen_cfg.get("drop_raw", True):
                    df = df.drop(columns=available_gen, errors="ignore")
                logger.info("Generation features added for {} variables", len(available_gen))
            else:
                logger.debug("No generation columns found in DataFrame, skipping")

        return df

    # ------------------------------------------------------------------
    # feature-engine: lags
    # ------------------------------------------------------------------

    def _add_lags(
        self,
        df: pd.DataFrame,
        variables: list[str],
        lag_values: list[int],
    ) -> pd.DataFrame:
        lag_tf = LagFeatures(
            variables=variables,  # type: ignore[arg-type]
            periods=lag_values,
            sort_index=True,
            missing_values="ignore",
            drop_original=False,
            drop_na=False,
        )
        result: pd.DataFrame = lag_tf.fit_transform(df)
        return result

    # ------------------------------------------------------------------
    # feature-engine: rolling windows
    # ------------------------------------------------------------------

    def _add_rolling(
        self,
        df: pd.DataFrame,
        variables: list[str],
        min_lag: int,
        roll_cfg: dict[str, Any],
    ) -> pd.DataFrame:
        windows: list[int] = roll_cfg["windows"]
        functions: list[str] = roll_cfg["functions"]

        for w in windows:
            win_tf = WindowFeatures(
                variables=variables,  # type: ignore[arg-type]
                window=w,
                functions=functions,
                periods=min_lag,
                sort_index=True,
                missing_values="ignore",
                drop_original=False,
                drop_na=False,
            )
            df = win_tf.fit_transform(df)
        return df

    # ------------------------------------------------------------------
    # Custom: generation composite ratios
    # ------------------------------------------------------------------

    @staticmethod
    def _add_generation_composites(
        df: pd.DataFrame,
        gen_cfg: dict[str, Any],
    ) -> pd.DataFrame:
        """Add renewable and thermal ratio features from generation lags.

        Computes supply-side mix ratios (renewable/total, thermal/total)
        using lagged generation data. These capture grid supply composition
        which correlates with demand patterns.
        """
        comp_cfg: dict[str, Any] = gen_cfg.get("composites", {})
        if not comp_cfg.get("enabled", False):
            return df

        lag: int = comp_cfg.get("lag", 48)
        total_col = f"{comp_cfg.get('total_var', 'gen_total')}_lag_{lag}"
        if total_col not in df.columns:
            return df

        total = df[total_col].replace(0, float("nan"))

        # Renewable ratio
        renewable_vars: list[str] = comp_cfg.get(
            "renewable_vars", ["gen_wind", "gen_sun", "gen_river", "gen_dammed_hydro"]
        )
        renewable_cols = [f"{v}_lag_{lag}" for v in renewable_vars]
        available_ren = [c for c in renewable_cols if c in df.columns]
        if available_ren:
            df["renewable_ratio_lag_48"] = df[available_ren].sum(axis=1) / total

        # Thermal ratio
        thermal_vars: list[str] = comp_cfg.get(
            "thermal_vars", ["gen_natural_gas", "gen_lignite", "gen_import_coal"]
        )
        thermal_cols = [f"{v}_lag_{lag}" for v in thermal_vars]
        available_th = [c for c in thermal_cols if c in df.columns]
        if available_th:
            df["thermal_ratio_lag_48"] = df[available_th].sum(axis=1) / total

        return df

    # ------------------------------------------------------------------
    # feature-engine: expanding
    # ------------------------------------------------------------------

    def _add_expanding(
        self,
        df: pd.DataFrame,
        variables: list[str],
        min_lag: int,
        exp_cfg: dict[str, Any],
    ) -> pd.DataFrame:
        exp_tf = ExpandingWindowFeatures(
            variables=variables,  # type: ignore[arg-type]
            min_periods=exp_cfg["min_periods"],
            functions=exp_cfg["functions"],
            periods=min_lag,
            sort_index=True,
            missing_values="ignore",
            drop_original=False,
            drop_na=False,
        )
        result: pd.DataFrame = exp_tf.fit_transform(df)
        return result
