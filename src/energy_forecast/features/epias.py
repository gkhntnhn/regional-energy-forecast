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
        """Generate EPIAS-derived features.

        Args:
            X: DataFrame with raw EPIAS columns and DatetimeIndex.

        Returns:
            DataFrame with derived features (raw values dropped if configured).
        """
        df = X.copy()
        raw_vars: list[str] = self.config["variables"]
        min_lag: int = self.config["lags"]["min_lag"]

        available_vars = [v for v in raw_vars if v in df.columns]
        if not available_vars:
            logger.warning("No EPIAS columns found in DataFrame, skipping")
            return df

        df = self._add_lags(df, available_vars)
        df = self._add_rolling(df, available_vars, min_lag)
        df = self._add_expanding(df, available_vars, min_lag)

        # Drop raw columns
        if self.config.get("drop_raw", True):
            df = df.drop(columns=available_vars, errors="ignore")

        return df

    # ------------------------------------------------------------------
    # feature-engine: lags
    # ------------------------------------------------------------------

    def _add_lags(self, df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
        lag_values: list[int] = self.config["lags"]["values"]
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
    ) -> pd.DataFrame:
        roll_cfg: dict[str, Any] = self.config["rolling"]
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
    # feature-engine: expanding
    # ------------------------------------------------------------------

    def _add_expanding(
        self,
        df: pd.DataFrame,
        variables: list[str],
        min_lag: int,
    ) -> pd.DataFrame:
        exp_cfg: dict[str, Any] = self.config["expanding"]
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
