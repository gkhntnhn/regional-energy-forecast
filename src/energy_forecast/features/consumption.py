"""Consumption lag and rolling feature engineering."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)

from energy_forecast.features.base import BaseFeatureEngineer
from energy_forecast.features.custom import (
    EwmaFeatures,
    MomentumFeatures,
    QuantileFeatures,
)


class ConsumptionFeatureEngineer(BaseFeatureEngineer):
    """Generates lag, rolling, EWMA, momentum, and quantile features.

    CRITICAL: All lags use min_lag=48 to prevent data leakage.
    WindowFeatures ``periods=48`` ensures shift-after-roll safety.

    Args:
        config: Consumption feature configuration dict.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate consumption features.

        Args:
            X: DataFrame with ``consumption`` column and DatetimeIndex.

        Returns:
            DataFrame with consumption features added.
        """
        df = X.copy()
        min_lag: int = self.config["lags"]["min_lag"]

        df = self._add_lags(df)
        df = self._add_rolling(df, min_lag)
        df = self._add_expanding(df, min_lag)
        df = self._add_ewma(df, min_lag)
        df = self._add_momentum(df, min_lag)
        df = self._add_quantile(df, min_lag)
        df = self._add_trend_ratio(df)
        df = self._add_week_ratio(df)
        df = self._add_target_encoding(df, min_lag)
        return df

    # ------------------------------------------------------------------
    # feature-engine: lags
    # ------------------------------------------------------------------

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_values: list[int] = self.config["lags"]["values"]
        lag_tf = LagFeatures(
            variables=["consumption"],
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

    def _add_rolling(self, df: pd.DataFrame, min_lag: int) -> pd.DataFrame:
        windows: list[int] = self.config["rolling"]["windows"]
        functions: list[str] = self.config["rolling"]["functions"]
        for w in windows:
            win_tf = WindowFeatures(
                variables=["consumption"],
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

    def _add_expanding(self, df: pd.DataFrame, min_lag: int) -> pd.DataFrame:
        exp_cfg: dict[str, Any] = self.config["expanding"]
        exp_tf = ExpandingWindowFeatures(
            variables=["consumption"],
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

    # ------------------------------------------------------------------
    # Custom: EWMA, momentum, quantile
    # ------------------------------------------------------------------

    def _add_ewma(self, df: pd.DataFrame, min_lag: int) -> pd.DataFrame:
        spans: list[int] = self.config["ewma"]["spans"]
        ewma_tf = EwmaFeatures(
            variables=["consumption"],
            spans=spans,
            periods=min_lag,
        )
        result: pd.DataFrame = ewma_tf.fit_transform(df)
        return result

    def _add_momentum(self, df: pd.DataFrame, min_lag: int) -> pd.DataFrame:
        periods: list[int] = self.config["momentum"]["periods"]
        mom_tf = MomentumFeatures(
            variables=["consumption"],
            min_lag=min_lag,
            momentum_periods=periods,
        )
        result: pd.DataFrame = mom_tf.fit_transform(df)
        return result

    def _add_quantile(self, df: pd.DataFrame, min_lag: int) -> pd.DataFrame:
        q_cfg: dict[str, Any] = self.config["quantile"]
        q_tf = QuantileFeatures(
            variables=["consumption"],
            quantiles=q_cfg["quantiles"],
            window=q_cfg["window"],
            periods=min_lag,
        )
        result: pd.DataFrame = q_tf.fit_transform(df)
        return result

    # ------------------------------------------------------------------
    # Trend ratio: lag_short / lag_long → weekly trend proxy
    # ------------------------------------------------------------------

    def _add_trend_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        trend_cfg: dict[str, Any] | None = self.config.get("trend_ratio")
        if not trend_cfg:
            return df
        for pair in trend_cfg["pairs"]:
            num_lag: int = pair["numerator_lag"]
            den_lag: int = pair["denominator_lag"]
            num_col = f"consumption_lag_{num_lag}"
            den_col = f"consumption_lag_{den_lag}"
            out_col = f"consumption_trend_ratio_{num_lag}_{den_lag}"
            if num_col in df.columns and den_col in df.columns:
                df[out_col] = df[num_col] / df[den_col].replace(0, float("nan"))
        return df

    # ------------------------------------------------------------------
    # Week ratio: current week vs historical average (normalization)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_week_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """Ratio of 1-week lag to expanding mean for trend normalization.

        Captures whether recent consumption is above/below historical
        average, helping reduce heteroscedasticity across demand levels.
        """
        num = "consumption_lag_168"
        den = "consumption_expanding_mean"
        if num in df.columns and den in df.columns:
            df["consumption_week_ratio"] = df[num] / df[den].replace(0, float("nan"))
        return df

    # ------------------------------------------------------------------
    # Target encoding: hour x dow expanding mean (leakage-safe)
    # ------------------------------------------------------------------

    def _add_target_encoding(self, df: pd.DataFrame, min_lag: int) -> pd.DataFrame:
        """Hour×DayOfWeek target encoding via expanding mean.

        168 unique bins (Mon 00:00 ... Sun 23:00). Expanding mean with
        min_lag shift prevents leakage. Gives CatBoost an explicit
        representation of the most important interaction.
        """
        te_cfg: dict[str, Any] = self.config.get("target_encoding", {})
        if not te_cfg.get("enabled", False):
            return df
        if "consumption" not in df.columns:
            return df

        idx = cast(pd.DatetimeIndex, df.index)
        group_key = pd.Series(idx.dayofweek * 24 + idx.hour, index=df.index)

        profile = df["consumption"].groupby(group_key).transform(
            lambda g: g.expanding().mean().shift(min_lag)
        )
        df["consumption_hourly_profile"] = profile
        return df
