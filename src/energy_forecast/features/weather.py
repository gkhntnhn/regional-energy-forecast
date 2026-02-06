"""Weather feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from feature_engine.timeseries.forecasting import WindowFeatures

from energy_forecast.features.base import BaseFeatureEngineer
from energy_forecast.features.custom import DegreeDayFeatures


class WeatherFeatureEngineer(BaseFeatureEngineer):
    """Generates HDD/CDD, rolling, extreme flags, and severity features.

    NOTE: Weather forecast data is NOT leakage — available from OpenMeteo
    at prediction time. Rolling uses ``periods=1`` (default).

    Args:
        config: Weather feature configuration dict.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate weather features.

        Args:
            X: DataFrame with weather columns (temperature_2m, etc.).

        Returns:
            DataFrame with weather features added.
        """
        df = X.copy()
        df = self._add_degree_days(df)
        df = self._add_rolling(df)
        df = self._add_extreme_flags(df)
        df = self._add_severity(df)
        df = self._add_temp_change(df)
        return df

    # ------------------------------------------------------------------
    # Custom: HDD / CDD
    # ------------------------------------------------------------------

    def _add_degree_days(self, df: pd.DataFrame) -> pd.DataFrame:
        th: dict[str, Any] = self.config["thresholds"]
        dd_tf = DegreeDayFeatures(
            temp_variable="temperature_2m",
            hdd_base=th["hdd_base"],
            cdd_base=th["cdd_base"],
        )
        result: pd.DataFrame = dd_tf.fit_transform(df)
        return result

    # ------------------------------------------------------------------
    # feature-engine: rolling
    # ------------------------------------------------------------------

    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        roll_cfg: dict[str, Any] = self.config["rolling"]
        windows: list[int] = roll_cfg["windows"]
        functions: list[str] = roll_cfg["functions"]

        roll_vars = [v for v in ["temperature_2m"] if v in df.columns]
        if not roll_vars:
            return df

        for w in windows:
            win_tf = WindowFeatures(
                variables=roll_vars,  # type: ignore[arg-type]
                window=w,
                functions=functions,
                periods=1,  # weather is NOT leakage — minimal shift
                sort_index=True,
                missing_values="ignore",
                drop_original=False,
                drop_na=False,
            )
            df = win_tf.fit_transform(df)
        return df

    # ------------------------------------------------------------------
    # Custom: extreme flags
    # ------------------------------------------------------------------

    def _add_extreme_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        th: dict[str, Any] = self.config["thresholds"]

        if "temperature_2m" in df.columns:
            df["wth_extreme_cold"] = (df["temperature_2m"] < th["extreme_cold"]).astype(int)
            df["wth_extreme_hot"] = (df["temperature_2m"] > th["extreme_hot"]).astype(int)

        if "wind_speed_10m" in df.columns:
            df["wth_extreme_wind"] = (df["wind_speed_10m"] > th["high_wind"]).astype(int)

        sev_cfg: dict[str, Any] = self.config.get("severity", {})
        precip_th: float = sev_cfg.get("precip_threshold", 10.0)
        if "precipitation" in df.columns:
            df["wth_heavy_precip"] = (df["precipitation"] > precip_th).astype(int)

        return df

    # ------------------------------------------------------------------
    # Custom: WMO severity
    # ------------------------------------------------------------------

    def _add_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        sev_cfg: dict[str, Any] = self.config.get("severity", {})
        if not sev_cfg.get("enabled", True):
            return df
        if "weather_code" not in df.columns:
            return df

        df["wth_severity"] = df["weather_code"].apply(_map_wmo_severity)
        df["wth_is_severe"] = (df["wth_severity"] >= 2).astype(int)
        return df

    # ------------------------------------------------------------------
    # Custom: temperature change
    # ------------------------------------------------------------------

    @staticmethod
    def _add_temp_change(df: pd.DataFrame) -> pd.DataFrame:
        if "temperature_2m" not in df.columns:
            return df
        temp = df["temperature_2m"]
        df["wth_temp_change_3h"] = temp - temp.shift(3)
        df["wth_temp_change_24h"] = temp - temp.shift(24)
        return df


def _map_wmo_severity(code: float) -> int:
    """Map WMO weather code to 4-level severity (0-3).

    0 = Clear/Cloudy, 1 = Fog/Drizzle/Light rain,
    2 = Snow/Freezing, 3 = Thunderstorm/Heavy.
    """
    if np.isnan(code):
        return 0
    c = int(code)
    if c < 4:
        return 0
    if c < 70:
        return 1
    if c < 90:
        return 2
    return 3
