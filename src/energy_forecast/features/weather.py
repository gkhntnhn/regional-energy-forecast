"""Weather feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from feature_engine.timeseries.forecasting import WindowFeatures

from energy_forecast.features.base import BaseFeatureEngineer
from energy_forecast.features.custom import DegreeDayFeatures

# WMO 4677 code → 8-group mapping for categorical feature
WMO_GROUP_MAP: dict[int, str] = {
    0: "clear",
    1: "cloudy",
    2: "cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    56: "drizzle",
    57: "drizzle",
    61: "rain",
    63: "rain",
    65: "rain",
    66: "rain",
    67: "rain",
    71: "snow",
    73: "snow",
    75: "snow",
    77: "snow",
    80: "showers",
    81: "showers",
    82: "showers",
    85: "showers",
    86: "showers",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}


def map_wmo_group(code: float) -> str:
    """Map WMO 4677 weather code to one of 8 weather groups.

    Groups: clear, cloudy, fog, drizzle, rain, snow, showers, thunderstorm.
    Unknown or NaN codes map to ``"unknown"``.
    """
    if pd.isna(code):
        return "unknown"
    return WMO_GROUP_MAP.get(int(code), "unknown")


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
        df = self._add_interactions(df)
        df = self._add_rolling(df)
        df = self._add_weather_lags(df)
        df = self._add_quadratic_temperature(df)
        df = self._add_extreme_flags(df)
        df = self._add_severity(df)
        df = self._add_weather_group(df)
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
    # Custom: cross-feature interactions
    # ------------------------------------------------------------------

    @staticmethod
    def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Add weather × calendar interaction features.

        Requires ``is_peak`` from CalendarFeatureEngineer (runs first in pipeline).
        """
        if "wth_cdd" in df.columns and "is_peak" in df.columns:
            df["cdd_x_is_peak"] = df["wth_cdd"] * df["is_peak"]
        return df

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
    # Custom: weather lags (thermal inertia)
    # ------------------------------------------------------------------

    def _add_weather_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged weather features for thermal inertia modeling.

        Buildings have thermal mass — yesterday's temperature still affects
        today's heating/cooling demand. Not leakage since weather is a
        known external variable.
        """
        lag_cfg: dict[str, Any] = self.config.get("weather_lags", {})
        if not lag_cfg.get("enabled", False):
            return df

        hours: list[int] = lag_cfg.get("hours", [6, 12, 18, 24, 30, 36, 42, 48])
        columns: list[str] = lag_cfg.get(
            "columns", ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
        )

        for col in columns:
            if col not in df.columns:
                continue
            for lag in hours:
                df[f"wth_{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    # ------------------------------------------------------------------
    # Custom: quadratic temperature (U-shaped energy relationship)
    # ------------------------------------------------------------------

    def _add_quadratic_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temperature squared to capture non-linear energy relationship.

        Energy demand vs temperature is U-shaped: both very cold (heating)
        and very hot (cooling) extremes increase consumption.
        """
        qt_cfg: dict[str, Any] = self.config.get("quadratic_temperature", {})
        if not qt_cfg.get("enabled", False):
            return df
        if "temperature_2m" not in df.columns:
            return df

        df["wth_temperature_squared"] = df["temperature_2m"] ** 2
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
    # Custom: WMO weather group (string categorical)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_weather_group(df: pd.DataFrame) -> pd.DataFrame:
        """Map WMO weather_code to a categorical weather_group label."""
        if "weather_code" not in df.columns:
            return df
        df["weather_group"] = df["weather_code"].map(map_wmo_group)
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
