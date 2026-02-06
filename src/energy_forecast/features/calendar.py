"""Calendar and temporal feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from loguru import logger

from energy_forecast.features.base import BaseFeatureEngineer


class CalendarFeatureEngineer(BaseFeatureEngineer):
    """Generates datetime, cyclical, holiday, and business hour features.

    Uses feature-engine ``DatetimeFeatures`` and ``CyclicalFeatures``
    for standard extractions. Custom logic for holidays, Ramadan,
    bridge days, and business hours.

    Args:
        config: Calendar feature configuration dict.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate calendar features.

        Args:
            X: DataFrame with DatetimeIndex.

        Returns:
            DataFrame with calendar features added.
        """
        df = X.copy()
        df = self._extract_datetime(df)
        df = self._add_cyclical(df)
        df = self._add_holiday_features(df)
        df = self._add_business_features(df)
        return df

    # ------------------------------------------------------------------
    # feature-engine: datetime extraction
    # ------------------------------------------------------------------

    def _extract_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime components via DatetimeFeatures."""
        extract_list: list[str] = self.config["datetime"]["extract"]

        # DatetimeFeatures doesn't know "week_of_year" — handle separately
        fe_features: list[str] = []
        has_week_of_year = False
        for feat in extract_list:
            if feat == "week_of_year":
                has_week_of_year = True
            else:
                fe_features.append(feat)

        if fe_features:
            dt_tf = DatetimeFeatures(
                variables="index",
                features_to_extract=fe_features,
                drop_original=False,
            )
            df = dt_tf.fit_transform(df)

        if has_week_of_year:
            idx = cast(pd.DatetimeIndex, df.index)
            df["week_of_year"] = idx.isocalendar().week.astype(int).values

        return df

    # ------------------------------------------------------------------
    # feature-engine: cyclical sin/cos
    # ------------------------------------------------------------------

    def _add_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical sin/cos encoding via CyclicalFeatures."""
        cyclical_cfg: dict[str, Any] = self.config["cyclical"]
        cyc_vars = [v for v in cyclical_cfg if v in df.columns]
        if not cyc_vars:
            return df

        max_values: dict[str, int | float] = {}
        for var in cyc_vars:
            cfg = cyclical_cfg[var]
            max_values[var] = cfg["period"] if isinstance(cfg, dict) else int(cfg)

        cyc_tf = CyclicalFeatures(
            variables=cyc_vars,  # type: ignore[arg-type]
            max_values=max_values,
            drop_original=False,
        )
        result: pd.DataFrame = cyc_tf.fit_transform(df)
        return result

    # ------------------------------------------------------------------
    # Custom: holidays, Ramadan, bridge
    # ------------------------------------------------------------------

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday, Ramadan, bridge day, and proximity features."""
        h_cfg: dict[str, Any] = self.config.get("holidays", {})
        h_path = h_cfg.get("path", "data/static/turkish_holidays.parquet")

        holidays_df = self._load_holidays(h_path)

        if holidays_df is not None and len(holidays_df) > 0:
            holiday_dates = set(pd.to_datetime(holidays_df["date"]).dt.date)
            date_series = pd.Series(
                cast(pd.DatetimeIndex, df.index).date, index=df.index
            )
            df["is_holiday"] = date_series.isin(holiday_dates).astype(int)

            # Ramadan
            if h_cfg.get("include_ramadan", True) and "is_ramadan" in holidays_df.columns:
                ram_dates = set(
                    holidays_df.loc[holidays_df["is_ramadan"] == 1, "date"]
                    .pipe(pd.to_datetime)
                    .dt.date
                )
                df["is_ramadan"] = date_series.isin(ram_dates).astype(int)
            else:
                df["is_ramadan"] = 0

            # Bridge days
            if h_cfg.get("bridge_days", True):
                df = self._detect_bridge_days(df)

            # Proximity
            df = self._add_holiday_proximity(df, holiday_dates)
        else:
            df["is_holiday"] = 0
            df["is_ramadan"] = 0
            df["is_bridge_day"] = 0
            df["days_until_holiday"] = -1
            df["days_since_holiday"] = -1

        return df

    @staticmethod
    def _load_holidays(path: str) -> pd.DataFrame | None:
        """Load holidays parquet with graceful fallback."""
        p = Path(path)
        if not p.exists():
            logger.warning("Holiday file not found: {}", p)
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            logger.warning("Failed to read holiday file: {}", p)
            return None

    @staticmethod
    def _detect_bridge_days(df: pd.DataFrame) -> pd.DataFrame:
        """Weekday sandwiched between holiday/weekend on both sides."""
        idx = cast(pd.DatetimeIndex, df.index)
        is_off = (df.get("is_holiday", pd.Series(0, index=df.index)) == 1) | (
            idx.dayofweek >= 5
        )
        prev_off = is_off.shift(24, fill_value=False)
        next_off = is_off.shift(-24, fill_value=False)
        is_weekday = idx.dayofweek < 5
        df["is_bridge_day"] = (is_weekday & prev_off & next_off).astype(int)
        return df

    @staticmethod
    def _add_holiday_proximity(
        df: pd.DataFrame,
        holiday_dates: set[Any],
    ) -> pd.DataFrame:
        """Days until next holiday, days since last holiday."""
        sorted_h = sorted(holiday_dates)
        if not sorted_h:
            df["days_until_holiday"] = -1
            df["days_since_holiday"] = -1
            return df

        unique_dates = pd.Series(cast(pd.DatetimeIndex, df.index).date).unique()
        until_map: dict[Any, int] = {}
        since_map: dict[Any, int] = {}
        for d in unique_dates:
            future = [h for h in sorted_h if h >= d]
            until_map[d] = (future[0] - d).days if future else -1
            past = [h for h in sorted_h if h <= d]
            since_map[d] = (d - past[-1]).days if past else -1

        date_arr = cast(pd.DatetimeIndex, df.index).date
        df["days_until_holiday"] = np.array([until_map.get(d, -1) for d in date_arr])
        df["days_since_holiday"] = np.array([since_map.get(d, -1) for d in date_arr])
        return df

    # ------------------------------------------------------------------
    # Custom: business hours, peak, season
    # ------------------------------------------------------------------

    def _add_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add business-hour, peak, weekend, and season flags."""
        bh: dict[str, Any] = self.config.get("business_hours", {})
        bh_start: int = bh.get("start", 8)
        bh_end: int = bh.get("end", 18)
        peak_start: int = bh.get("peak_start", 17)
        peak_end: int = bh.get("peak_end", 22)

        idx = cast(pd.DatetimeIndex, df.index)
        hour = idx.hour
        dow = idx.dayofweek
        month = idx.month

        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_monday"] = (dow == 0).astype(int)
        df["is_friday"] = (dow == 4).astype(int)
        df["is_business_hours"] = ((hour >= bh_start) & (hour < bh_end)).astype(int)
        df["is_peak"] = ((hour >= peak_start) & (hour < peak_end)).astype(int)

        # Meteorological seasons: 0=winter, 1=spring, 2=summer, 3=autumn
        df["season"] = np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            [0, 1, 2, 3],
            default=0,
        )
        df["is_heating_season"] = month.isin([11, 12, 1, 2, 3]).astype(int)
        df["is_cooling_season"] = month.isin([6, 7, 8, 9]).astype(int)
        return df
