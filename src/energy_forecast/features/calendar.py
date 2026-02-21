"""Calendar and temporal feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from loguru import logger
from sklearn.preprocessing import SplineTransformer

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
        df = self._add_spline_seasonality(df)
        df = self._add_holiday_features(df)
        df = self._add_holiday_anticipation(df)
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
        """Add holiday, Ramadan, bridge day, interaction, and proximity features."""
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

            # Bayram day number (1-3 Ramazan, 1-4 Kurban, 0 otherwise)
            if "bayram_gun_no" in holidays_df.columns:
                bayram_map = dict(
                    zip(
                        pd.to_datetime(holidays_df["date"]).dt.date,
                        holidays_df["bayram_gun_no"],
                        strict=False,
                    )
                )
                df["bayram_gun_no"] = date_series.map(bayram_map).fillna(0).astype(int)
            else:
                df["bayram_gun_no"] = 0

            # Countdown to next bayram (computed for ALL dates, not just holidays)
            if "bayram_gun_no" in holidays_df.columns:
                bayram_dates = sorted(
                    pd.to_datetime(
                        holidays_df.loc[holidays_df["bayram_gun_no"] == 1, "date"]
                    ).dt.date
                )
            else:
                bayram_dates = []
            if bayram_dates:
                df = self._add_bayram_countdown(df, bayram_dates)
            else:
                df["bayrama_kalan_gun"] = -1

            # Holiday type classification: dini / resmi / none
            df = self._add_holiday_type(df, holidays_df, date_series)

            # Holiday × hour interaction (captures different hourly patterns)
            idx = cast(pd.DatetimeIndex, df.index)
            df["is_holiday_x_hour"] = df["is_holiday"] * idx.hour

            # Ramadan × hour interaction (captures shifted eating/activity patterns)
            df["is_ramadan_x_hour"] = df["is_ramadan"] * idx.hour

            # Bridge days (extended detection for holiday+weekend adjacency)
            if h_cfg.get("bridge_days", True):
                df = self._detect_bridge_days(df)

            # Proximity
            df = self._add_holiday_proximity(df, holiday_dates)
        else:
            df["is_holiday"] = 0
            df["is_ramadan"] = 0
            df["bayram_gun_no"] = 0
            df["bayrama_kalan_gun"] = -1
            df["tatil_tipi"] = 0
            df["is_holiday_x_hour"] = 0
            df["is_ramadan_x_hour"] = 0
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
        except Exception as e:
            logger.warning("Failed to read holiday file {}: {}", p, e)
            return None

    # Religious holiday keywords for tatil_tipi classification
    _DINI_KEYWORDS = frozenset({
        "ramazan", "eid al-fitr", "eid al-adha",
        "kurban bayrami", "ramazan bayrami",
    })

    @staticmethod
    def _add_holiday_type(
        df: pd.DataFrame,
        holidays_df: pd.DataFrame,
        date_series: pd.Series[Any],
    ) -> pd.DataFrame:
        """Classify holidays as dini (religious) or resmi (official).

        Encoded as integer: 0=none, 1=resmi, 2=dini.
        """
        name_col = "holiday_name" if "holiday_name" in holidays_df.columns else "name"
        if name_col not in holidays_df.columns:
            df["tatil_tipi"] = 0
            return df

        # Build date → type mapping
        type_map: dict[Any, int] = {}
        for _, row in holidays_df.iterrows():
            d = pd.to_datetime(row["date"]).date()
            name_lower = str(row[name_col]).lower()
            is_dini = any(kw in name_lower for kw in CalendarFeatureEngineer._DINI_KEYWORDS)
            # dini=2 takes priority if a date has both (e.g., "Eid al-Fitr; Republic Day")
            current = type_map.get(d, 0)
            new_val = 2 if is_dini else 1
            type_map[d] = max(current, new_val)

        df["tatil_tipi"] = date_series.map(type_map).fillna(0).astype(int)
        return df

    @staticmethod
    def _detect_bridge_days(df: pd.DataFrame) -> pd.DataFrame:
        """Detect bridge days: weekdays near holiday+weekend clusters.

        A bridge day is a weekday that, if taken off, extends a
        holiday+weekend combination into a longer break. Detects:
        1. Classic sandwich: weekday between two off-days
        2. Before cluster: weekday followed by 2+ consecutive off-days
           (e.g., Thursday before Friday-holiday + weekend)
        3. After cluster: weekday preceded by 2+ consecutive off-days
           (e.g., Tuesday after weekend + Monday-holiday)

        Cases 2 and 3 require a holiday within the off-day cluster to avoid
        flagging every Friday/Monday adjacent to a normal weekend.
        """
        idx = cast(pd.DatetimeIndex, df.index)
        is_holiday = df.get("is_holiday", pd.Series(0, index=df.index)) == 1
        is_off = is_holiday | (idx.dayofweek >= 5)
        is_weekday = idx.dayofweek < 5

        prev_off = is_off.shift(24, fill_value=False)
        next_off = is_off.shift(-24, fill_value=False)
        prev_prev_off = is_off.shift(48, fill_value=False)
        next_next_off = is_off.shift(-48, fill_value=False)

        # Holiday within 2 days on either side
        prev_holiday = is_holiday.shift(24, fill_value=False)
        next_holiday = is_holiday.shift(-24, fill_value=False)
        prev_prev_holiday = is_holiday.shift(48, fill_value=False)
        next_next_holiday = is_holiday.shift(-48, fill_value=False)
        nearby_holiday = prev_holiday | next_holiday | prev_prev_holiday | next_next_holiday

        # 1. Classic: sandwiched between two off-days
        classic = is_weekday & prev_off & next_off

        # 2. Before a 2-day off cluster containing a holiday
        before_cluster = is_weekday & next_off & next_next_off & nearby_holiday

        # 3. After a 2-day off cluster containing a holiday
        after_cluster = is_weekday & prev_off & prev_prev_off & nearby_holiday

        df["is_bridge_day"] = (classic | before_cluster | after_cluster).astype(int)
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

    @staticmethod
    def _add_bayram_countdown(
        df: pd.DataFrame,
        bayram_starts: list[Any],
    ) -> pd.DataFrame:
        """Days until next bayram start date (Ramazan or Kurban)."""
        unique_dates = pd.Series(cast(pd.DatetimeIndex, df.index).date).unique()
        countdown_map: dict[Any, int] = {}
        for d in unique_dates:
            future = [b for b in bayram_starts if b >= d]
            countdown_map[d] = (future[0] - d).days if future else -1

        date_arr = cast(pd.DatetimeIndex, df.index).date
        df["bayrama_kalan_gun"] = np.array(
            [countdown_map.get(d, -1) for d in date_arr]
        )
        return df

    # ------------------------------------------------------------------
    # Custom: holiday anticipation (forward-looking, NOT leakage)
    # ------------------------------------------------------------------

    def _add_holiday_anticipation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward-looking holiday features.

        For each window N, counts holidays in the next N days.
        NOT leakage — calendar is deterministic (known future).
        Captures pre-holiday demand changes (stocking, production shifts).
        """
        ant_cfg: dict[str, Any] = self.config.get("anticipation", {})
        if not ant_cfg.get("enabled", False):
            return df

        windows: list[int] = ant_cfg.get("windows", [3, 7, 15])
        holiday_cols = ["is_holiday", "is_ramadan"]

        for col in holiday_cols:
            if col not in df.columns:
                continue
            series = df[col]
            for n_days in windows:
                n_hours = n_days * 24
                # Forward rolling sum: how many holiday hours in next N days
                # shift(-n_hours) + rolling(n_hours) = forward window
                fwd = series.shift(-n_hours).rolling(n_hours, min_periods=1).sum()
                # Normalize to daily count
                df[f"{col}_next_{n_days}d"] = fwd / 24.0

        return df

    # ------------------------------------------------------------------
    # Custom: periodic spline seasonality
    # ------------------------------------------------------------------

    def _add_spline_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add periodic spline encoding for hour-of-day.

        More flexible than sin/cos (which captures only one harmonic).
        Splines can model asymmetric and multi-modal daily patterns.
        """
        sp_cfg: dict[str, Any] = self.config.get("spline_seasonality", {})
        if not sp_cfg.get("enabled", False):
            return df

        n_splines: int = sp_cfg.get("n_splines", 12)
        if "hour" not in df.columns:
            return df

        n_knots = n_splines + 1
        spline_tf = SplineTransformer(
            degree=3,
            n_knots=n_knots,
            knots=np.linspace(0, 24, n_knots).reshape(-1, 1),
            extrapolation="periodic",
            include_bias=True,
        )
        hour_arr: np.ndarray[Any, Any] = np.asarray(df["hour"].values, dtype=float).reshape(-1, 1)
        spline_feats = spline_tf.fit_transform(hour_arr)
        for i in range(spline_feats.shape[1]):
            df[f"spline_hour_{i}"] = spline_feats[:, i]

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
        df["is_sunday"] = (dow == 6).astype(int)
        df["is_monday"] = (dow == 0).astype(int)
        df["is_friday"] = (dow == 4).astype(int)
        df["is_business_hours"] = ((hour >= bh_start) & (hour < bh_end)).astype(int)
        df["is_peak"] = ((hour >= peak_start) & (hour < peak_end)).astype(int)

        # Interaction: weekend-specific hourly pattern
        df["is_weekend_x_hour"] = df["is_weekend"] * hour

        # Ramp periods: morning/evening demand transition windows
        df["is_ramp_morning"] = hour.isin([6, 7, 8, 9]).astype(int)
        df["is_ramp_evening"] = hour.isin([18, 19, 20, 21]).astype(int)

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
