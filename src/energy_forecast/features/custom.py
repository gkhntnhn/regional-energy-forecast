"""Custom sklearn-compatible transformers for domain-specific features.

These transformers have no equivalent in feature-engine:
- EwmaFeatures: Exponential weighted moving average with leakage-safe shift
- MomentumFeatures: Velocity and percentage change features
- QuantileFeatures: Rolling quantile features with shift
- DegreeDayFeatures: Heating/Cooling Degree Days from temperature
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EwmaFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """EWMA features with min_lag shift for leakage prevention.

    Computes ``series.ewm(span=S).mean().shift(periods)`` for each span.

    Args:
        variables: Columns to compute EWMA on.
        spans: EWMA span values.
        periods: Shift amount after EWMA (min_lag=48).
    """

    def __init__(
        self,
        variables: list[str],
        spans: list[int],
        periods: int = 48,
    ) -> None:
        self.variables = variables
        self.spans = spans
        self.periods = periods

    def fit(self, X: pd.DataFrame, y: Any = None) -> EwmaFeatures:
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add EWMA features to DataFrame."""
        df = X.copy()
        for var in self.variables:
            if var not in df.columns:
                continue
            series = df[var]
            for span in self.spans:
                col_name = f"{var}_ewma_{span}"
                df[col_name] = series.ewm(span=span, adjust=False).mean().shift(self.periods)
        return df


class MomentumFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Momentum (velocity) and percentage change features.

    Momentum: ``shift(min_lag) - shift(min_lag + period)``
    Pct change: ``momentum / shift(min_lag + period) * 100``

    Args:
        variables: Columns to compute momentum on.
        min_lag: Minimum lag for leakage safety.
        momentum_periods: Periods for momentum computation.
    """

    def __init__(
        self,
        variables: list[str],
        min_lag: int = 48,
        momentum_periods: list[int] | None = None,
    ) -> None:
        self.variables = variables
        self.min_lag = min_lag
        self.momentum_periods = momentum_periods or [24, 168]

    def fit(self, X: pd.DataFrame, y: Any = None) -> MomentumFeatures:
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and pct_change features to DataFrame."""
        df = X.copy()
        for var in self.variables:
            if var not in df.columns:
                continue
            series = df[var]
            recent = series.shift(self.min_lag)
            for period in self.momentum_periods:
                older = series.shift(self.min_lag + period)
                momentum = recent - older
                df[f"{var}_momentum_{period}"] = momentum
                df[f"{var}_pct_change_{period}"] = momentum / (older + 1e-9) * 100
        return df


class QuantileFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Rolling quantile features with min_lag shift.

    Computes ``series.shift(periods).rolling(window).quantile(q)``.

    Args:
        variables: Columns to compute quantiles on.
        quantiles: Quantile values (e.g., [0.25, 0.50, 0.75]).
        window: Rolling window size.
        periods: Shift amount (min_lag=48).
    """

    def __init__(
        self,
        variables: list[str],
        quantiles: list[float] | None = None,
        window: int = 168,
        periods: int = 48,
    ) -> None:
        self.variables = variables
        self.quantiles = quantiles or [0.25, 0.50, 0.75]
        self.window = window
        self.periods = periods

    def fit(self, X: pd.DataFrame, y: Any = None) -> QuantileFeatures:
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add rolling quantile features to DataFrame."""
        df = X.copy()
        for var in self.variables:
            if var not in df.columns:
                continue
            shifted = df[var].shift(self.periods)
            for q in self.quantiles:
                pct = int(q * 100)
                col_name = f"{var}_q{pct}_{self.window}"
                df[col_name] = shifted.rolling(self.window).quantile(q)
        return df


class DegreeDayFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Heating/Cooling Degree Day features from temperature.

    HDD = max(hdd_base - temperature, 0)
    CDD = max(temperature - cdd_base, 0)

    Args:
        temp_variable: Temperature column name.
        hdd_base: HDD base temperature.
        cdd_base: CDD base temperature.
    """

    def __init__(
        self,
        temp_variable: str = "temperature_2m",
        hdd_base: float = 18.0,
        cdd_base: float = 24.0,
    ) -> None:
        self.temp_variable = temp_variable
        self.hdd_base = hdd_base
        self.cdd_base = cdd_base

    def fit(self, X: pd.DataFrame, y: Any = None) -> DegreeDayFeatures:
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add HDD and CDD features to DataFrame."""
        df = X.copy()
        if self.temp_variable not in df.columns:
            return df
        temp = df[self.temp_variable]
        df["wth_hdd"] = np.maximum(self.hdd_base - temp, 0.0)
        df["wth_cdd"] = np.maximum(temp - self.cdd_base, 0.0)
        return df
