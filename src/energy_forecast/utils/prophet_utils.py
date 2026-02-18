"""Shared Prophet format utilities (DRY across prophet.py and ensemble.py)."""

from __future__ import annotations

import pandas as pd


def to_prophet_format(
    df: pd.DataFrame,
    regressor_names: list[str],
    target_col: str | None = None,
) -> pd.DataFrame:
    """Convert a DatetimeIndex DataFrame to Prophet ds/y/regressors format.

    Args:
        df: DataFrame with DatetimeIndex.
        regressor_names: Regressor column names to include.
        target_col: If provided, include as ``y`` column.

    Returns:
        Prophet-formatted DataFrame with ``ds`` and optional ``y`` + regressors.
    """
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = df.index

    if target_col is not None and target_col in df.columns:
        prophet_df["y"] = df[target_col].values

    for col in regressor_names:
        if col in df.columns:
            prophet_df[col] = df[col].values

    return prophet_df
