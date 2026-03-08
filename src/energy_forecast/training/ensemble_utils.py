"""Shared utilities for ensemble training and prediction.

Provides ``build_context_features`` to eliminate the triple-duplicated
if/elif block that extracts hour, day_of_week, is_weekend, month, and
is_holiday from a DatetimeIndex into a target DataFrame.
"""

from __future__ import annotations

import pandas as pd


def build_context_features(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    context_features: list[str],
    *,
    max_len: int | None = None,
    cast_categorical_to_str: bool = False,
) -> None:
    """Add context features from DatetimeIndex to *target_df* in-place.

    Extracts temporal attributes from ``target_df.index`` and optionally
    copies ``is_holiday`` from *source_df*.

    Args:
        target_df: DataFrame to add context features to (mutated in-place).
        source_df: Original data slice (for ``is_holiday`` lookup).
        context_features: Which features to add (e.g. ``["hour", "month"]``).
        max_len: Truncate array values to this length (for aligned slicing).
        cast_categorical_to_str: Convert hour/day_of_week/month to ``str``
            (required when feeding to CatBoost meta-learner).
    """
    dt_idx = pd.DatetimeIndex(target_df.index)

    if "hour" in context_features:
        target_df["hour"] = dt_idx.hour
    if "day_of_week" in context_features:
        target_df["day_of_week"] = dt_idx.dayofweek
    if "is_weekend" in context_features:
        target_df["is_weekend"] = (dt_idx.dayofweek >= 5).astype(int)
    if "month" in context_features:
        target_df["month"] = dt_idx.month
    if "is_holiday" in context_features:
        if "is_holiday" in source_df.columns:
            vals = source_df["is_holiday"].values
            if max_len is not None:
                vals = vals[:max_len]
            target_df["is_holiday"] = vals
        else:
            target_df["is_holiday"] = 0

    if cast_categorical_to_str:
        for col in ["hour", "day_of_week", "month"]:
            if col in target_df.columns:
                target_df[col] = target_df[col].astype(str)
