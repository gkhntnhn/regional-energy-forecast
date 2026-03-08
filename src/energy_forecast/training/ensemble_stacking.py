"""Ensemble stacking: OOF builder, meta-learner training, and test evaluation.

Extracted from ``ensemble_trainer.py`` to reduce its size. All functions are
module-level (no class dependency) and receive explicit parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from energy_forecast.training.ensemble_utils import build_context_features
from energy_forecast.training.metrics import mape as mape_fn
from energy_forecast.training.splitter import TimeSeriesSplitter

if TYPE_CHECKING:
    from energy_forecast.config import CrossValidationConfig, StackingMetaLearnerConfig


def build_oof_dataframe(
    model_results: dict[str, Any],
    df: pd.DataFrame,
    active_models: list[str],
    cv_config: CrossValidationConfig,
    context_features: list[str],
) -> pd.DataFrame:
    """Build OOF prediction matrix from all CV splits with context features.

    Reconstructs timestamps by re-running the splitter to get val indices,
    then joins base model val_predictions with hour/weekday/holiday context.

    Args:
        model_results: Base model training results with val_predictions.
        df: Original feature DataFrame with DatetimeIndex.
        active_models: List of active model names.
        cv_config: Cross-validation configuration.
        context_features: List of context feature names for stacking.

    Returns:
        DataFrame: ``[pred_catboost, pred_prophet, ..., hour, dow, ..., y_true]``
    """
    splitter = TimeSeriesSplitter.from_config(cv_config)

    oof_parts: list[pd.DataFrame] = []
    for split_info, _train_df, val_slice, _test_df in splitter.iter_splits(df):
        split_idx = split_info.split_idx

        # Collect each model's val predictions for this split
        preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
        y_true: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

        for model_name in active_models:
            sr = model_results[model_name].training_result.split_results[split_idx]
            if sr.val_predictions is not None:
                preds[model_name] = sr.val_predictions
                if y_true is None and sr.val_actuals is not None:
                    y_true = sr.val_actuals

        if not preds or y_true is None:
            continue

        # Truncate to common length (TFT may produce fewer rows)
        min_len = min(len(p) for p in preds.values())
        min_len = min(min_len, len(y_true))
        val_slice = val_slice.iloc[:min_len]

        row_df = pd.DataFrame(index=val_slice.index)
        for model_name, pred_arr in preds.items():
            row_df[f"pred_{model_name}"] = pred_arr[:min_len]

        build_context_features(
            row_df, val_slice, context_features, max_len=min_len,
        )

        row_df["y_true"] = y_true[:min_len]
        oof_parts.append(row_df)

    return pd.concat(oof_parts, axis=0).sort_index()


def train_meta_learner(
    oof_df: pd.DataFrame,
    meta_learner_config: StackingMetaLearnerConfig,
) -> tuple[CatBoostRegressor, float]:
    """Train CatBoost meta-learner on OOF predictions.

    Uses temporal 80/20 split for validation (no shuffle).

    Args:
        oof_df: OOF DataFrame with ``pred_*``, context, and ``y_true`` columns.
        meta_learner_config: Meta-learner hyper-parameter configuration.

    Returns:
        Tuple of (trained meta-learner, validation MAPE).
    """
    cfg = meta_learner_config
    feature_cols = [c for c in oof_df.columns if c != "y_true"]

    x_meta = oof_df[feature_cols].copy()
    y_meta = oof_df["y_true"]

    # Categorical features for CatBoost
    cat_cols = [c for c in ["hour", "day_of_week", "month"] if c in feature_cols]
    for col in cat_cols:
        x_meta[col] = x_meta[col].astype(str)
    cat_indices = [feature_cols.index(c) for c in cat_cols]

    # Temporal 80/20 split (no shuffle — time series)
    split_point = int(len(x_meta) * 0.8)
    x_train, x_val = x_meta.iloc[:split_point], x_meta.iloc[split_point:]
    y_train, y_val = y_meta.iloc[:split_point], y_meta.iloc[split_point:]

    logger.info(
        "Training meta-learner: {} features, {} train / {} val rows",
        len(feature_cols), len(x_train), len(x_val),
    )

    meta_model = CatBoostRegressor(
        depth=cfg.depth,
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        loss_function=cfg.loss_function,
        l2_leaf_reg=cfg.l2_leaf_reg,
        early_stopping_rounds=cfg.early_stopping_rounds,
        task_type=cfg.task_type,
        cat_features=cat_indices,
        verbose=50,
    )
    meta_model.fit(x_train, y_train, eval_set=(x_val, y_val))

    # Validation MAPE
    val_pred = meta_model.predict(x_val)
    meta_val_mape = float(mape_fn(y_val.to_numpy(), val_pred))

    # Log feature importance
    importances = dict(
        zip(feature_cols, meta_model.get_feature_importance(), strict=True)
    )
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Meta-learner val MAPE: {:.3f}%", meta_val_mape)
    logger.info("Meta-learner feature importance (top 5):")
    for name, imp in sorted_imp[:5]:
        logger.info("  {}: {:.1f}", name, imp)

    return meta_model, meta_val_mape


def compute_stacking_test_mape(
    meta_model: CatBoostRegressor,
    model_results: dict[str, Any],
    df: pd.DataFrame,
    active_models: list[str],
    cv_config: CrossValidationConfig,
    context_features: list[str],
) -> list[float]:
    """Compute real test MAPE by applying meta-learner to test predictions.

    Args:
        meta_model: Trained CatBoost meta-learner.
        model_results: Base model results with test_predictions per split.
        df: Original feature DataFrame for timestamp reconstruction.
        active_models: List of active model names.
        cv_config: Cross-validation configuration.
        context_features: List of context feature names for stacking.

    Returns:
        List of test MAPEs, one per CV split.
    """
    splitter = TimeSeriesSplitter.from_config(cv_config)
    test_mapes: list[float] = []

    for split_info, _train_df, _val_df, test_slice in splitter.iter_splits(df):
        split_idx = split_info.split_idx

        # Collect test predictions
        preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
        y_test: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None

        for model_name in active_models:
            sr = model_results[model_name].training_result.split_results[split_idx]
            if sr.test_predictions is not None:
                preds[model_name] = sr.test_predictions
                if y_test is None and sr.test_actuals is not None:
                    y_test = sr.test_actuals

        if not preds or y_test is None:
            continue

        min_len = min(len(p) for p in preds.values())
        min_len = min(min_len, len(y_test))
        test_slice = test_slice.iloc[:min_len]

        # Build meta-learner input
        meta_df = pd.DataFrame(index=test_slice.index)
        for model_name, pred_arr in preds.items():
            meta_df[f"pred_{model_name}"] = pred_arr[:min_len]

        build_context_features(
            meta_df, test_slice, context_features,
            max_len=min_len, cast_categorical_to_str=True,
        )

        ensemble_pred = meta_model.predict(meta_df)
        test_mapes.append(float(mape_fn(y_test[:min_len], ensemble_pred)))

    return test_mapes
