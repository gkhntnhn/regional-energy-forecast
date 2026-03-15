"""Backward-compatible re-export of ``build_context_features``.

The canonical implementation lives in ``energy_forecast.utils.ensemble_helpers``.
This module re-exports it so existing ``training.ensemble_stacking`` imports
continue to work without modification.
"""

from energy_forecast.utils.ensemble_helpers import build_context_features

__all__ = ["build_context_features"]
