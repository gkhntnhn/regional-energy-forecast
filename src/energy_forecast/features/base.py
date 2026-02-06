"""Base class for all feature engineers (sklearn-compatible)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureEngineer(ABC, BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Abstract base for feature engineering modules.

    All feature engineers must implement fit() and transform().
    Follows sklearn transformer API for pipeline compatibility.

    Args:
        config: Feature-specific YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self, X: pd.DataFrame, y: Any = None) -> BaseFeatureEngineer:
        """Fit the feature engineer (learn parameters if needed).

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate features from input DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with new feature columns added.
        """
        ...
