"""Feature pipeline orchestrator."""

from __future__ import annotations

from typing import Any

import pandas as pd


class FeaturePipeline:
    """Orchestrates all feature engineering modules.

    Runs all configured feature engineers in sequence and
    validates the output (no duplicates, no raw EPIAS values).

    Args:
        config: Pipeline configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full feature pipeline.

        Args:
            df: Raw input DataFrame with consumption, weather, EPIAS data.

        Returns:
            Feature-engineered DataFrame ready for modeling.
        """
        raise NotImplementedError
