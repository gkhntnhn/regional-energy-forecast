"""Feature pipeline orchestrator."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from loguru import logger

from energy_forecast.config.settings import Settings
from energy_forecast.features.base import BaseFeatureEngineer
from energy_forecast.features.calendar import CalendarFeatureEngineer
from energy_forecast.features.consumption import ConsumptionFeatureEngineer
from energy_forecast.features.epias import EpiasFeatureEngineer
from energy_forecast.features.solar import SolarFeatureEngineer
from energy_forecast.features.weather import WeatherFeatureEngineer


class FeaturePipeline:
    """Orchestrates all feature engineering modules.

    Runs configured feature engineers in sequence and validates
    the output (no duplicates, no raw EPIAS values).

    Args:
        config: Full ``Settings`` object.
    """

    MODULE_MAP: ClassVar[dict[str, type[BaseFeatureEngineer]]] = {
        "calendar": CalendarFeatureEngineer,
        "consumption": ConsumptionFeatureEngineer,
        "weather": WeatherFeatureEngineer,
        "solar": SolarFeatureEngineer,
        "epias": EpiasFeatureEngineer,
    }

    def __init__(self, config: Settings) -> None:
        self._settings = config
        self._pipeline_cfg = config.pipeline
        self._engineers: list[tuple[str, BaseFeatureEngineer]] = []
        self._build_engineers()

    def _build_engineers(self) -> None:
        """Instantiate engineers from config module list."""
        for name in self._pipeline_cfg.modules:
            if name not in self.MODULE_MAP:
                msg = f"Unknown feature module: {name}"
                raise ValueError(msg)
            cls = self.MODULE_MAP[name]
            feature_cfg: Any = getattr(self._settings.features, name)
            engineer = cls(feature_cfg.model_dump())
            self._engineers.append((name, engineer))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full feature pipeline.

        Args:
            df: Raw input DataFrame with consumption, weather, EPIAS data.

        Returns:
            Feature-engineered DataFrame ready for modeling.
        """
        result = df.copy()
        for name, engineer in self._engineers:
            logger.info("Running {} feature engineer", name)
            result = engineer.fit_transform(result)
            logger.info("{} complete — shape: {}", name, result.shape)

        if self._pipeline_cfg.validate_output:
            self._validate_output(result)

        return result

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Post-pipeline validation checks."""
        # Check duplicate columns
        if self._pipeline_cfg.check_duplicate_columns:
            dupes = df.columns[df.columns.duplicated()].tolist()
            if dupes:
                msg = f"Duplicate columns in pipeline output: {dupes}"
                raise ValueError(msg)

        # Check raw EPIAS columns are dropped
        if self._pipeline_cfg.drop_raw_epias:
            raw_epias = self._settings.features.epias.variables
            remaining = [v for v in raw_epias if v in df.columns]
            if remaining:
                msg = f"Raw EPIAS columns not dropped: {remaining}"
                raise ValueError(msg)

            # Check raw generation columns are dropped
            gen_vars = self._settings.features.epias.generation.variables
            remaining_gen = [v for v in gen_vars if v in df.columns]
            if remaining_gen:
                msg = f"Raw generation columns not dropped: {remaining_gen}"
                raise ValueError(msg)

        # Check DatetimeIndex preserved
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "Pipeline output must have DatetimeIndex"
            raise TypeError(msg)

    def get_feature_names(self) -> list[str]:
        """Return list of module names in the pipeline."""
        return [name for name, _ in self._engineers]
