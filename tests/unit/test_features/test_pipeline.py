"""Unit tests for FeaturePipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecast.config import get_default_config
from energy_forecast.config.settings import (
    _DEFAULT_REGION,
    PipelineConfig,
    Settings,
)
from energy_forecast.features.pipeline import FeaturePipeline


@pytest.fixture()
def default_settings() -> Settings:
    """Default Settings without YAML files."""
    return get_default_config()


@pytest.fixture()
def pipeline(default_settings: Settings) -> FeaturePipeline:
    """FeaturePipeline with all 5 modules enabled."""
    return FeaturePipeline(default_settings)


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_runs_all_modules(
        self,
        pipeline: FeaturePipeline,
        sample_full_df: pd.DataFrame,
    ) -> None:
        """Pipeline runs all 5 modules without error."""
        result = pipeline.run(sample_full_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_full_df)

    def test_adds_features(
        self,
        pipeline: FeaturePipeline,
        sample_full_df: pd.DataFrame,
    ) -> None:
        """Output has more columns than input after feature engineering."""
        result = pipeline.run(sample_full_df)
        assert result.shape[1] > sample_full_df.shape[1]

    def test_preserves_index(
        self,
        pipeline: FeaturePipeline,
        sample_full_df: pd.DataFrame,
    ) -> None:
        """DatetimeIndex is preserved through the pipeline."""
        result = pipeline.run(sample_full_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_duplicate_columns(
        self,
        pipeline: FeaturePipeline,
        sample_full_df: pd.DataFrame,
    ) -> None:
        """Pipeline output has no duplicate column names."""
        result = pipeline.run(sample_full_df)
        duplicated = result.columns[result.columns.duplicated()].tolist()
        assert len(duplicated) == 0, f"Duplicate columns found: {duplicated}"

    def test_drops_raw_epias(
        self,
        pipeline: FeaturePipeline,
        sample_full_df: pd.DataFrame,
    ) -> None:
        """Raw EPIAS columns are not in pipeline output."""
        raw_epias_cols = [
            "Real_Time_Consumption",
            "DAM_Purchase",
            "Bilateral_Agreement_Purchase",
            "Load_Forecast",
        ]
        result = pipeline.run(sample_full_df)
        for col in raw_epias_cols:
            assert col not in result.columns, f"Raw EPIAS column {col} not dropped"

    def test_unknown_module_raises(self) -> None:
        """Unknown module name raises ValueError."""
        settings = Settings(
            region=_DEFAULT_REGION,
            pipeline=PipelineConfig(
                modules=["calendar", "nonexistent_module"],
                validate_output=False,
            ),
        )
        with pytest.raises(ValueError, match="Unknown feature module"):
            FeaturePipeline(settings)

    def test_feature_names(self, pipeline: FeaturePipeline) -> None:
        """get_feature_names returns list of module names."""
        names = pipeline.get_feature_names()
        assert isinstance(names, list)
        assert names == ["calendar", "consumption", "weather", "solar", "epias"]

    def test_empty_modules(self, sample_full_df: pd.DataFrame) -> None:
        """Empty module list returns the same DataFrame unchanged."""
        settings = Settings(
            region=_DEFAULT_REGION,
            pipeline=PipelineConfig(
                modules=[],
                validate_output=False,
            ),
        )
        pipe = FeaturePipeline(settings)
        result = pipe.run(sample_full_df)
        assert result.shape == sample_full_df.shape
        assert list(result.columns) == list(sample_full_df.columns)
