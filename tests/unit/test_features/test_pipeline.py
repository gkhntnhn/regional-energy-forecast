"""Unit tests for FeaturePipeline."""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from energy_forecast.config import (
    _DEFAULT_REGION,
    PipelineConfig,
    Settings,
    get_default_config,
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


class TestValidateOutput:
    """Tests for _validate_output error branches."""

    def test_duplicate_columns_raises(self) -> None:
        """Duplicate columns in output raise ValueError."""
        settings = Settings(
            region=_DEFAULT_REGION,
            pipeline=PipelineConfig(
                modules=[],
                validate_output=True,
                check_duplicate_columns=True,
            ),
        )
        pipe = FeaturePipeline(settings)
        # Create DataFrame with duplicate columns
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=idx)
        df = pd.concat([df, df[["a"]]], axis=1)  # duplicate 'a'

        with pytest.raises(ValueError, match="Duplicate columns"):
            pipe._validate_output(df)

    def test_raw_epias_remaining_raises(self) -> None:
        """Raw EPIAS columns still present raise ValueError."""
        settings = get_default_config()
        settings = settings.model_copy(
            update={
                "pipeline": PipelineConfig(
                    modules=[],
                    validate_output=True,
                    drop_raw_epias=True,
                    check_duplicate_columns=False,
                ),
            },
        )
        pipe = FeaturePipeline(settings)
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        # Include a raw EPIAS variable name
        raw_var = settings.features.epias.variables[0]
        df = pd.DataFrame({raw_var: [1, 2, 3]}, index=idx)

        with pytest.raises(ValueError, match="Raw EPIAS columns not dropped"):
            pipe._validate_output(df)

    def test_raw_generation_remaining_raises(self) -> None:
        """Raw generation columns still present raise ValueError."""
        settings = get_default_config()
        settings = settings.model_copy(
            update={
                "pipeline": PipelineConfig(
                    modules=[],
                    validate_output=True,
                    drop_raw_epias=True,
                    check_duplicate_columns=False,
                ),
            },
        )
        pipe = FeaturePipeline(settings)
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        # Include a raw generation variable name
        gen_var = settings.features.epias.generation.variables[0]
        df = pd.DataFrame({gen_var: [1, 2, 3]}, index=idx)

        with pytest.raises(ValueError, match="Raw generation columns not dropped"):
            pipe._validate_output(df)

    def test_non_datetime_index_raises(self) -> None:
        """Non-DatetimeIndex raises TypeError."""
        settings = Settings(
            region=_DEFAULT_REGION,
            pipeline=PipelineConfig(
                modules=[],
                validate_output=True,
                check_duplicate_columns=False,
                drop_raw_epias=False,
            ),
        )
        pipe = FeaturePipeline(settings)
        df = pd.DataFrame({"a": [1, 2, 3]})  # RangeIndex

        with pytest.raises(TypeError, match="DatetimeIndex"):
            pipe._validate_output(df)


class TestValidateInput:
    """Tests for Pandera-based input validation (audit P0-2 / X6)."""

    @staticmethod
    def _validating_pipe() -> FeaturePipeline:
        return FeaturePipeline(
            Settings(
                region=_DEFAULT_REGION,
                pipeline=PipelineConfig(
                    modules=[],
                    validate_input=True,
                    validate_output=False,
                ),
            )
        )

    def test_valid_consumption_passes(self) -> None:
        """Schema-conformant input does not raise."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h").rename("datetime")
        df = pd.DataFrame({"consumption": [100.0, 110.0, 120.0]}, index=idx)
        self._validating_pipe()._validate_input(df)

    def test_negative_consumption_raises(self) -> None:
        """Out-of-range consumption surfaces SchemaErrors."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h").rename("datetime")
        df = pd.DataFrame({"consumption": [100.0, -50.0, 120.0]}, index=idx)
        with pytest.raises(pa.errors.SchemaErrors):
            self._validating_pipe()._validate_input(df)

    def test_out_of_range_temperature_raises(self) -> None:
        """Physically impossible temperature surfaces SchemaErrors."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h").rename("datetime")
        df = pd.DataFrame(
            {"temperature_2m": [20.0, 999.0, 22.0]},  # 999 violates le=60
            index=idx,
        )
        with pytest.raises(pa.errors.SchemaErrors):
            self._validating_pipe()._validate_input(df)

    def test_skipped_when_no_relevant_columns(self) -> None:
        """All schemas skip silently for unrelated columns (no false fail)."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h").rename("datetime")
        df = pd.DataFrame({"unrelated_column": [1, 2, 3]}, index=idx)
        # No raise — all 3 schemas have no overlapping columns.
        self._validating_pipe()._validate_input(df)

    def test_validate_input_false_skips(self) -> None:
        """validate_input=False bypasses validation in run()."""
        settings = Settings(
            region=_DEFAULT_REGION,
            pipeline=PipelineConfig(
                modules=[],
                validate_input=False,
                validate_output=False,
            ),
        )
        pipe = FeaturePipeline(settings)
        idx = pd.date_range("2024-01-01", periods=1, freq="h").rename("datetime")
        # Negative consumption — would fail if validation enabled.
        df = pd.DataFrame({"consumption": [-100.0]}, index=idx)
        pipe.run(df)  # no raise — validation skipped
