"""Consumption data loader from Excel files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from energy_forecast.config.settings import DataLoaderConfig
from energy_forecast.data.exceptions import DataValidationError


class DataLoader:
    """Loads and validates consumption Excel data.

    Args:
        config: Data loader configuration from settings.
    """

    def __init__(self, config: DataLoaderConfig) -> None:
        self.config = config

    def load_excel(self, path: Path) -> pd.DataFrame:
        """Load consumption data from Excel file.

        Reads the Excel file, validates structure and values, merges
        date+time columns into a DatetimeIndex, and creates a continuous
        hourly time series.

        Args:
            path: Path to .xlsx file with date, time, consumption columns.

        Returns:
            DataFrame with hourly DatetimeIndex and ``consumption`` column.
            Missing hours are filled with NaN.

        Raises:
            FileNotFoundError: If the file does not exist.
            DataValidationError: If data fails validation checks.
        """
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        logger.info("Loading Excel file: {}", path)
        df = pd.read_excel(path, engine="openpyxl")

        self._validate_columns(df)
        df = self._rename_columns(df)
        self._validate_raw_data(df)
        df = self._merge_datetime(df)
        df = self._create_continuous_index(df)

        logger.info(
            "Loaded {} rows, range {} to {}",
            len(df),
            df.index.min(),
            df.index.max(),
        )
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check that required columns exist in the raw DataFrame."""
        col_cfg = self.config.excel
        required = {col_cfg.date, col_cfg.time, col_cfg.consumption}
        missing = required - set(df.columns)
        if missing:
            msg = f"Missing required columns: {missing}"
            raise DataValidationError(msg)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard names (date, time, consumption)."""
        col_cfg = self.config.excel
        rename_map: dict[str, str] = {}
        if col_cfg.date != "date":
            rename_map[col_cfg.date] = "date"
        if col_cfg.time != "time":
            rename_map[col_cfg.time] = "time"
        if col_cfg.consumption != "consumption":
            rename_map[col_cfg.consumption] = "consumption"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df[["date", "time", "consumption"]].copy()

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """Validate raw data values."""
        if len(df) == 0:
            msg = "Excel file contains no data rows"
            raise DataValidationError(msg)

        time_min, time_max = self.config.time_range
        invalid_time = df[~df["time"].between(time_min, time_max)]
        if len(invalid_time) > 0:
            bad_vals = invalid_time["time"].unique().tolist()
            msg = f"Invalid time values (expected {time_min}-{time_max}): {bad_vals}"
            raise DataValidationError(msg)

        val_cfg = self.config.validation
        neg = df[df["consumption"] < val_cfg.min_consumption]
        if len(neg) > 0:
            msg = (
                f"Consumption values below minimum ({val_cfg.min_consumption}): "
                f"found {len(neg)} rows"
            )
            raise DataValidationError(msg)

        over = df[df["consumption"] > val_cfg.max_consumption]
        if len(over) > 0:
            msg = (
                f"Consumption values above maximum ({val_cfg.max_consumption}): "
                f"found {len(over)} rows"
            )
            raise DataValidationError(msg)

        # Check missing ratio (NaN in consumption)
        missing_ratio = df["consumption"].isna().sum() / len(df)
        if missing_ratio > val_cfg.max_missing_ratio:
            msg = (
                f"Missing consumption ratio {missing_ratio:.2%} exceeds "
                f"maximum {val_cfg.max_missing_ratio:.2%}"
            )
            raise DataValidationError(msg)

    def _merge_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge date + time columns into single datetime column.

        Formula: ``pd.to_datetime(date) + pd.to_timedelta(time, unit='h')``
        """
        dt = pd.to_datetime(df["date"], format=self.config.date_format)
        dt = dt + pd.to_timedelta(df["time"], unit="h")
        result = pd.DataFrame({"consumption": df["consumption"].values}, index=dt)
        result.index.name = "datetime"
        return result

    def _create_continuous_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps to create continuous hourly time series."""
        full_idx = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq="h",
            name="datetime",
        )
        df = df[~df.index.duplicated(keep="first")]
        return df.reindex(full_idx)

    def extend_for_forecast(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 48,
    ) -> pd.DataFrame:
        """Extend DataFrame with NaN rows for forecast period.

        Args:
            df: DataFrame with DatetimeIndex.
            horizon_hours: Number of hours to extend (default 48).

        Returns:
            Extended DataFrame with NaN consumption for forecast hours.
        """
        last_ts = df.index.max()
        forecast_idx = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=horizon_hours,
            freq="h",
            name="datetime",
        )
        forecast_df = pd.DataFrame(
            {"consumption": [float("nan")] * horizon_hours},
            index=forecast_idx,
        )
        return pd.concat([df, forecast_df])
