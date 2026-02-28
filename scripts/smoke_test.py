"""End-to-end smoke test for the entire pipeline.

Validates that all components work together:
1. Load consumption data from Excel
2. Fetch EPIAS market data (from cache or API)
3. Fetch OpenMeteo weather data
4. Run feature engineering pipeline
5. Train CatBoost, Prophet, TFT with minimal parameters
6. Run ensemble prediction

Usage:
    python scripts/smoke_test.py --config configs/smoke_test.yaml
    python scripts/smoke_test.py --config configs/smoke_test.yaml --skip-tft
    python scripts/smoke_test.py --config configs/smoke_test.yaml --skip-prophet --skip-tft
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

# Load .env file for EPIAS credentials
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_forecast.config import Settings, load_config  # noqa: E402
from energy_forecast.config.settings import SearchParamConfig  # noqa: E402
from energy_forecast.data.epias_client import EpiasClient  # noqa: E402
from energy_forecast.data.loader import DataLoader  # noqa: E402
from energy_forecast.data.openmeteo_client import OpenMeteoClient  # noqa: E402
from energy_forecast.features.pipeline import FeaturePipeline  # noqa: E402
from energy_forecast.training.catboost_trainer import CatBoostTrainer  # noqa: E402
from energy_forecast.training.ensemble_trainer import EnsembleTrainer  # noqa: E402
from energy_forecast.training.experiment import ExperimentTracker  # noqa: E402
from energy_forecast.training.prophet_trainer import ProphetTrainer  # noqa: E402
from energy_forecast.training.tft_trainer import TFTTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result from a single smoke test step."""

    name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration_seconds: float
    notes: str = ""
    error: str | None = None


@dataclass
class SmokeTestReport:
    """Aggregated smoke test results."""

    steps: list[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        """Check if all non-skipped steps passed."""
        return all(s.status in ("PASS", "SKIP") for s in self.steps)

    @property
    def failed_count(self) -> int:
        """Count failed steps."""
        return sum(1 for s in self.steps if s.status == "FAIL")


# ---------------------------------------------------------------------------
# SmokeTestRunner
# ---------------------------------------------------------------------------


class SmokeTestRunner:
    """Orchestrates end-to-end smoke test.

    Args:
        config_path: Path to smoke test YAML config.
        skip_prophet: Skip Prophet training.
        skip_tft: Skip TFT training.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        config_path: Path,
        *,
        skip_prophet: bool = False,
        skip_tft: bool = False,
        verbose: bool = False,
    ) -> None:
        self.config_path = config_path
        self.skip_prophet = skip_prophet
        self.skip_tft = skip_tft
        self.verbose = verbose
        self.report = SmokeTestReport()

        # Will be populated during setup
        self.settings: Settings | None = None
        self.smoke_config: dict[str, Any] = {}
        self.consumption_df: pd.DataFrame | None = None
        self.epias_df: pd.DataFrame | None = None
        self.weather_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None

    def run(self) -> SmokeTestReport:
        """Execute full smoke test pipeline."""
        start = time.monotonic()

        # Step 0: Load configs
        self._run_step("0. Load Config", self._step_load_config)

        # Step 1: Load Excel data
        self._run_step("1. Load Excel Data", self._step_load_excel)

        # Step 2: Fetch EPIAS data
        self._run_step("2. Fetch EPIAS Data", self._step_fetch_epias)

        # Step 3: Fetch OpenMeteo data
        self._run_step("3. Fetch OpenMeteo Data", self._step_fetch_openmeteo)

        # Step 4: Run feature pipeline
        self._run_step("4. Feature Pipeline", self._step_feature_pipeline)

        # Step 5: Train CatBoost
        self._run_step("5. Train CatBoost", self._step_train_catboost)

        # Step 6: Train Prophet
        if self.skip_prophet:
            self.report.steps.append(
                StepResult("6. Train Prophet", "SKIP", 0.0, "Skipped via --skip-prophet")
            )
        else:
            self._run_step("6. Train Prophet", self._step_train_prophet)

        # Step 7: Train TFT
        if self.skip_tft:
            self.report.steps.append(
                StepResult("7. Train TFT", "SKIP", 0.0, "Skipped via --skip-tft")
            )
        else:
            self._run_step("7. Train TFT", self._step_train_tft)

        # Step 8: Ensemble prediction
        self._run_step("8. Ensemble Predict", self._step_ensemble_predict)

        self.report.total_duration_seconds = time.monotonic() - start
        return self.report

    def _run_step(
        self,
        name: str,
        step_fn: Any,
    ) -> None:
        """Execute a step with timing and error handling."""
        logger.info("=" * 60)
        logger.info("Starting: {}", name)
        logger.info("=" * 60)

        start = time.monotonic()
        try:
            notes = step_fn()
            duration = time.monotonic() - start
            self.report.steps.append(StepResult(name, "PASS", duration, notes or ""))
            logger.info("{} completed in {:.1f}s", name, duration)
        except Exception as e:
            duration = time.monotonic() - start
            self.report.steps.append(StepResult(name, "FAIL", duration, error=str(e)))
            logger.error("{} FAILED: {}", name, e)
            if self.verbose:
                logger.exception("Full traceback:")

    # -----------------------------------------------------------------------
    # Step implementations
    # -----------------------------------------------------------------------

    def _step_load_config(self) -> str:
        """Load base config and merge smoke test overrides."""
        # Load base settings
        configs_dir = PROJECT_ROOT / "configs"
        self.settings = load_config(configs_dir)

        # Load smoke test overrides
        with open(self.config_path, encoding="utf-8") as f:
            self.smoke_config = yaml.safe_load(f)

        # Apply smoke test overrides to hyperparameters
        self._apply_hyperparameter_overrides()

        return f"Loaded from {configs_dir}"

    def _apply_hyperparameter_overrides(self) -> None:
        """Apply smoke test config overrides to settings."""
        if self.settings is None:
            return

        hp = self.settings.hyperparameters

        # Override CatBoost
        if "catboost" in self.smoke_config:
            cb_override = self.smoke_config["catboost"]
            cb_config = hp.catboost
            if "n_trials" in cb_override:
                object.__setattr__(cb_config, "n_trials", cb_override["n_trials"])
            if "search_space" in cb_override:
                new_space = {
                    k: SearchParamConfig(**v) for k, v in cb_override["search_space"].items()
                }
                object.__setattr__(cb_config, "search_space", new_space)

        # Override Prophet
        if "prophet" in self.smoke_config:
            p_override = self.smoke_config["prophet"]
            p_config = hp.prophet
            if "n_trials" in p_override:
                object.__setattr__(p_config, "n_trials", p_override["n_trials"])
            if "search_space" in p_override:
                new_space = {
                    k: SearchParamConfig(**v) for k, v in p_override["search_space"].items()
                }
                object.__setattr__(p_config, "search_space", new_space)

        # Override TFT
        if "tft" in self.smoke_config:
            tft_override = self.smoke_config["tft"]
            tft_config = hp.tft
            if "n_trials" in tft_override:
                object.__setattr__(tft_config, "n_trials", tft_override["n_trials"])
            if "search_space" in tft_override:
                new_space = {
                    k: SearchParamConfig(**v) for k, v in tft_override["search_space"].items()
                }
                object.__setattr__(tft_config, "search_space", new_space)

            # Override TFT training params
            if "training" in tft_override:
                train_ovr = tft_override["training"]
                tft_model_config = self.settings.tft
                train_cfg = tft_model_config.training
                for key, val in train_ovr.items():
                    if hasattr(train_cfg, key):
                        object.__setattr__(train_cfg, key, val)

        # Override cross-validation
        if "cross_validation" in self.smoke_config:
            cv_override = self.smoke_config["cross_validation"]
            cv_config = hp.cross_validation
            for key, val in cv_override.items():
                if hasattr(cv_config, key):
                    object.__setattr__(cv_config, key, val)

    def _step_load_excel(self) -> str:
        """Load consumption data from Excel."""
        if self.settings is None:
            raise RuntimeError("Settings not loaded")

        # Get Excel path from smoke config or default
        excel_path = self.smoke_config.get("data", {}).get(
            "excel_path", "data/raw/Consumption_Input_Format.xlsx"
        )
        excel_path = PROJECT_ROOT / excel_path

        loader = DataLoader(self.settings.data_loader)
        self.consumption_df = loader.load_excel(excel_path)

        return f"{len(self.consumption_df)} rows"

    def _step_fetch_epias(self) -> str:
        """Fetch EPIAS market data."""
        if self.settings is None or self.consumption_df is None:
            raise RuntimeError("Prerequisites not ready")

        # Get date range from consumption data
        start_date = self.consumption_df.index.min().strftime("%Y-%m-%d")
        end_date = self.consumption_df.index.max().strftime("%Y-%m-%d")

        # Get credentials from environment
        username = os.getenv("EPIAS_USERNAME", "")
        password = os.getenv("EPIAS_PASSWORD", "")

        if not username or not password:
            # Try to use cached data only
            logger.warning("EPIAS credentials not set, using cache only")
            cache_dir = Path(self.settings.epias_api.cache_dir)
            years = range(
                self.consumption_df.index.min().year,
                self.consumption_df.index.max().year + 1,
            )
            dfs = []
            for year in years:
                cache_path = cache_dir / f"epias_market_{year}.parquet"
                if cache_path.exists():
                    dfs.append(pd.read_parquet(cache_path))
            if dfs:
                self.epias_df = pd.concat(dfs)
                self.epias_df = self.epias_df.sort_index()
                return f"{len(self.epias_df)} rows (from cache)"
            raise RuntimeError("No EPIAS cache and no credentials")

        with EpiasClient(username, password, self.settings.epias_api) as client:
            self.epias_df = client.fetch(start_date, end_date)

        return f"{len(self.epias_df)} rows"

    def _step_fetch_openmeteo(self) -> str:
        """Fetch OpenMeteo weather data."""
        if self.settings is None or self.consumption_df is None:
            raise RuntimeError("Prerequisites not ready")

        # Get date range from consumption data
        start_date = self.consumption_df.index.min().strftime("%Y-%m-%d")
        end_date = self.consumption_df.index.max().strftime("%Y-%m-%d")

        with OpenMeteoClient(
            self.settings.openmeteo,
            self.settings.region,
            self.settings.project.timezone,
        ) as client:
            self.weather_df = client.fetch_historical(start_date, end_date)

        return f"{len(self.weather_df)} rows"

    def _step_feature_pipeline(self) -> str:
        """Run feature engineering pipeline."""
        if (
            self.settings is None
            or self.consumption_df is None
            or self.epias_df is None
            or self.weather_df is None
        ):
            raise RuntimeError("Prerequisites not ready")

        # Merge all data sources
        merged = self.consumption_df.copy()

        # Align EPIAS data
        epias_aligned = self.epias_df.reindex(merged.index)
        for col in self.epias_df.columns:
            merged[col] = epias_aligned[col]

        # Align weather data
        weather_aligned = self.weather_df.reindex(merged.index)
        for col in self.weather_df.columns:
            merged[col] = weather_aligned[col]

        # Forward-fill ONLY weather columns (consumption/EPIAS NaN must be preserved
        # to avoid data leakage — see CLAUDE.md and prepare_dataset.py:588-596).
        weather_prefixes = (
            "temperature", "humidity", "dew_point", "apparent_temperature",
            "precipitation", "snow_depth", "weather_code", "surface_pressure",
            "wind_speed", "wind_direction", "shortwave_radiation",
        )
        weather_cols = [c for c in merged.columns if c.startswith(weather_prefixes)]
        if weather_cols:
            merged[weather_cols] = merged[weather_cols].ffill().bfill()

        # Run feature pipeline
        pipeline = FeaturePipeline(self.settings)
        self.features_df = pipeline.run(merged)

        n_features = len(self.features_df.columns)
        return f"{len(self.features_df)} rows, {n_features} features"

    def _step_train_catboost(self) -> str:
        """Train CatBoost with minimal parameters."""
        if self.settings is None or self.features_df is None:
            raise RuntimeError("Prerequisites not ready")

        tracker = ExperimentTracker(enabled=False)
        trainer = CatBoostTrainer(self.settings, tracker)
        result = trainer.run(self.features_df)

        mape = result.training_result.avg_val_mape
        return f"MAPE: {mape:.2f}%"

    def _step_train_prophet(self) -> str:
        """Train Prophet with minimal parameters."""
        if self.settings is None or self.features_df is None:
            raise RuntimeError("Prerequisites not ready")

        tracker = ExperimentTracker(enabled=False)
        trainer = ProphetTrainer(self.settings, tracker)
        result = trainer.run(self.features_df)

        mape = result.training_result.avg_val_mape
        return f"MAPE: {mape:.2f}%"

    def _step_train_tft(self) -> str:
        """Train TFT with minimal parameters."""
        if self.settings is None or self.features_df is None:
            raise RuntimeError("Prerequisites not ready")

        tracker = ExperimentTracker(enabled=False)
        trainer = TFTTrainer(self.settings, tracker)
        result = trainer.run(self.features_df)

        mape = result.training_result.avg_val_mape
        return f"MAPE: {mape:.2f}%"

    def _step_ensemble_predict(self) -> str:
        """Run ensemble training and prediction."""
        if self.settings is None or self.features_df is None:
            raise RuntimeError("Prerequisites not ready")

        # Determine active models based on skip flags
        active_models = ["catboost"]
        if not self.skip_prophet:
            active_models.append("prophet")
        if not self.skip_tft:
            active_models.append("tft")

        tracker = ExperimentTracker(enabled=False)
        trainer = EnsembleTrainer(
            self.settings,
            tracker,
            active_models_override=active_models,
        )
        result = trainer.run(self.features_df)

        mape = result.training_result.avg_val_mape
        weights = result.training_result.optimized_weights
        weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in weights.items())
        return f"MAPE: {mape:.2f}%, weights: {weights_str}"

    def print_summary(self) -> None:
        """Print formatted summary table (ASCII-safe for Windows)."""
        print("\n")
        print("+" + "=" * 78 + "+")
        print("|" + " " * 25 + "SMOKE TEST RESULTS" + " " * 35 + "|")
        print("+" + "=" * 78 + "+")
        print(f"| {'Step':<25} | {'Status':^7} | {'Duration':^10} | {'Notes':<25} |")
        print("+" + "-" * 27 + "+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 27 + "+")

        for step in self.report.steps:
            status_icon = {
                "PASS": "[PASS]",
                "FAIL": "[FAIL]",
                "SKIP": "[SKIP]",
            }.get(step.status, step.status)

            duration_str = f"{step.duration_seconds:.1f}s"
            notes = step.notes[:25] if step.notes else ""
            if step.error:
                notes = step.error[:25]

            name = step.name[:25]
            print(f"| {name:<25} | {status_icon:^7} | {duration_str:>10} | {notes:<25} |")

        print("+" + "-" * 27 + "+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 27 + "+")

        if self.report.all_passed:
            overall = "[OK] ALL PASSED"
        else:
            overall = f"[X] {self.report.failed_count} FAILED"
        total_time = f"Total: {self.report.total_duration_seconds:.1f}s"
        print(f"| OVERALL: {overall:<30} {total_time:>35} |")
        print("+" + "=" * 78 + "+")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test for the energy forecast pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke_test.yaml"),
        help="Path to smoke test config (default: configs/smoke_test.yaml)",
    )
    parser.add_argument(
        "--skip-prophet",
        action="store_true",
        help="Skip Prophet training (faster CI)",
    )
    parser.add_argument(
        "--skip-tft",
        action="store_true",
        help="Skip TFT training (much faster CI)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose error logging",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point. Returns exit code 0 (pass) or 1 (fail)."""
    args = parse_args()

    # Configure logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )

    logger.info("Starting smoke test")
    logger.info("Config: {}", args.config)
    logger.info("Skip Prophet: {}, Skip TFT: {}", args.skip_prophet, args.skip_tft)

    runner = SmokeTestRunner(
        config_path=args.config,
        skip_prophet=args.skip_prophet,
        skip_tft=args.skip_tft,
        verbose=args.verbose,
    )

    report = runner.run()
    runner.print_summary()

    if report.all_passed:
        logger.info("Smoke test PASSED")
        return 0
    else:
        logger.error("Smoke test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
