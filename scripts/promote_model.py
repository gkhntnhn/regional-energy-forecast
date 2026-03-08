"""Promote a trained model run to final_models/.

Usage:
    # Promote best run for a model type (lowest test MAPE)
    uv run python scripts/promote_model.py --model catboost

    # Promote a specific run by ID
    uv run python scripts/promote_model.py --model catboost --run-id 42
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote model to final_models/")
    parser.add_argument(
        "--model",
        choices=["catboost", "prophet", "tft", "ensemble"],
        required=True,
    )
    parser.add_argument("--run-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_url = os.environ.get("DATABASE_URL_SYNC", "")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set — cannot promote without DB")
        sys.exit(1)

    from energy_forecast.db.engine import (
        create_sync_engine,
        create_sync_session_factory,
    )
    from energy_forecast.db.repositories.model_repo import ModelRunRepository

    engine = create_sync_engine(db_url)
    factory = create_sync_session_factory(engine)

    with factory() as session:
        repo = ModelRunRepository(session)

        if args.run_id is not None:
            run = repo.get_by_id(args.run_id)
            if run is None:
                logger.error("Run #{} not found", args.run_id)
                sys.exit(1)
            if run.model_type != args.model:
                logger.error(
                    "Run #{} is {} not {}",
                    args.run_id,
                    run.model_type,
                    args.model,
                )
                sys.exit(1)
        else:
            run = repo.get_best_by_type(args.model)
            if run is None:
                logger.error("No completed runs found for {}", args.model)
                sys.exit(1)

        if run.status != "completed":
            logger.error("Run #{} status is '{}' (must be 'completed')", run.id, run.status)
            sys.exit(1)

        # Copy model files
        if not run.model_path:
            logger.error("Run #{} has no model_path", run.id)
            sys.exit(1)

        src = Path(run.model_path)
        if not src.exists():
            logger.error("Source path does not exist: {}", src)
            sys.exit(1)

        dst = _PROJECT_ROOT / "final_models" / args.model
        dst.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst / src.name)

        # Mark promoted in DB
        repo.promote(run.id)
        session.commit()

        mape_str = f"{run.test_mape:.2f}%" if run.test_mape else "N/A"
        logger.info(
            "Promoted {} run #{} (test MAPE: {}) to {}",
            args.model,
            run.id,
            mape_str,
            dst,
        )

    engine.dispose()


if __name__ == "__main__":
    main()
