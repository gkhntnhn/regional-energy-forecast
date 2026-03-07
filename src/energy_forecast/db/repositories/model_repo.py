"""Model run repository — sync CRUD for training pipeline.

Uses sync SQLAlchemy sessions (psycopg2) because the training pipeline
is fully synchronous. This avoids unnecessary async complexity in CLI tools.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from energy_forecast.db.models import ModelRunModel
from energy_forecast.utils import TZ_ISTANBUL


class ModelRunRepository:
    """Data access layer for model_runs table (sync)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(
        self,
        model_type: str,
        *,
        n_trials: int | None = None,
        n_splits: int | None = None,
        feature_count: int | None = None,
    ) -> ModelRunModel:
        """Record the start of a training run."""
        run = ModelRunModel(
            model_type=model_type,
            status="running",
            n_trials=n_trials,
            n_splits=n_splits,
            feature_count=feature_count,
            started_at=datetime.now(tz=TZ_ISTANBUL),
        )
        self._session.add(run)
        self._session.flush()
        return run

    def complete_run(
        self,
        run_id: int,
        *,
        metrics: dict[str, float],
        model_path: str,
        hyperparams: dict[str, Any] | None = None,
        duration_seconds: int | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        """Mark a training run as completed with results."""
        now = datetime.now(tz=TZ_ISTANBUL)
        values: dict[str, Any] = {
            "status": "completed",
            "completed_at": now,
            "model_path": model_path,
            "hyperparameters": hyperparams,
            "val_mape": metrics.get("val_mape"),
            "test_mape": metrics.get("test_mape"),
            "val_rmse": metrics.get("val_rmse"),
            "test_rmse": metrics.get("test_rmse"),
        }
        if duration_seconds is not None:
            values["duration_seconds"] = duration_seconds
        if mlflow_run_id is not None:
            values["run_id"] = mlflow_run_id
        self._session.execute(
            update(ModelRunModel)
            .where(ModelRunModel.id == run_id)
            .values(**values)
        )
        self._session.flush()

    def fail_run(self, run_id: int, error: str) -> None:
        """Mark a training run as failed."""
        self._session.execute(
            update(ModelRunModel)
            .where(ModelRunModel.id == run_id)
            .values(
                status="failed",
                completed_at=datetime.now(tz=TZ_ISTANBUL),
                error_message=error[:2000],
            )
        )
        self._session.flush()

    def get_by_id(self, run_id: int) -> ModelRunModel | None:
        """Get a specific run by ID."""
        result = self._session.execute(
            select(ModelRunModel).where(ModelRunModel.id == run_id)
        )
        return result.scalar_one_or_none()

    def get_latest_by_type(self, model_type: str) -> ModelRunModel | None:
        """Get the most recent completed run for a model type."""
        result = self._session.execute(
            select(ModelRunModel)
            .where(ModelRunModel.model_type == model_type)
            .where(ModelRunModel.status == "completed")
            .order_by(ModelRunModel.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def get_best_by_type(self, model_type: str) -> ModelRunModel | None:
        """Get the completed run with lowest test MAPE for a model type."""
        result = self._session.execute(
            select(ModelRunModel)
            .where(ModelRunModel.model_type == model_type)
            .where(ModelRunModel.status == "completed")
            .where(ModelRunModel.test_mape.is_not(None))
            .order_by(ModelRunModel.test_mape.asc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def get_promoted_models(self) -> list[ModelRunModel]:
        """Get all currently promoted models."""
        result = self._session.execute(
            select(ModelRunModel)
            .where(ModelRunModel.is_promoted.is_(True))
            .order_by(ModelRunModel.model_type)
        )
        return list(result.scalars().all())

    def promote(self, run_id: int) -> None:
        """Mark a run as promoted to final_models/."""
        # First, get the model type to demote any previous promotion
        run = self.get_by_id(run_id)
        if run is None:
            msg = f"Run {run_id} not found"
            raise ValueError(msg)

        # Demote previous promoted run of same type
        self._session.execute(
            update(ModelRunModel)
            .where(ModelRunModel.model_type == run.model_type)
            .where(ModelRunModel.is_promoted.is_(True))
            .values(is_promoted=False, promoted_at=None)
        )

        # Promote the new run
        self._session.execute(
            update(ModelRunModel)
            .where(ModelRunModel.id == run_id)
            .values(
                is_promoted=True,
                promoted_at=datetime.now(tz=TZ_ISTANBUL),
            )
        )
        self._session.flush()
