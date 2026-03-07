"""Tests for ModelRunRepository (sync, using SQLite in-memory)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from energy_forecast.db.base import Base
from energy_forecast.db.models import ModelRunModel
from energy_forecast.db.repositories.model_repo import ModelRunRepository


@pytest.fixture
def sync_session() -> Session:
    """Create a sync SQLite session for model_repo tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        yield session  # type: ignore[misc]


class TestModelRunCRUD:
    """Basic CRUD operations."""

    def test_create_run(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        run = repo.create_run("catboost", n_trials=50, n_splits=12, feature_count=153)
        assert run.id is not None
        assert run.model_type == "catboost"
        assert run.status == "running"
        assert run.n_trials == 50

    def test_complete_run(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        run = repo.create_run("prophet")
        sync_session.commit()

        repo.complete_run(
            run.id,
            metrics={"val_mape": 3.55, "test_mape": 3.62},
            model_path="models/prophet/prophet_2026-03-03",
            hyperparams={"changepoint_prior": 0.0012},
            duration_seconds=7200,
        )
        sync_session.commit()

        updated = repo.get_by_id(run.id)
        assert updated is not None
        assert updated.status == "completed"
        assert updated.val_mape == pytest.approx(3.55)
        assert updated.test_mape == pytest.approx(3.62)
        assert updated.model_path == "models/prophet/prophet_2026-03-03"
        assert updated.duration_seconds == 7200

    def test_fail_run(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        run = repo.create_run("tft")
        sync_session.commit()

        repo.fail_run(run.id, "CUDA out of memory")
        sync_session.commit()

        updated = repo.get_by_id(run.id)
        assert updated is not None
        assert updated.status == "failed"
        assert "CUDA" in (updated.error_message or "")

    def test_get_by_id_not_found(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        assert repo.get_by_id(99999) is None


class TestModelRunQueries:
    """Query operations."""

    def test_get_latest_by_type(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)

        # Create two completed runs — latest has higher ID
        r1 = repo.create_run("catboost")
        sync_session.commit()
        repo.complete_run(r1.id, metrics={"val_mape": 3.0}, model_path="a")
        sync_session.commit()

        r2 = repo.create_run("catboost")
        sync_session.commit()
        repo.complete_run(r2.id, metrics={"val_mape": 2.5}, model_path="b")
        sync_session.commit()

        latest = repo.get_latest_by_type("catboost")
        assert latest is not None
        # r2 is the latest (higher ID = created later)
        assert latest.id == r2.id
        assert latest.val_mape == pytest.approx(2.5)

    def test_get_best_by_type(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)

        r1 = repo.create_run("prophet")
        sync_session.commit()
        repo.complete_run(
            r1.id, metrics={"test_mape": 4.0}, model_path="a"
        )

        r2 = repo.create_run("prophet")
        sync_session.commit()
        repo.complete_run(
            r2.id, metrics={"test_mape": 3.2}, model_path="b"
        )
        sync_session.commit()

        best = repo.get_best_by_type("prophet")
        assert best is not None
        assert best.id == r2.id
        assert best.test_mape == pytest.approx(3.2)

    def test_get_best_no_completed(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        repo.create_run("tft")  # status=running
        sync_session.commit()

        assert repo.get_best_by_type("tft") is None


class TestPromoteWorkflow:
    """Promote/demote operations."""

    def test_promote(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        run = repo.create_run("catboost")
        sync_session.commit()
        repo.complete_run(
            run.id, metrics={"test_mape": 2.5}, model_path="models/cb"
        )
        sync_session.commit()

        repo.promote(run.id)
        sync_session.commit()

        updated = repo.get_by_id(run.id)
        assert updated is not None
        assert updated.is_promoted is True
        assert updated.promoted_at is not None

    def test_promote_demotes_previous(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)

        # First run promoted
        r1 = repo.create_run("catboost")
        sync_session.commit()
        repo.complete_run(r1.id, metrics={}, model_path="a")
        repo.promote(r1.id)
        sync_session.commit()

        # Second run promoted — first should be demoted
        r2 = repo.create_run("catboost")
        sync_session.commit()
        repo.complete_run(r2.id, metrics={}, model_path="b")
        repo.promote(r2.id)
        sync_session.commit()

        old = repo.get_by_id(r1.id)
        new = repo.get_by_id(r2.id)
        assert old is not None and old.is_promoted is False
        assert new is not None and new.is_promoted is True

    def test_get_promoted_models(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)

        r1 = repo.create_run("catboost")
        sync_session.commit()
        repo.complete_run(r1.id, metrics={}, model_path="a")
        repo.promote(r1.id)

        r2 = repo.create_run("prophet")
        sync_session.commit()
        repo.complete_run(r2.id, metrics={}, model_path="b")
        repo.promote(r2.id)
        sync_session.commit()

        promoted = repo.get_promoted_models()
        assert len(promoted) == 2
        types = {r.model_type for r in promoted}
        assert types == {"catboost", "prophet"}

    def test_promote_not_found(self, sync_session: Session) -> None:
        repo = ModelRunRepository(sync_session)
        with pytest.raises(ValueError, match="not found"):
            repo.promote(99999)
