"""Alembic migration environment — sync driver (psycopg2)."""

from __future__ import annotations

import os
from logging.config import fileConfig

from dotenv import load_dotenv

load_dotenv()

from alembic import context
from sqlalchemy import create_engine

import energy_forecast.db.models  # noqa: F401, E402  # register all models for autogenerate
from energy_forecast.db.base import Base

# Alembic Config object
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Get sync database URL from environment."""
    url = os.environ.get("DATABASE_URL_SYNC", "")
    if not url:
        msg = (
            "DATABASE_URL_SYNC environment variable is required for migrations. "
            "Example: postgresql+psycopg2://user:pass@localhost:5432/energy_forecast"
        )
        raise RuntimeError(msg)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (SQL script generation)."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    engine = create_engine(get_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
