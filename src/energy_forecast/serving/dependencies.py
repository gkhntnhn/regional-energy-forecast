"""Shared FastAPI dependencies for endpoint injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request


@dataclass
class DBContext:
    """Database context resolved from app.state."""

    use_db: bool
    session_factory: Any  # async_sessionmaker or None


async def get_db_context(request: Request) -> DBContext:
    """Resolve DB context from app.state for dependency injection.

    Returns:
        DBContext with use_db flag and optional session_factory.
    """
    use_db: bool = getattr(request.app.state, "use_db", False)
    session_factory = request.app.state.session_factory if use_db else None
    return DBContext(use_db=use_db, session_factory=session_factory)
