"""Tests for rate_limit module."""

from __future__ import annotations

from slowapi import Limiter

from energy_forecast.serving.rate_limit import limiter


class TestRateLimiter:
    """Tests for the shared rate limiter instance."""

    def test_limiter_is_limiter_instance(self) -> None:
        """limiter should be a slowapi Limiter instance."""
        assert isinstance(limiter, Limiter)

    def test_limiter_has_key_func(self) -> None:
        """limiter should have a key function set."""
        assert limiter._key_func is not None
