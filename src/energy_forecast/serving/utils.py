"""Serving utility functions."""


def mask_email(email: str) -> str:
    """Mask email for display: 'user@domain.com' → 'use***'."""
    return email[:3] + "***" if email else ""
