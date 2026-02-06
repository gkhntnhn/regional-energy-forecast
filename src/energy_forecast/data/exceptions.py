"""Custom exceptions for the data module."""

from __future__ import annotations


class DataError(Exception):
    """Base exception for data module."""


class DataValidationError(DataError):
    """Input data validation failed."""


class EpiasApiError(DataError):
    """EPIAS API request failed."""


class EpiasAuthError(EpiasApiError):
    """EPIAS authentication failed."""


class OpenMeteoApiError(DataError):
    """OpenMeteo API request failed."""
