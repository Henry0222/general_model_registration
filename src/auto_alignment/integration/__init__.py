"""Versioned public integration boundary; importing it has no GUI side effects."""
from ..version import __version__ as GENERAL_MODEL_REGISTRATION_VERSION

INTEGRATION_API_VERSION = 1
__all__ = ["INTEGRATION_API_VERSION", "GENERAL_MODEL_REGISTRATION_VERSION"]
