"""Dental STL automatic registration and distance visualization."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "AlignmentConfig": ".config",
    "AnalysisOutcome": ".pipeline",
    "run_analysis": ".pipeline",
}

__all__ = list(_EXPORT_MODULES)
__version__ = "1.3.0"


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
