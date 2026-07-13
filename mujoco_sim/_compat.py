"""Helpers for behavior-preserving legacy module aliases."""
from __future__ import annotations

import importlib
import sys
from types import ModuleType


def alias_module(legacy_name: str, canonical_name: str) -> ModuleType:
    """Make an old import path resolve to the canonical module object.

    Returning the same module object preserves class identity and ensures
    monkeypatching a legacy path changes the globals used by the implementation.
    This helper is intentionally not used by ``python -m`` launch shims.
    """
    canonical = importlib.import_module(canonical_name)
    sys.modules[legacy_name] = canonical
    return canonical


def export_module(namespace: dict, canonical_name: str) -> ModuleType:
    """Populate an executable launch shim from a canonical app module."""
    canonical = importlib.import_module(canonical_name)
    namespace.update({
        name: value for name, value in vars(canonical).items()
        if not name.startswith("__")
    })
    return canonical
