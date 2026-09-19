"""Compatibility stub removed; import from :mod:`tnfr.utils.io` instead."""

from __future__ import annotations

from typing import NoReturn

__all__ = ()

def __getattr__(name: str) -> NoReturn:
    """Indicate that :mod:`tnfr.io` no longer exports IO helpers."""

def __dir__() -> tuple[str, ...]:
    """Return an empty set of exports to mirror the removed shim."""
