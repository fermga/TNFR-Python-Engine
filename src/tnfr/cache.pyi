"""Compatibility stub removed; import from :mod:`tnfr.utils.cache` instead."""

from __future__ import annotations

from typing import NoReturn

__all__ = ()

def __getattr__(name: str) -> NoReturn:
    """Indicate that :mod:`tnfr.cache` no longer exports cache helpers."""

def __dir__() -> tuple[str, ...]:
    """Return an empty set of exports to mirror the removed shim."""
