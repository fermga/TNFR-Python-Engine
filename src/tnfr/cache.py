"""Compatibility proxy that re-exports :mod:`tnfr.utils.cache`."""

from __future__ import annotations

from tnfr.utils.cache import *  # noqa: F401,F403
from tnfr.utils.cache import __all__ as _cache_all

__all__ = tuple(_cache_all)

del _cache_all
