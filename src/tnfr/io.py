"""Compatibility proxy that re-exports :mod:`tnfr.utils.io`."""

from __future__ import annotations

from tnfr.utils.io import *  # noqa: F401,F403
from tnfr.utils.io import __all__ as _io_all

__all__ = tuple(_io_all)

del _io_all
