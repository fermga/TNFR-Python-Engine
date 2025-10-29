"""Compatibility layer for :mod:`tnfr.helpers.numeric`."""

from __future__ import annotations

import warnings

from ..utils.numeric import *  # noqa: F401,F403 - re-export compatibility
from ..utils.numeric import __all__ as _UTILS_ALL

warnings.warn(
    "'tnfr.helpers.numeric' is deprecated and will be removed in a future release; "
    "import from 'tnfr.utils.numeric' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _UTILS_ALL
