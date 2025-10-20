"""Backward compatibility shim for glyph constants."""

from __future__ import annotations

import warnings

from .config.constants import *  # noqa: F401,F403 - re-export legacy API
from .config.constants import __all__ as _CONFIG_ALL

warnings.warn(
    "'tnfr.constants_glyphs' is deprecated; use 'tnfr.config.constants' instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _CONFIG_ALL
