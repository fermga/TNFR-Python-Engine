"""Compatibility wrapper exposing logging helpers from :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

from .utils.init import (
    WarnOnce,
    _LOGGING_CONFIGURED,
    _reset_logging_state,
    _configure_root,
    get_logger,
    warn_once,
)

__all__ = ("_configure_root", "get_logger", "WarnOnce", "warn_once")

warnings.warn(
    "'tnfr.logging_utils' is deprecated; import from 'tnfr.utils' instead",
    DeprecationWarning,
    stacklevel=2,
)

_reset_logging_state()
