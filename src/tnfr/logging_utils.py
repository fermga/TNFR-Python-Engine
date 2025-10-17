"""Compatibility wrapper exposing logging helpers from :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

import tnfr.utils.init as _utils_init

__all__ = ("_configure_root", "get_logger", "WarnOnce", "warn_once", "_LOGGING_CONFIGURED")

get_logger = _utils_init.get_logger
warn_once = _utils_init.warn_once
WarnOnce = _utils_init.WarnOnce
_configure_root = _utils_init._configure_root


def __getattr__(name: str):
    if name == "_LOGGING_CONFIGURED":
        return getattr(_utils_init, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"_LOGGING_CONFIGURED"})


warnings.warn(
    "'tnfr.logging_utils' is deprecated; import from 'tnfr.utils' instead",
    DeprecationWarning,
    stacklevel=2,
)

_utils_init._reset_logging_state()
