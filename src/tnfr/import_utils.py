"""Compatibility wrapper exposing import helpers from :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

from .utils.init import (
    EMIT_MAP,
    IMPORT_LOG,
    WarnOnce,
    _FAILED_IMPORT_LIMIT,
    _DEFAULT_CACHE_SIZE,
    _IMPORT_STATE,
    _reset_import_state,
    _warn_failure,
    cached_import,
    get_numpy,
    get_nodonx,
    prune_failed_imports,
)

__all__ = (
    "cached_import",
    "get_numpy",
    "get_nodonx",
    "prune_failed_imports",
    "IMPORT_LOG",
)

warnings.warn(
    "'tnfr.import_utils' is deprecated; import from 'tnfr.utils' instead",
    DeprecationWarning,
    stacklevel=2,
)

_reset_import_state()
