"""Compatibility wrapper exposing import helpers from :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

import tnfr.utils.init as _utils_init

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

cached_import = _utils_init.cached_import
get_numpy = _utils_init.get_numpy
get_nodonx = _utils_init.get_nodonx
prune_failed_imports = _utils_init.prune_failed_imports

IMPORT_LOG = _utils_init.IMPORT_LOG
EMIT_MAP = _utils_init.EMIT_MAP
WarnOnce = _utils_init.WarnOnce
_FAILED_IMPORT_LIMIT = _utils_init._FAILED_IMPORT_LIMIT
_DEFAULT_CACHE_SIZE = _utils_init._DEFAULT_CACHE_SIZE
_IMPORT_STATE = _utils_init._IMPORT_STATE
_reset_import_state = _utils_init._reset_import_state
_warn_failure = _utils_init._warn_failure
