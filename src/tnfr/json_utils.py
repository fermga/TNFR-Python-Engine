"""Compatibility wrapper exposing JSON helpers from :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

from .utils.io import (
    DEFAULT_PARAMS,
    JsonDumpsParams,
    clear_orjson_param_warnings,
    json_dumps,
)

__all__ = (
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "json_dumps",
    "clear_orjson_param_warnings",
)

warnings.warn(
    "'tnfr.json_utils' is deprecated; import from 'tnfr.utils.io' instead",
    DeprecationWarning,
    stacklevel=2,
)
