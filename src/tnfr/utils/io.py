"""Compatibility shims for JSON helpers.

The canonical entry point for JSON serialisation lives under :mod:`tnfr.io`.
This module re-exports those helpers for backwards compatibility and emits a
deprecation warning when imported directly.
"""

from __future__ import annotations

import warnings

from ..io import (
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
    "'tnfr.utils.io' is deprecated and will be removed in a future release; "
    "import JSON helpers from 'tnfr.io' instead.",
    DeprecationWarning,
    stacklevel=2,
)
