"""Shared utility helpers exposed under :mod:`tnfr.utils`."""

from __future__ import annotations

from . import init as _init
from .data import (
    MAX_MATERIALIZE_DEFAULT,
    STRING_TYPES,
    convert_value,
    ensure_collection,
    flatten_structure,
    is_non_string_sequence,
    mix_groups,
    negative_weights_warn_once,
    normalize_counter,
    normalize_materialize_limit,
    normalize_weights,
)
from .graph import (
    get_graph,
    get_graph_mapping,
    mark_dnfr_prep_dirty,
    supports_add_edge,
)
from .io import (
    DEFAULT_PARAMS,
    JsonDumpsParams,
    clear_orjson_param_warnings,
    json_dumps,
)

__all__ = (
    "IMPORT_LOG",
    "WarnOnce",
    "cached_import",
    "get_logger",
    "get_nodonx",
    "get_numpy",
    "prune_failed_imports",
    "warn_once",
    "convert_value",
    "normalize_weights",
    "normalize_counter",
    "normalize_materialize_limit",
    "ensure_collection",
    "flatten_structure",
    "is_non_string_sequence",
    "STRING_TYPES",
    "MAX_MATERIALIZE_DEFAULT",
    "negative_weights_warn_once",
    "mix_groups",
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "json_dumps",
    "clear_orjson_param_warnings",
    "_configure_root",
    "_LOGGING_CONFIGURED",
    "_reset_logging_state",
    "_reset_import_state",
    "_IMPORT_STATE",
    "_warn_failure",
    "_FAILED_IMPORT_LIMIT",
    "_DEFAULT_CACHE_SIZE",
    "EMIT_MAP",
)

WarnOnce = _init.WarnOnce
cached_import = _init.cached_import
get_logger = _init.get_logger
get_nodonx = _init.get_nodonx
get_numpy = _init.get_numpy
prune_failed_imports = _init.prune_failed_imports
warn_once = _init.warn_once
_configure_root = _init._configure_root
_reset_logging_state = _init._reset_logging_state
_reset_import_state = _init._reset_import_state
_warn_failure = _init._warn_failure
_FAILED_IMPORT_LIMIT = _init._FAILED_IMPORT_LIMIT
_DEFAULT_CACHE_SIZE = _init._DEFAULT_CACHE_SIZE
EMIT_MAP = _init.EMIT_MAP

_DYNAMIC_EXPORTS = {"IMPORT_LOG", "_IMPORT_STATE", "_LOGGING_CONFIGURED"}


def __getattr__(name: str):  # pragma: no cover - trivial delegation
    if name in _DYNAMIC_EXPORTS:
        return getattr(_init, name)
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - trivial delegation
    return sorted(set(globals()) | _DYNAMIC_EXPORTS)
