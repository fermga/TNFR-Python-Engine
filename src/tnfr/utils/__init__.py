"""Shared utility helpers exposed under :mod:`tnfr.utils`."""

from __future__ import annotations

from typing import Any, Final

from . import init as _init
from ..cache import CacheManager
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
from .cache import (
    EdgeCacheManager,
    NODE_SET_CHECKSUM_KEY,
    cached_node_list,
    cached_nodes_and_A,
    clear_node_repr_cache,
    edge_version_cache,
    edge_version_update,
    ensure_node_index_map,
    ensure_node_offset_map,
    get_graph_version,
    increment_edge_version,
    increment_graph_version,
    configure_graph_cache_limits,
    node_set_checksum,
    stable_json,
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
    "warm_cached_import",
    "LazyImportProxy",
    "get_logger",
    "get_nodenx",
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
    "CacheManager",
    "EdgeCacheManager",
    "NODE_SET_CHECKSUM_KEY",
    "cached_node_list",
    "cached_nodes_and_A",
    "clear_node_repr_cache",
    "edge_version_cache",
    "edge_version_update",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph_version",
    "increment_edge_version",
    "increment_graph_version",
    "configure_graph_cache_limits",
    "node_set_checksum",
    "stable_json",
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "json_dumps",
    "clear_orjson_param_warnings",
    "validate_window",
    "run_validators",
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
warm_cached_import = _init.warm_cached_import
LazyImportProxy = _init.LazyImportProxy
get_logger = _init.get_logger
get_nodenx = _init.get_nodenx
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

#: Mapping of dynamically proxied names to the runtime types they expose.
#:
#: ``IMPORT_LOG`` and ``_IMPORT_STATE`` refer to the
#: :class:`~tnfr.utils.init.ImportRegistry` instance that tracks cached import
#: metadata, while ``_LOGGING_CONFIGURED`` is the module-level flag guarding the
#: lazy logging bootstrap performed in :mod:`tnfr.utils.init`.
_DYNAMIC_EXPORT_TYPES: Final[dict[str, type[object]]] = {
    "IMPORT_LOG": _init.ImportRegistry,
    "_IMPORT_STATE": _init.ImportRegistry,
    "_LOGGING_CONFIGURED": bool,
}
_DYNAMIC_EXPORTS: Final[frozenset[str]] = frozenset(_DYNAMIC_EXPORT_TYPES)
_VALIDATOR_EXPORTS = {"validate_window", "run_validators"}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    if name in _DYNAMIC_EXPORTS:
        return getattr(_init, name)
    if name in _VALIDATOR_EXPORTS:
        if name == "validate_window":
            from .validators import validate_window as _validate_window

            return _validate_window
        from .validators import run_validators as _run_validators

        return _run_validators
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - trivial delegation
    return sorted(set(globals()) | _DYNAMIC_EXPORTS | _VALIDATOR_EXPORTS)
