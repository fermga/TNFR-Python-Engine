"""Centralized utility functions for the TNFR engine.

This module serves as the canonical single point of access for generic helper
functions, including numeric operations, cache infrastructure, data normalization,
parsing utilities, and graph helpers. All functions maintain deterministic behavior
and respect TNFR structural semantics.

**Module Organization**:

* :mod:`tnfr.utils.numeric` - Compensated arithmetic, angle operations, clamping
* :mod:`tnfr.utils.cache` - Cache layers, versioning, graph-level caching
* :mod:`tnfr.utils.data` - Type conversion, weight normalization, collections
* :mod:`tnfr.utils.io` - JSON/YAML/TOML parsing, atomic file operations
* :mod:`tnfr.utils.graph` - Graph metadata access, ΔNFR prep management
* :mod:`tnfr.utils.chunks` - Chunk size computation for parallel operations
* :mod:`tnfr.utils.callbacks` - Callback registration and invocation
* :mod:`tnfr.utils.init` - Lazy imports, logging configuration

**Stability Guarantees**:

All public functions exported from this module constitute the stable utility API.
Functions prefixed with ``_`` are internal implementation details subject to change.

**Example Usage**::

    from tnfr.utils import clamp, json_dumps, normalize_weights
    from tnfr.utils import CacheManager, cached_node_list

    # Numeric operations
    value = clamp(x, 0.0, 1.0)

    # Data normalization
    weights = normalize_weights(raw_weights, ['phase', 'epi', 'vf'])

    # Caching
    nodes = cached_node_list(G)

    # Serialization
    data = json_dumps(obj, sort_keys=True)

See :doc:`/docs/utils_reference` for comprehensive documentation.
"""

from __future__ import annotations

from typing import Any, Final

from . import init as _init
from ..locking import get_lock

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

from .cache import (
    CacheCapacityConfig,
    CacheLayer,
    CacheManager,
    CacheStatistics,
    InstrumentedLRUCache,
    ManagedLRUCache,
    MappingCacheLayer,
    RedisCacheLayer,
    ShelveCacheLayer,
    SecurityError,
    SecurityWarning,
    create_hmac_signer,
    create_hmac_validator,
    create_secure_shelve_layer,
    create_secure_redis_layer,
    prune_lock_mapping,
    DNFR_PREP_STATE_KEY,
    DnfrPrepState,
    DnfrCache,
    NODE_SET_CHECKSUM_KEY,
    ScopedCounterCache,
    EdgeCacheManager,
    cached_node_list,
    cached_nodes_and_A,
    clear_node_repr_cache,
    configure_graph_cache_limits,
    configure_global_cache_layers,
    edge_version_cache,
    edge_version_update,
    ensure_node_index_map,
    ensure_node_offset_map,
    new_dnfr_cache,
    _SeedHashCache,
    _GRAPH_CACHE_MANAGER_KEY,
    _graph_cache_manager,
    build_cache_manager,
    get_graph_version,
    increment_edge_version,
    increment_graph_version,
    node_set_checksum,
    reset_global_cache_manager,
    stable_json,
    _GRAPH_CACHE_LAYERS_KEY,
)
from .data import (
    MAX_MATERIALIZE_DEFAULT,
    STRING_TYPES,
    convert_value,
    normalize_optional_int,
    ensure_collection,
    flatten_structure,
    is_non_string_sequence,
    mix_groups,
    negative_weights_warn_once,
    normalize_counter,
    normalize_materialize_limit,
    normalize_weights,
)
from .chunks import auto_chunk_size, resolve_chunk_size
from .graph import (
    get_graph,
    get_graph_mapping,
    mark_dnfr_prep_dirty,
    supports_add_edge,
)
from .numeric import (
    angle_diff,
    angle_diff_array,
    clamp,
    clamp01,
    kahan_sum_nd,
    similarity_abs,
    within_range,
)
from .io import (
    DEFAULT_PARAMS,
    JsonDumpsParams,
    StructuredFileError,
    clear_orjson_param_warnings,
    json_dumps,
    read_structured_file,
    safe_write,
)
from .callbacks import (
    CallbackEvent,
    CallbackManager,
    callback_manager,
    CallbackSpec,
)
from .topology import (
    compute_k_top_spectral,
    compute_laplacian_spectrum,
    compute_fiedler_value,
)

__all__ = (
    "IMPORT_LOG",
    "WarnOnce",
    "cached_import",
    "warm_cached_import",
    "LazyImportProxy",
    "get_logger",
    "get_lock",
    "get_nodenx",
    "get_numpy",
    "prune_failed_imports",
    "warn_once",
    "convert_value",
    "normalize_optional_int",
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
    "angle_diff",
    "angle_diff_array",
    "clamp",
    "clamp01",
    "auto_chunk_size",
    "resolve_chunk_size",
    "CacheCapacityConfig",
    "CacheLayer",
    "CacheManager",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "ManagedLRUCache",
    "MappingCacheLayer",
    "RedisCacheLayer",
    "ShelveCacheLayer",
    "SecurityError",
    "SecurityWarning",
    "create_hmac_signer",
    "create_hmac_validator",
    "create_secure_shelve_layer",
    "create_secure_redis_layer",
    "prune_lock_mapping",
    "EdgeCacheManager",
    "DNFR_PREP_STATE_KEY",
    "DnfrPrepState",
    "DnfrCache",
    "NODE_SET_CHECKSUM_KEY",
    "ScopedCounterCache",
    "cached_node_list",
    "cached_nodes_and_A",
    "clear_node_repr_cache",
    "edge_version_cache",
    "edge_version_update",
    "configure_global_cache_layers",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "new_dnfr_cache",
    "get_graph_version",
    "increment_edge_version",
    "increment_graph_version",
    "configure_graph_cache_limits",
    "build_cache_manager",
    "_graph_cache_manager",
    "_GRAPH_CACHE_MANAGER_KEY",
    "node_set_checksum",
    "stable_json",
    "reset_global_cache_manager",
    "_SeedHashCache",
    "_GRAPH_CACHE_LAYERS_KEY",
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "json_dumps",
    "clear_orjson_param_warnings",
    "read_structured_file",
    "safe_write",
    "StructuredFileError",
    "kahan_sum_nd",
    "similarity_abs",
    "within_range",
    "_configure_root",
    "_LOGGING_CONFIGURED",
    "_reset_logging_state",
    "_reset_import_state",
    "_IMPORT_STATE",
    "_warn_failure",
    "_FAILED_IMPORT_LIMIT",
    "_DEFAULT_CACHE_SIZE",
    "EMIT_MAP",
    "CallbackEvent",
    "CallbackManager",
    "callback_manager",
    "CallbackSpec",
    # Topology / spectral analysis (experimental, U6 research)
    "compute_k_top_spectral",
    "compute_laplacian_spectrum",
    "compute_fiedler_value",
)

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


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    if name in _DYNAMIC_EXPORTS:
        return getattr(_init, name)
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - trivial delegation
    return sorted(set(globals()) | set(_DYNAMIC_EXPORTS))
