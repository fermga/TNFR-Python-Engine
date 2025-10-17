"""Shared utility helpers exposed under :mod:`tnfr.utils`."""

from __future__ import annotations

from .init import (
    IMPORT_LOG,
    WarnOnce,
    cached_import,
    get_logger,
    get_nodonx,
    get_numpy,
    prune_failed_imports,
    warn_once,
)
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
)
