"""Curated high-level helpers exposed by :mod:`tnfr.helpers`.

The module is intentionally small and surfaces utilities that are stable for
external use, covering data preparation, glyph history management, and graph
cache invalidation.
"""

from __future__ import annotations

from ..utils.cache import (
    EdgeCacheManager,
    cached_node_list,
    cached_nodes_and_A,
    edge_version_cache,
    edge_version_update,
    ensure_node_index_map,
    ensure_node_offset_map,
    increment_edge_version,
    node_set_checksum,
    stable_json,
)
from ..utils.graph import get_graph, get_graph_mapping, mark_dnfr_prep_dirty
from .numeric import (
    angle_diff,
    clamp,
    clamp01,
    kahan_sum_nd,
)


def __getattr__(name: str):
    if name in _GLYPH_HISTORY_EXPORTS:
        from .. import glyph_history as _glyph_history

        value = getattr(_glyph_history, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = (
    "EdgeCacheManager",
    "angle_diff",
    "cached_node_list",
    "cached_nodes_and_A",
    "clamp",
    "clamp01",
    "edge_version_cache",
    "edge_version_update",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph",
    "get_graph_mapping",
    "increment_edge_version",
    "kahan_sum_nd",
    "mark_dnfr_prep_dirty",
    "node_set_checksum",
    "stable_json",
    "count_glyphs",
    "ensure_history",
    "last_glyph",
    "push_glyph",
    "recent_glyph",
)

_GLYPH_HISTORY_EXPORTS = (
    "count_glyphs",
    "ensure_history",
    "last_glyph",
    "push_glyph",
    "recent_glyph",
)
