from __future__ import annotations

from typing import Any

from ..cache import CacheManager as CacheManager
from ..glyph_history import (
    HistoryDict,
    count_glyphs as count_glyphs,
    ensure_history as ensure_history,
    last_glyph as last_glyph,
    push_glyph as push_glyph,
    recent_glyph as recent_glyph,
)
from ..utils.cache import (
    EdgeCacheManager as EdgeCacheManager,
    cached_node_list as cached_node_list,
    cached_nodes_and_A as cached_nodes_and_A,
    edge_version_cache as edge_version_cache,
    edge_version_update as edge_version_update,
    ensure_node_index_map as ensure_node_index_map,
    ensure_node_offset_map as ensure_node_offset_map,
    node_set_checksum as node_set_checksum,
    stable_json as stable_json,
)
from ..utils.graph import (
    get_graph as get_graph,
    get_graph_mapping as get_graph_mapping,
    increment_edge_version as increment_edge_version,
    mark_dnfr_prep_dirty as mark_dnfr_prep_dirty,
)
from .numeric import (
    angle_diff as angle_diff,
    clamp as clamp,
    clamp01 as clamp01,
    kahan_sum_nd as kahan_sum_nd,
)

__all__ = (
    "CacheManager",
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
    "__getattr__",
)


def __getattr__(name: str) -> Any: ...
