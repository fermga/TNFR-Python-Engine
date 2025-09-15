"""Curated high-level helpers exposed by :mod:`tnfr.helpers`.

The module is intentionally small and surfaces utilities that are stable for
external use, covering data preparation, glyph history management, and graph
cache invalidation.
"""

from __future__ import annotations

from ..collections_utils import ensure_collection
from ..glyph_history import (
    count_glyphs,
    ensure_history,
    last_glyph,
    push_glyph,
    recent_glyph,
)
from ..cache import node_set_checksum, stable_json
from ..graph_utils import get_graph, get_graph_mapping, mark_dnfr_prep_dirty
from .edge_cache import (
    EdgeCacheManager,
    cached_nodes_and_A,
    edge_version_cache,
    edge_version_update,
    increment_edge_version,
)
from .node_cache import cached_node_list, ensure_node_index_map, ensure_node_offset_map
from .numeric import (
    angle_diff,
    clamp,
    clamp01,
    kahan_sum,
    kahan_sum2d,
    kahan_sum_nd,
    list_mean,
    neighbor_mean,
)

__all__ = (
    "EdgeCacheManager",
    "angle_diff",
    "cached_node_list",
    "cached_nodes_and_A",
    "clamp",
    "clamp01",
    "count_glyphs",
    "edge_version_cache",
    "edge_version_update",
    "ensure_collection",
    "ensure_history",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph",
    "get_graph_mapping",
    "increment_edge_version",
    "kahan_sum",
    "kahan_sum2d",
    "kahan_sum_nd",
    "last_glyph",
    "list_mean",
    "mark_dnfr_prep_dirty",
    "neighbor_mean",
    "node_set_checksum",
    "push_glyph",
    "recent_glyph",
    "stable_json",
)
