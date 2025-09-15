from __future__ import annotations

from ..graph_utils import mark_dnfr_prep_dirty

from ..collections_utils import (
    MAX_MATERIALIZE_DEFAULT,
    ensure_collection,
    normalize_weights,
    normalize_counter,
    mix_groups,
)
from ..glyph_history import (
    push_glyph,
    recent_glyph,
    ensure_history,
    last_glyph,
    count_glyphs,
)

from .numeric import (
    clamp,
    clamp01,
    list_mean,
    kahan_sum_nd,
    kahan_sum,
    kahan_sum2d,
    angle_diff,
    neighbor_mean,
)
from .node_cache import (
    NODE_SET_CHECKSUM_KEY,
    get_graph,
    get_graph_mapping,
    node_set_checksum,
    stable_json,
    cached_node_list,
    ensure_node_index_map,
    ensure_node_offset_map,
)
from .edge_cache import (
    EdgeCacheManager,
    edge_version_cache,
    cached_nodes_and_A,
    invalidate_edge_version_cache,
    increment_edge_version,
    edge_version_update,
)

__all__ = (
    "MAX_MATERIALIZE_DEFAULT",
    "ensure_collection",
    "clamp",
    "clamp01",
    "list_mean",
    "kahan_sum_nd",
    "kahan_sum",
    "kahan_sum2d",
    "angle_diff",
    "normalize_weights",
    "neighbor_mean",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "last_glyph",
    "count_glyphs",
    "normalize_counter",
    "mix_groups",
    "cached_node_list",
    "NODE_SET_CHECKSUM_KEY",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "stable_json",
    "EdgeCacheManager",
    "edge_version_cache",
    "cached_nodes_and_A",
    "invalidate_edge_version_cache",
    "increment_edge_version",
    "edge_version_update",
    "node_set_checksum",
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
)
