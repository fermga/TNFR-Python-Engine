from __future__ import annotations

from ..graph_utils import mark_dnfr_prep_dirty

from .numeric import (
    clamp,
    clamp01,
    list_mean,
    kahan_sum_nd,
    kahan_sum,
    kahan_sum2d,
    angle_diff,
    neighbor_mean,
    neighbor_phase_mean,
    neighbor_phase_mean_list,
)
from .cache import (
    get_graph,
    get_graph_mapping,
    node_set_checksum,
    ensure_node_index_map,
    ensure_node_offset_map,
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
    "neighbor_phase_mean",
    "neighbor_phase_mean_list",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "last_glyph",
    "count_glyphs",
    "normalize_counter",
    "mix_groups",
    "ensure_node_index_map",
    "ensure_node_offset_map",
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


def __getattr__(name: str):
    if name in {
        "MAX_MATERIALIZE_DEFAULT",
        "ensure_collection",
        "normalize_weights",
        "normalize_counter",
        "mix_groups",
    }:
        from .. import collections_utils as _cu
        return getattr(_cu, name)
    if name in {
        "push_glyph",
        "recent_glyph",
        "ensure_history",
        "last_glyph",
        "count_glyphs",
    }:
        from .. import glyph_history as _gh
        return getattr(_gh, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
