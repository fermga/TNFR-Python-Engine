"""Curated high-level helpers exposed by :mod:`tnfr.helpers`.

The module is intentionally small and surfaces utilities that are stable for
external use, covering data preparation, glyph history management, and graph
cache invalidation.
"""

from __future__ import annotations

from typing import Any, Callable

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


def _glyph_history_proxy(name: str) -> Callable[..., Any]:
    """Return a wrapper that delegates to :mod:`tnfr.glyph_history` lazily."""

    target: dict[str, Callable[..., Any] | None] = {"func": None}

    def _call(*args: Any, **kwargs: Any):
        func = target["func"]
        if func is None:
            from .. import glyph_history as _glyph_history

            func = getattr(_glyph_history, name)
            target["func"] = func
        return func(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = name
    _call.__doc__ = f"Proxy for :func:`tnfr.glyph_history.{name}`."
    return _call


count_glyphs = _glyph_history_proxy("count_glyphs")
ensure_history = _glyph_history_proxy("ensure_history")
last_glyph = _glyph_history_proxy("last_glyph")
push_glyph = _glyph_history_proxy("push_glyph")
recent_glyph = _glyph_history_proxy("recent_glyph")
