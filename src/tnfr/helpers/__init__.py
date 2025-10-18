"""Curated high-level helpers exposed by :mod:`tnfr.helpers`.

The module is intentionally small and surfaces utilities that are stable for
external use, covering data preparation, glyph history management, and graph
cache invalidation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - import-time only for typing
    from ..utils import (
        EdgeCacheManager,
        cached_node_list,
        cached_nodes_and_A,
        edge_version_cache,
        edge_version_update,
        ensure_node_index_map,
        ensure_node_offset_map,
        get_graph,
        get_graph_mapping,
        increment_edge_version,
        mark_dnfr_prep_dirty,
        node_set_checksum,
        stable_json,
    )
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


_UTIL_EXPORTS = {
    "EdgeCacheManager",
    "cached_node_list",
    "cached_nodes_and_A",
    "edge_version_cache",
    "edge_version_update",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph",
    "get_graph_mapping",
    "increment_edge_version",
    "mark_dnfr_prep_dirty",
    "node_set_checksum",
    "stable_json",
}


def __getattr__(name: str):  # pragma: no cover - simple delegation
    if name in _UTIL_EXPORTS:
        from .. import utils as _utils

        value = getattr(_utils, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - simple reflection
    return sorted(set(__all__))


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
