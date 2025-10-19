"""Curated high-level helpers exposed by :mod:`tnfr.helpers`.

The module is intentionally small and surfaces utilities that are stable for
external use, covering data preparation, glyph history management, and graph
cache invalidation.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Protocol, cast

from ..types import TNFRGraph

if TYPE_CHECKING:  # pragma: no cover - import-time only for typing
    from ..utils import (
        CacheManager,
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
    from ..glyph_history import HistoryDict
from .numeric import (
    angle_diff,
    clamp,
    clamp01,
    kahan_sum_nd,
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


_UTIL_EXPORTS = {
    "CacheManager",
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


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    if name in _UTIL_EXPORTS:
        from .. import utils as _utils

        value = getattr(_utils, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - simple reflection
    return sorted(set(__all__))


class _PushGlyphCallable(Protocol):
    def __call__(self, nd: MutableMapping[str, Any], glyph: str, window: int) -> None:
        ...


class _RecentGlyphCallable(Protocol):
    def __call__(self, nd: MutableMapping[str, Any], glyph: str, window: int) -> bool:
        ...


class _EnsureHistoryCallable(Protocol):
    def __call__(self, G: TNFRGraph) -> "HistoryDict | dict[str, Any]":
        ...


class _LastGlyphCallable(Protocol):
    def __call__(self, nd: Mapping[str, Any]) -> str | None:
        ...


class _CountGlyphsCallable(Protocol):
    def __call__(
        self, G: TNFRGraph, window: int | None = ..., *, last_only: bool = ...
    ) -> Counter[str]:
        ...


def _glyph_history_proxy(name: str) -> Callable[..., Any]:
    """Return a wrapper that delegates to :mod:`tnfr.glyph_history` lazily."""

    target: dict[str, Callable[..., Any] | None] = {"func": None}

    def _call(*args: Any, **kwargs: Any) -> Any:
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


count_glyphs = cast(_CountGlyphsCallable, _glyph_history_proxy("count_glyphs"))
ensure_history = cast(_EnsureHistoryCallable, _glyph_history_proxy("ensure_history"))
last_glyph = cast(_LastGlyphCallable, _glyph_history_proxy("last_glyph"))
push_glyph = cast(_PushGlyphCallable, _glyph_history_proxy("push_glyph"))
recent_glyph = cast(_RecentGlyphCallable, _glyph_history_proxy("recent_glyph"))
