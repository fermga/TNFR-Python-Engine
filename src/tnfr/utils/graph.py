"""Utilities for graph-level bookkeeping shared by TNFR components."""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Any, Mapping

__all__ = (
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
)


def get_graph(obj: Any) -> Any:
    """Return ``obj.graph`` when present or ``obj`` otherwise."""

    return getattr(obj, "graph", obj)


def get_graph_mapping(
    G: Any, key: str, warn_msg: str
) -> Mapping[str, Any] | None:
    """Return an immutable view of ``G``'s stored mapping for ``key``."""

    graph = get_graph(G)
    getter = getattr(graph, "get", None)
    if getter is None:
        return None

    data = getter(key)
    if data is None:
        return None
    if not isinstance(data, Mapping):
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
        return None
    return MappingProxyType(data)


def mark_dnfr_prep_dirty(G: Any) -> None:
    """Flag ΔNFR preparation data as stale by marking ``G.graph``."""

    graph = get_graph(G)
    graph["_dnfr_prep_dirty"] = True


def supports_add_edge(graph: Any) -> bool:
    """Return ``True`` if ``graph`` exposes an ``add_edge`` method."""

    return hasattr(graph, "add_edge")
