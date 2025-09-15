"""Utilities for graph-level bookkeeping.

This module centralises helpers that operate on the metadata stored inside
graph objects.  Besides flagging ΔNFR preparation caches it also exposes
lightweight adapters to obtain the canonical ``graph`` mapping and to read
validated configuration dictionaries.
"""

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
    """Return an immutable view of ``G``'s stored mapping for ``key``.

    The helper normalises access to ``G.graph[key]`` by returning
    ``None`` when the key is missing or holds a non-mapping value.  When a
    mapping is found it is wrapped in :class:`types.MappingProxyType` to guard
    against accidental mutation.  ``warn_msg`` is emitted via
    :func:`warnings.warn` when the stored value is not a mapping.
    """
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
    """Flag ΔNFR preparation data as stale.

    Parameters
    ----------
    G : Any
        Graph-like object whose ``graph`` attribute will receive the
        ``"_dnfr_prep_dirty"`` flag.

    Returns
    -------
    None
        This function mutates ``G`` in place.
    """
    graph = get_graph(G)
    graph["_dnfr_prep_dirty"] = True


def supports_add_edge(graph: Any) -> bool:
    """Return ``True`` if ``graph`` exposes an ``add_edge`` method.

    Parameters
    ----------
    graph : Any
        Object representing a graph.

    Returns
    -------
    bool
        ``True`` when ``graph`` implements ``add_edge``; ``False`` otherwise.
    """
    return hasattr(graph, "add_edge")
