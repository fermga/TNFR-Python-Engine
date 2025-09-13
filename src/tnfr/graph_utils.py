"""Utilities for graph-level bookkeeping.

Currently includes helpers to invalidate cached ΔNFR preparation data.
"""

from __future__ import annotations
from typing import Any

__all__ = ("mark_dnfr_prep_dirty", "supports_add_edge")


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
    from .helpers.cache import get_graph

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
