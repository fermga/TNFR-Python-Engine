"""Utilities for graph-level bookkeeping.

Currently includes helpers to invalidate cached ΔNFR preparation data.
"""

from __future__ import annotations
from typing import Any

__all__ = ("mark_dnfr_prep_dirty",)


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
