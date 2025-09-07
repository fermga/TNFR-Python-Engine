"""Utilities for graph-level bookkeeping."""

from __future__ import annotations
from typing import Any

__all__ = ["mark_dnfr_prep_dirty"]

def mark_dnfr_prep_dirty(G: Any) -> None:
    """Mark cached ΔNFR preparation data as stale for ``G``.

    This sets a flag in ``G.graph`` so that subsequent calls to
    :func:`_prepare_dnfr_data` know that node attributes or topology have
    changed and cached arrays need to be refreshed.
    """
    graph = G.graph if hasattr(G, "graph") else G
    graph["_dnfr_prep_dirty"] = True
