"""Unified buffer cache for TNFR metrics hot paths.

This module consolidates buffer management across hot path computations
(Sense index, coherence, ΔNFR) to eliminate duplication and ensure consistent
cache key patterns and invalidation strategies.
"""

from __future__ import annotations

from typing import Any, Callable

from ..types import GraphLike
from ..utils import edge_version_cache

__all__ = ("ensure_numpy_buffers",)


def ensure_numpy_buffers(
    G: GraphLike,
    *,
    key_prefix: str,
    count: int,
    buffer_count: int,
    np: Any,
    dtype: Any = None,
    max_cache_entries: int | None = 128,
) -> tuple[Any, ...]:
    """Return reusable NumPy buffers with unified caching strategy.

    Parameters
    ----------
    G : GraphLike
        Graph whose edge version controls cache invalidation.
    key_prefix : str
        Prefix for the cache key, e.g. ``"_si_buffers"`` or ``"_coherence_temp"``.
    count : int
        Number of elements per buffer.
    buffer_count : int
        Number of buffers to allocate.
    np : Any
        NumPy module or compatible array backend.
    dtype : Any, optional
        Data type for the buffers. Defaults to ``float``.
    max_cache_entries : int or None, optional
        Maximum number of cached buffer sets. Defaults to 128. Use ``None`` for
        unlimited cache size.

    Returns
    -------
    tuple[Any, ...]
        Tuple of ``buffer_count`` NumPy arrays each sized to ``count`` elements.

    Notes
    -----
    This function consolidates buffer allocation patterns across Si computation,
    coherence matrix computation, and ΔNFR preparation. By centralizing buffer
    management, we ensure consistent cache key naming, avoid duplication, and
    maintain coherent cache invalidation when the graph edge structure changes.

    Examples
    --------
    >>> import numpy as np
    >>> import networkx as nx
    >>> G = nx.Graph([(0, 1)])
    >>> buffers = ensure_numpy_buffers(
    ...     G, key_prefix="_test", count=10, buffer_count=3, np=np
    ... )
    >>> len(buffers)
    3
    >>> buffers[0].shape
    (10,)
    """

    if dtype is None:
        dtype = float
    if count <= 0:
        count = 1

    def builder() -> tuple[Any, ...]:
        return tuple(np.empty(count, dtype=dtype) for _ in range(buffer_count))

    return edge_version_cache(
        G,
        (key_prefix, count, buffer_count),
        builder,
        max_entries=max_cache_entries,
    )
