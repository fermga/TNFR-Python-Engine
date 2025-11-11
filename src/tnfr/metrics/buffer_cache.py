"""Unified buffer cache for TNFR metrics hot paths.

This module consolidates buffer management across hot path computations
(Sense index, coherence, ΔNFR) to eliminate duplication and ensure consistent
cache key patterns and invalidation strategies.

Cache Key Structure
-------------------
All buffer caches use a tuple key: ``(key_prefix, count, buffer_count)``

This ensures:
- Collision avoidance between different computations (via unique key_prefix)
- Automatic invalidation on graph topology changes (via edge_version_cache)
- Efficient cache lookups without hash collisions

Common Key Prefixes
-------------------
- ``_si_buffers``: Sense index main computation buffers
- ``_si_chunk_workspace``: Si chunked processing scratch space
- ``_si_neighbor_buffers``: Si neighbor phase aggregation buffers
- ``_coherence_temp``: Coherence matrix temporary buffers
- ``_dnfr_prep_buffers``: ΔNFR preparation workspace

See docs/CACHING_STRATEGY.md for complete cache documentation.
"""

from __future__ import annotations

from typing import Any

from ..types import GraphLike
from ..utils import edge_version_cache, get_graph

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

    This function centralizes buffer allocation for vectorized computations,
    ensuring consistent cache key structure and automatic invalidation on
    topology changes. Buffers are tied to the graph's edge version and
    automatically cleared when edges are added or removed.

    Cache Behavior
    --------------
    - **Key**: ``(key_prefix, count, buffer_count)`` ensures uniqueness
    - **Invalidation**: Automatic on edge version changes
    - **Capacity**: Controlled by ``max_cache_entries`` parameter
    - **Override**: Graph-level config via ``_cache_config['buffer_max_entries']``

    Parameters
    ----------
    G : GraphLike
        Graph whose edge version controls cache invalidation.
    key_prefix : str
        Prefix for the cache key, e.g. ``"_si_buffers"`` or ``"_coherence_temp"``.
        Must be unique per computation to avoid key collisions. See module
        docstring for standard prefixes.
    count : int
        Number of elements per buffer. Typically set to node count for
        node-level computations or edge count for edge-level operations.
    buffer_count : int
        Number of buffers to allocate. Each buffer is independent and can be
        used for different intermediate values in the computation.
    np : Any
        NumPy module or compatible array backend. Must support ``np.empty``.
    dtype : Any, optional
        Data type for the buffers. Default: ``float``.
    max_cache_entries : int or None, optional
        Maximum number of cached buffer sets for this key prefix. Default: ``128``.
        Set to ``None`` for unlimited cache size (use with caution on large graphs).
        Can be overridden globally via graph-level configuration.

    Returns
    -------
    tuple[Any, ...]
        Tuple of ``buffer_count`` NumPy arrays each sized to ``count`` elements.
        Arrays are reused from cache when available, avoiding repeated allocation.

    Notes
    -----
    This function consolidates buffer allocation patterns across Si computation,
    coherence matrix computation, and ΔNFR preparation. By centralizing buffer
    management, we ensure consistent cache key naming, avoid duplication, and
    maintain coherent cache invalidation when the graph edge structure changes.

    The buffer allocation pattern follows TNFR caching principles:
    1. **Determinism**: Same graph topology → same cached buffers
    2. **Coherence**: Edge changes → automatic cache invalidation
    3. **Efficiency**: Reuse eliminates allocation overhead in hot loops

    Performance Considerations
    --------------------------
    - Cache hits avoid O(n) allocation overhead
    - Memory cost: O(buffer_count * count * sizeof(dtype)) per cache entry
    - Recommended for buffers reused across multiple computation steps
    - Consider chunked processing for very large graphs (n > 100k nodes)

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
    >>> all(isinstance(buf, np.ndarray) for buf in buffers)
    True

    Allocate workspace for a computation with 100 nodes:

    >>> G_large = nx.complete_graph(100)
    >>> workspace = ensure_numpy_buffers(
    ...     G_large,
    ...     key_prefix="_my_computation",
    ...     count=100,
    ...     buffer_count=2,
    ...     np=np
    ... )
    >>> workspace[0].size == 100
    True

    See Also
    --------
    edge_version_cache : Underlying cache mechanism
    configure_hot_path_caches : Global cache configuration
    """
    # Allow graph-level override of max_cache_entries
    graph = get_graph(G)
    cache_config = graph.get("_cache_config")
    if isinstance(cache_config, dict) and max_cache_entries is not None:
        override = cache_config.get("buffer_max_entries")
        if override is not None:
            max_cache_entries = int(override)

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
