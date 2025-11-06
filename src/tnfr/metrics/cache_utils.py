"""Cache utilities and telemetry for TNFR hot path optimizations.

This module provides helper functions for cache profiling, configuration,
and monitoring across the metrics computation pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, MutableMapping

from ..types import GraphLike
from ..utils import get_graph

__all__ = (
    "get_cache_config",
    "configure_hot_path_caches",
    "log_cache_metrics",
    "CacheStats",
)

logger = logging.getLogger(__name__)


class CacheStats:
    """Aggregate cache statistics for telemetry and profiling."""

    __slots__ = ("hits", "misses", "evictions", "hit_rate", "total_accesses")

    def __init__(
        self,
        hits: int = 0,
        misses: int = 0,
        evictions: int = 0,
    ) -> None:
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.total_accesses = hits + misses
        self.hit_rate = hits / self.total_accesses if self.total_accesses > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.2%})"
        )

    def merge(self, other: CacheStats) -> CacheStats:
        """Combine statistics from another CacheStats instance."""
        return CacheStats(
            hits=self.hits + other.hits,
            misses=self.misses + other.misses,
            evictions=self.evictions + other.evictions,
        )


def get_cache_config(
    G: GraphLike,
    *,
    key: str = "_cache_config",
) -> dict[str, Any]:
    """Retrieve cache configuration from graph metadata.

    Parameters
    ----------
    G : GraphLike
        Graph containing cache configuration.
    key : str, default: "_cache_config"
        Configuration key in graph attributes.

    Returns
    -------
    dict[str, Any]
        Cache configuration dictionary. Returns empty dict if not configured.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.graph["_cache_config"] = {"buffer_max_entries": 256}
    >>> config = get_cache_config(G)
    >>> config["buffer_max_entries"]
    256
    """
    graph = get_graph(G)
    config = graph.get(key)
    if not isinstance(config, Mapping):
        return {}
    return dict(config)


def configure_hot_path_caches(
    G: GraphLike,
    *,
    buffer_max_entries: int | None = None,
    si_chunk_size: int | None = None,
    trig_cache_size: int | None = None,
    coherence_cache_size: int | None = None,
) -> None:
    """Configure cache capacities for hot path computations.

    This function provides a unified interface for setting cache limits
    across the metrics computation pipeline. It consolidates configuration
    that would otherwise be scattered across multiple graph attributes.

    Parameters
    ----------
    G : GraphLike
        Graph to configure.
    buffer_max_entries : int or None, optional
        Maximum number of buffer sets cached by ensure_numpy_buffers.
        None means use default (128).
    si_chunk_size : int or None, optional
        Chunk size for Si computation. None means auto-detect.
    trig_cache_size : int or None, optional
        Maximum entries in trigonometric cache. None means default (128).
    coherence_cache_size : int or None, optional
        Cache size for coherence matrix computations. None means default.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> configure_hot_path_caches(
    ...     G,
    ...     buffer_max_entries=256,
    ...     si_chunk_size=1000,
    ...     trig_cache_size=512
    ... )
    >>> G.graph.get("_cache_config")  # doctest: +SKIP
    {'buffer_max_entries': 256, 'trig_cache_size': 512}
    """
    graph = get_graph(G)
    config: MutableMapping[str, Any] = graph.setdefault("_cache_config", {})

    if buffer_max_entries is not None:
        config["buffer_max_entries"] = int(buffer_max_entries)

    if si_chunk_size is not None:
        graph["SI_CHUNK_SIZE"] = int(si_chunk_size)

    if trig_cache_size is not None:
        config["trig_cache_size"] = int(trig_cache_size)

    if coherence_cache_size is not None:
        config["coherence_cache_size"] = int(coherence_cache_size)


def log_cache_metrics(
    G: GraphLike,
    *,
    logger_instance: logging.Logger | None = None,
    level: int = logging.INFO,
) -> None:
    """Log cache metrics for monitoring and profiling.

    This function extracts cache statistics from the graph's CacheManager
    and logs them for telemetry analysis. It's useful for identifying
    cache inefficiencies and tuning cache sizes.

    Parameters
    ----------
    G : GraphLike
        Graph whose cache metrics should be logged.
    logger_instance : logging.Logger or None, optional
        Logger to use. If None, uses module logger.
    level : int, default: logging.INFO
        Logging level for the output.

    Examples
    --------
    >>> import networkx as nx
    >>> import logging
    >>> G = nx.Graph()
    >>> log_cache_metrics(G, level=logging.DEBUG)  # doctest: +SKIP
    """
    if logger_instance is None:
        logger_instance = logger

    graph = get_graph(G)
    manager = graph.get("_tnfr_cache_manager")
    if manager is None:
        logger_instance.log(level, "No cache manager found on graph")
        return

    try:
        aggregate = manager.aggregate_metrics()
        total = aggregate.hits + aggregate.misses
        hit_rate = aggregate.hits / total if total > 0 else 0.0

        logger_instance.log(
            level,
            "Cache metrics: hits=%d misses=%d evictions=%d hit_rate=%.2f%%",
            aggregate.hits,
            aggregate.misses,
            aggregate.evictions,
            hit_rate * 100,
        )

        # Log per-cache breakdown
        for name, stats in manager.iter_metrics():
            cache_total = stats.hits + stats.misses
            cache_hit_rate = stats.hits / cache_total if cache_total > 0 else 0.0
            logger_instance.log(
                logging.DEBUG,
                "  %s: hits=%d misses=%d hit_rate=%.2f%%",
                name,
                stats.hits,
                stats.misses,
                cache_hit_rate * 100,
            )
    except Exception as exc:
        logger_instance.warning("Failed to log cache metrics: %s", exc)
