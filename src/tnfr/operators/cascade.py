"""Cascade detection and analysis for THOL self-organization.

Provides tools to detect, measure, and analyze emergent cascades in
TNFR networks where THOL bifurcations propagate through coupled nodes.

TNFR Canonical Principle
-------------------------
From "El pulso que nos atraviesa" (TNFR Manual, §2.2.10):

    "THOL actúa como modulador central de plasticidad. Es el glifo que
    permite a la red reorganizar su topología sin intervención externa.
    Su activación crea bucles de aprendizaje resonante, trayectorias de
    reorganización emergente, estabilidad dinámica basada en coherencia local."

This module implements cascade detection: when THOL bifurcations propagate
through phase-aligned neighbors, creating chains of emergent reorganization.

Performance Optimization
------------------------
CASCADE DETECTION CACHING: `detect_cascade()` uses TNFR's canonical caching
infrastructure (`@cache_tnfr_computation`) to avoid recomputing cascade state.
Topology and edge weights enter the cache fingerprint. History/configuration
entries retain explicit dependency tags for targeted invalidation.

Cache identity covers the graph object, topology/weights, propagation history
and cascade configuration.
This provides significant performance improvement for large networks (>1000 nodes)
where cascade detection is called frequently (e.g., in `self_organization_metrics`).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "detect_cascade",
    "measure_cascade_radius",
    "invalidate_cascade_cache",
]

# Import cache utilities for performance optimization
from ..mathematics.unified_cache import CacheLevel, cache_tnfr_computation
from ._diagnostic_scores import (
    finite_real,
    sum_nonnegative_magnitudes,
)

_CACHING_AVAILABLE = True


def _estimate_cascade_cost(G: TNFRGraph) -> float:
    """Estimate computational cost for cascade detection.

    Used by cache eviction policy to prioritize expensive computations.
    Cost is proportional to number of propagation events to process.
    """
    propagations = G.graph.get("thol_propagations", [])
    # Base cost + cost per propagation event
    return 1.0 + len(propagations) * 0.1


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS,
    dependencies={"thol_propagations", "cascade_config", "graph_topology"},
    cost_estimator=_estimate_cascade_cost,
)
def detect_cascade(G: TNFRGraph) -> dict[str, Any]:
    """Detect if THOL triggered a propagation cascade in the network.

    A cascade is defined as a chain reaction where:
    1. Node A bifurcates (THOL)
    2. Sub-EPI propagates to coupled neighbors
    3. Neighbors' EPIs increase, potentially triggering their own bifurcations
    4. Process continues across ≥3 nodes

    **Performance**: This function uses TNFR's canonical cache infrastructure
    to avoid recomputing cascade state. First call builds cache (O(P × N_prop)),
    subsequent calls can reuse the cached result. Topology/weight changes alter
    the cache fingerprint; history/configuration changes require dependency
    invalidation through the mutation path or invalidate_cascade_cache().

    Parameters
    ----------
    G : TNFRGraph
        Graph with THOL propagation history

    Returns
    -------
    dict
        Cascade analysis containing:
        - is_cascade: bool (True if cascade detected)
        - affected_nodes: set of NodeIds involved
        - cascade_depth: maximum propagation chain length
        - total_propagations: total number of propagation events
        - mean_internal_edge_weight_magnitude: unbounded mean magnitude of
          induced edge weights
        - cascade_coherence: deprecated compatibility alias for that magnitude

    Notes
    -----
    The edge-weight magnitude is a coupling diagnostic. It reads neither
    DeltaNFR nor dEPI and therefore does not measure canonical C(t).

    Caching Strategy:
    - Cache level: DERIVED_METRICS (mid-persistence)
    - Dependencies: 'thol_propagations' (propagation history),
                   'cascade_config' (threshold parameters)
    - Invalidation: topology fingerprint plus explicit dependency invalidation
    - Cost: Proportional to number of propagation events

    Cache reuse avoids recomputing the traversal when the declared dependencies
    are unchanged; no fixed speedup is implied.

    Examples
    --------
    >>> # Network with cascade
    >>> analysis = detect_cascade(G)
    >>> analysis["is_cascade"]
    True
    >>> analysis["cascade_depth"]
    4  # Propagated through 4 levels
    >>> len(analysis["affected_nodes"])
    7  # 7 nodes affected
    """
    propagations = G.graph.get("thol_propagations", [])

    if not propagations:
        return {
            "is_cascade": False,
            "affected_nodes": set(),
            "cascade_depth": 0,
            "total_propagations": 0,
            "mean_internal_edge_weight_magnitude": 0.0,
            "cascade_coherence": 0.0,
            "canonical_coherence_certified": False,
        }

    # Build propagation graph
    affected_nodes = set()
    for prop in propagations:
        affected_nodes.add(prop["source_node"])
        for target, _ in prop["propagations"]:
            affected_nodes.add(target)

    # Compute cascade depth (longest propagation chain)
    # For now, approximate as number of propagation events
    cascade_depth = len(propagations)

    # Total propagations
    total_props = sum(len(p["propagations"]) for p in propagations)

    # Get cascade minimum nodes from config
    cascade_min_nodes = int(G.graph.get("THOL_CASCADE_MIN_NODES", 3))

    # Cascade = affects ≥ cascade_min_nodes
    is_cascade = len(affected_nodes) >= cascade_min_nodes

    mean_edge_magnitude = _mean_internal_edge_weight_magnitude(
        G, affected_nodes
    )
    return {
        "is_cascade": is_cascade,
        "affected_nodes": affected_nodes,
        "cascade_depth": cascade_depth,
        "total_propagations": total_props,
        "mean_internal_edge_weight_magnitude": mean_edge_magnitude,
        # Public compatibility alias. This value is not structural C(t).
        "cascade_coherence": mean_edge_magnitude,
        "canonical_coherence_certified": False,
    }


def _mean_internal_edge_weight_magnitude(
    G: TNFRGraph, affected_nodes: set[NodeId]
) -> float:
    """Return the unbounded mean magnitude of induced edge weights.

    Directed arcs and parallel edges each count once. Missing weights default
    to 1.0. The result is finite and nonnegative but has no upper bound, and it
    is not canonical structural coherence C(t).
    """

    if not affected_nodes:
        return 0.0

    magnitudes = []
    for left, right, data in G.edges(data=True):
        if left not in affected_nodes or right not in affected_nodes:
            continue
        weight = finite_real(
            data.get("weight", 1.0),
            label=f"THOL cascade edge weight {left!r}->{right!r}",
        )
        magnitudes.append(abs(weight))
    if not magnitudes:
        return 0.0
    total = sum_nonnegative_magnitudes(
        magnitudes, label="THOL internal edge-weight magnitude"
    )
    return total / len(magnitudes)


def measure_cascade_radius(G: TNFRGraph, source_node: NodeId) -> int:
    """Measure propagation radius from bifurcation source.

    Parameters
    ----------
    G : TNFRGraph
        Graph with propagation history
    source_node : NodeId
        Origin node of cascade

    Returns
    -------
    int
        Number of nodes reached by propagation (hop distance)

    Notes
    -----
    Uses BFS to trace propagation paths from source.

    Examples
    --------
    >>> # Linear cascade: 0 -> 1 -> 2 -> 3
    >>> radius = measure_cascade_radius(G, source_node=0)
    >>> radius
    3  # Reached 3 hops from source
    """
    propagations = G.graph.get("thol_propagations", [])

    # Build propagation edges from this source
    prop_edges = []
    for prop in propagations:
        if prop["source_node"] == source_node:
            for target, _ in prop["propagations"]:
                prop_edges.append((source_node, target))

    if not prop_edges:
        return 0

    # BFS to measure radius
    visited = {source_node}
    queue = deque([(source_node, 0)])  # (node, distance)
    max_distance = 0

    while queue:
        current, dist = queue.popleft()
        max_distance = max(max_distance, dist)

        for src, tgt in prop_edges:
            if src == current and tgt not in visited:
                visited.add(tgt)
                queue.append((tgt, dist + 1))

    return max_distance


def invalidate_cascade_cache() -> int:
    """Invalidate cached cascade detection results across all graphs.

    This function should be called when THOL propagations are added or
    cascade configuration parameters change. It triggers automatic cache
    invalidation via the dependency tracking system.

    Returns
    -------
    int
        Number of cache entries invalidated.

    Notes
    -----
    TNFR Caching: Uses canonical `invalidate_by_dependency()` mechanism.
    Dependencies invalidated: 'thol_propagations', 'cascade_config'.

    Call this function after mutating propagation history or cascade
    configuration outside a managed mutation path.

    Examples
    --------
    >>> # Add new propagations
    >>> G.graph["thol_propagations"].append(new_propagation)
    >>> # Cache invalidates automatically, but can force if needed
    >>> invalidate_cascade_cache()  # doctest: +SKIP
    2  # Invalidated 2 cache entries
    """
    if not _CACHING_AVAILABLE:
        return 0

    try:
        from ..utils.cache import get_global_cache

        cache = get_global_cache()
        count = 0
        count += cache.invalidate_by_dependency("thol_propagations")
        count += cache.invalidate_by_dependency("cascade_config")
        return count
    except (ImportError, AttributeError):  # pragma: no cover
        return 0
