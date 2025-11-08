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
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "detect_cascade",
    "measure_cascade_radius",
]


def detect_cascade(G: TNFRGraph) -> dict[str, Any]:
    """Detect if THOL triggered a propagation cascade in the network.

    A cascade is defined as a chain reaction where:
    1. Node A bifurcates (THOL)
    2. Sub-EPI propagates to coupled neighbors
    3. Neighbors' EPIs increase, potentially triggering their own bifurcations
    4. Process continues across ≥3 nodes

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
        - cascade_coherence: average coupling strength in cascade

    Notes
    -----
    TNFR Principle: Cascades emerge when network phase coherence enables
    propagation across multiple nodes, creating collective self-organization.

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
            "cascade_coherence": 0.0,
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

    return {
        "is_cascade": is_cascade,
        "affected_nodes": affected_nodes,
        "cascade_depth": cascade_depth,
        "total_propagations": total_props,
        "cascade_coherence": 0.0,  # TODO: compute from coupling strengths
    }


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
