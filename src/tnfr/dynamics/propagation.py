"""Network propagation dynamics for OZ-induced dissonance.

This module implements propagation of dissonance across network neighbors
following TNFR resonance principles. When OZ (Dissonance) is applied to a node,
structural dissonance propagates through the network based on phase compatibility,
frequency matching, and coupling strength.

According to TNFR canonical theory:
    "Nodal interference: Dissonance between nodes that disrupts coherence.
    Can induce reorganization or collapse."

OZ introduces topological asymmetry that propagates beyond the local node,
potentially triggering bifurcation cascades in phase-compatible neighbors.

References
----------
- TNFR.pdf §2.3.3: OZ introduces topological dissonance
- Issue: [OZ] Implement dissonance propagation and neighborhood network effects
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF

__all__ = [
    "propagate_dissonance",
    "compute_network_dissonance_field",
    "detect_bifurcation_cascade",
]


def propagate_dissonance(
    G: TNFRGraph,
    source_node: NodeId,
    dissonance_magnitude: float,
    propagation_mode: str = "phase_weighted",
) -> set[NodeId]:
    """Propagate OZ-induced dissonance to phase-compatible neighbors.

    When OZ is applied to a node, structural dissonance propagates through
    the network following TNFR resonance principles:

    1. **Phase compatibility**: Neighbors with |Δθ| < threshold receive more
    2. **Frequency matching**: Higher νf neighbors respond more strongly
    3. **Coupling strength**: Edge weights modulate propagation
    4. **Distance decay**: Effect diminishes with topological distance

    Parameters
    ----------
    G : TNFRGraph
        Network containing nodes
    source_node : NodeId
        Node where OZ was applied
    dissonance_magnitude : float
        |ΔNFR| increase at source (typically from OZ metrics)
    propagation_mode : str
        'phase_weighted' (default), 'uniform', 'frequency_weighted'

    Returns
    -------
    set[NodeId]
        Set of affected neighbor nodes

    Notes
    -----
    Propagation follows coupling physics:

    ΔNFR_neighbor = ΔNFR_source * w_coupling * w_phase * w_frequency

    Where:
    - w_coupling: Edge weight (default 1.0)
    - w_phase: Phase compatibility factor
    - w_frequency: Frequency matching factor

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Emission, Dissonance
    >>> from tnfr.dynamics.propagation import propagate_dissonance
    >>>
    >>> G, node0 = create_nfr("source", epi=0.5, vf=1.0)
    >>> # Add neighbors
    >>> for i in range(3):
    ...     G.add_node(f"n{i}")
    ...     G.add_edge(node0, f"n{i}")
    ...     Emission()(G, f"n{i}")
    >>>
    >>> # Apply OZ and propagate
    >>> Dissonance()(G, node0)
    >>> affected = propagate_dissonance(G, node0, 0.15)
    >>> print(f"Affected neighbors: {len(affected)}")

    See Also
    --------
    compute_network_dissonance_field : Compute field with distance decay
    detect_bifurcation_cascade : Detect cascade-triggered bifurcations
    """
    neighbors = list(G.neighbors(source_node))
    if not neighbors:
        return set()

    affected = set()
    source_theta = float(get_attr(G.nodes[source_node], ALIAS_THETA, 0.0))
    source_vf = float(get_attr(G.nodes[source_node], ALIAS_VF, 1.0))

    # Propagation threshold (configurable)
    phase_threshold = float(G.graph.get("OZ_PHASE_THRESHOLD", math.pi / 2))
    min_propagation = float(G.graph.get("OZ_MIN_PROPAGATION", 0.05))

    for neighbor in neighbors:
        neighbor_theta = float(get_attr(G.nodes[neighbor], ALIAS_THETA, 0.0))
        neighbor_vf = float(get_attr(G.nodes[neighbor], ALIAS_VF, 1.0))
        neighbor_dnfr = float(get_attr(G.nodes[neighbor], ALIAS_DNFR, 0.0))

        # Compute phase compatibility
        delta_theta = abs(source_theta - neighbor_theta)
        if delta_theta > phase_threshold:
            continue  # Phase incompatible, no propagation

        phase_weight = 1.0 - (delta_theta / phase_threshold)

        # Compute frequency matching
        if propagation_mode == "frequency_weighted":
            freq_ratio = min(neighbor_vf, source_vf) / max(neighbor_vf, source_vf, 1e-10)
            freq_weight = freq_ratio
        else:
            freq_weight = 1.0

        # Get edge weight (coupling strength)
        edge_data = G.get_edge_data(source_node, neighbor)
        coupling_weight = edge_data.get("weight", 1.0) if edge_data else 1.0

        # Compute propagated dissonance
        propagated_dnfr = dissonance_magnitude * coupling_weight * phase_weight * freq_weight

        if abs(propagated_dnfr) >= min_propagation:
            # Apply propagated dissonance to neighbor
            new_dnfr = neighbor_dnfr + propagated_dnfr
            # Use first alias for consistency
            G.nodes[neighbor][ALIAS_DNFR[0]] = new_dnfr
            affected.add(neighbor)

            # Log propagation for telemetry
            if "_oz_propagation" not in G.nodes[neighbor]:
                G.nodes[neighbor]["_oz_propagation"] = []
            G.nodes[neighbor]["_oz_propagation"].append(
                {
                    "from_node": source_node,
                    "magnitude": propagated_dnfr,
                    "phase_weight": phase_weight,
                    "coupling_weight": coupling_weight,
                }
            )

    return affected


def compute_network_dissonance_field(
    G: TNFRGraph,
    source_node: NodeId,
    radius: int = 2,
) -> dict[NodeId, float]:
    """Compute dissonance field propagation up to radius hops.

    Returns dict mapping node -> dissonance_level for all nodes
    within radius hops of source.

    Parameters
    ----------
    G : TNFRGraph
        Network
    source_node : NodeId
        OZ application point
    radius : int
        Maximum propagation distance (default 2)

    Returns
    -------
    dict[NodeId, float]
        Mapping of affected nodes to dissonance level

    Notes
    -----
    Uses exponential decay: dissonance_level = source_dnfr * (0.5 ** distance)

    Only nodes reachable via paths (connected) are included in the field.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Dissonance
    >>> from tnfr.dynamics.propagation import compute_network_dissonance_field
    >>>
    >>> G, node0 = create_nfr("source")
    >>> # Create path topology: 0-1-2-3
    >>> for i in range(1, 4):
    ...     G.add_node(i)
    ...     G.add_edge(i-1, i)
    >>>
    >>> Dissonance()(G, node0)
    >>> field = compute_network_dissonance_field(G, node0, radius=2)
    >>> # Returns: {1: high, 2: medium} (node 3 beyond radius)
    """
    import networkx as nx

    field = {}
    source_dnfr = abs(float(get_attr(G.nodes[source_node], ALIAS_DNFR, 0.0)))

    # Get decay factor from graph config
    decay_factor = float(G.graph.get("OZ_DECAY_FACTOR", 0.5))

    # BFS to propagate with distance decay
    for distance in range(1, radius + 1):
        # Get nodes at this distance
        nodes_at_distance = set()
        for node in G.nodes():
            try:
                path_length = nx.shortest_path_length(G, source_node, node)
                if path_length == distance:
                    nodes_at_distance.add(node)
            except nx.NetworkXNoPath:
                continue

        # Propagate with decay
        decay = decay_factor**distance
        for node in nodes_at_distance:
            field[node] = source_dnfr * decay

    return field


def detect_bifurcation_cascade(
    G: TNFRGraph,
    source_node: NodeId,
    threshold: float = 0.5,
) -> list[NodeId]:
    """Detect if OZ triggers bifurcation cascade in network.

    When source node undergoes bifurcation (∂²EPI/∂t² > τ), check if
    propagated dissonance pushes neighbors over their own thresholds.

    Parameters
    ----------
    G : TNFRGraph
        Network containing nodes
    source_node : NodeId
        Node where OZ was applied
    threshold : float
        Bifurcation threshold τ (default 0.5)

    Returns
    -------
    list[NodeId]
        Nodes that entered bifurcation state due to cascade

    Notes
    -----
    A node is considered in bifurcation cascade if:
    - It received propagated dissonance from source
    - Its ∂²EPI/∂t² now exceeds threshold τ

    The function marks cascade nodes with `_bifurcation_cascade` metadata
    for telemetry and further analysis.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Emission, Dissonance
    >>> from tnfr.dynamics.propagation import detect_bifurcation_cascade
    >>>
    >>> G, node0 = create_nfr("source", epi=0.5, vf=1.2)
    >>> # Add neighbors with EPI history
    >>> for i in range(3):
    ...     G.add_node(f"n{i}")
    ...     G.add_edge(node0, f"n{i}")
    ...     G.nodes[f"n{i}"]["_epi_history"] = [0.3, 0.45, 0.55]
    >>>
    >>> Dissonance()(G, node0, propagate_to_network=True)
    >>> cascade = detect_bifurcation_cascade(G, node0)
    >>> print(f"Cascade size: {len(cascade)}")

    See Also
    --------
    tnfr.operators.nodal_equation.compute_d2epi_dt2 : Compute structural acceleration
    tnfr.dynamics.bifurcation.get_bifurcation_paths : Identify viable paths
    """
    from ..operators.nodal_equation import compute_d2epi_dt2

    cascade_nodes = []

    # Get neighbors affected by propagation
    neighbors = list(G.neighbors(source_node))

    for neighbor in neighbors:
        # Check if neighbor has propagation record (was affected)
        if "_oz_propagation" not in G.nodes[neighbor]:
            continue

        # Check if neighbor now in bifurcation state
        d2epi_neighbor = compute_d2epi_dt2(G, neighbor)

        if abs(d2epi_neighbor) > threshold:
            cascade_nodes.append(neighbor)

            # Mark for telemetry
            G.nodes[neighbor]["_bifurcation_cascade"] = {
                "triggered_by": source_node,
                "d2epi": d2epi_neighbor,
                "threshold": threshold,
            }

            # Set bifurcation_ready flag for path detection
            G.nodes[neighbor]["_bifurcation_ready"] = True

    return cascade_nodes
