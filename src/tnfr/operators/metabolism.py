"""Vibrational metabolism functions for THOL (Self-organization) operator.

Implements canonical pattern digestion: capturing external network signals
and transforming them into internal structural reorganization (ΔNFR and sub-EPIs).

TNFR Canonical Principle
-------------------------
From "El pulso que nos atraviesa" (TNFR Manual, §2.2.10):

    "THOL es el glifo de la autoorganización activa. No necesita intervención
    externa, ni programación, ni control — su función es reorganizar la forma
    desde dentro, en respuesta a la coherencia vibracional del campo."

    "THOL no es una propiedad, es una dinámica. No es un atributo de lo vivo,
    es lo que hace que algo esté vivo. La autoorganización no es espontaneidad
    aleatoria, es resonancia estructurada desde el interior del nodo."

This module operationalizes vibrational metabolism:
1. **Capture**: Sample network environment (EPI gradient, phase variance, coupling)
2. **Metabolize**: Transform external patterns into internal structure (sub-EPIs)
3. **Integrate**: Sub-EPIs reflect both internal acceleration AND network context

Metabolic Formula
-----------------
sub-EPI = base_internal + network_contribution + complexity_bonus

Where:
- base_internal: parent_epi * scaling_factor (internal bifurcation)
- network_contribution: epi_gradient * weight (external pressure)
- complexity_bonus: phase_variance * weight (field complexity)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA
from ..utils import get_numpy
from ..utils.numeric import angle_diff

__all__ = [
    "capture_network_signals",
    "metabolize_signals_into_subepi",
    "propagate_subepi_to_network",
    "compute_cascade_depth",
    "compute_hierarchical_depth",
    "compute_propagation_radius",
    "compute_subepi_collective_coherence",
    "compute_metabolic_activity_index",
]


def capture_network_signals(G: TNFRGraph, node: NodeId) -> dict[str, Any] | None:
    """Capture external vibrational patterns from coupled neighbors.

    This function implements the "perception" phase of THOL's vibrational metabolism.
    It samples the network environment around the target node, computing structural
    gradients, phase variance, and coupling strength.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node and its network context
    node : NodeId
        Node performing metabolic capture

    Returns
    -------
    dict | None
        Network signal structure containing:
        - epi_gradient: Difference between mean neighbor EPI and node EPI
        - phase_variance: Variance of neighbor phases (instability indicator)
        - neighbor_count: Number of coupled neighbors
        - coupling_strength_mean: Average phase alignment with neighbors
        - mean_neighbor_epi: Mean EPI value of neighbors
        Returns None if node has no neighbors (isolated metabolism).

    Notes
    -----
    TNFR Principle: THOL doesn't operate in vacuum—it metabolizes the network's
    vibrational field. EPI gradient represents "structural pressure" from environment.
    Phase variance indicates "complexity" of external patterns to digest.

    Examples
    --------
    >>> # Node with coherent neighbors (low variance)
    >>> signals = capture_network_signals(G, node)
    >>> signals["phase_variance"]  # Low = stable field
    0.02

    >>> # Node in dissonant field (high variance)
    >>> signals = capture_network_signals(G_dissonant, node)
    >>> signals["phase_variance"]  # High = complex field
    0.45
    """
    np = get_numpy()
    from ..metrics.phase_compatibility import compute_phase_coupling_strength

    neighbors = list(G.neighbors(node))
    if not neighbors:
        return None

    node_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    node_theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

    # Aggregate neighbor states
    neighbor_epis = []
    neighbor_thetas = []
    coupling_strengths = []

    for n in neighbors:
        n_epi = float(get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        n_theta = float(get_attr(G.nodes[n], ALIAS_THETA, 0.0))

        neighbor_epis.append(n_epi)
        neighbor_thetas.append(n_theta)

        # Coupling strength using canonical phase compatibility formula
        # (unified across UM, RA, THOL operators - see phase_compatibility module)
        coupling_strength = compute_phase_coupling_strength(node_theta, n_theta)
        coupling_strengths.append(coupling_strength)

    # Compute structural gradients
    mean_neighbor_epi = float(np.mean(neighbor_epis))
    epi_gradient = mean_neighbor_epi - node_epi

    # Phase variance (complexity/dissonance indicator)
    phase_variance = float(np.var(neighbor_thetas))

    # Mean coupling strength
    coupling_strength_mean = float(np.mean(coupling_strengths))

    return {
        "epi_gradient": epi_gradient,
        "phase_variance": phase_variance,
        "neighbor_count": len(neighbors),
        "coupling_strength_mean": coupling_strength_mean,
        "mean_neighbor_epi": mean_neighbor_epi,
    }


def metabolize_signals_into_subepi(
    parent_epi: float,
    signals: dict[str, Any] | None,
    d2_epi: float,
    scaling_factor: float = 0.25,
    gradient_weight: float = 0.15,
    complexity_weight: float = 0.10,
) -> float:
    """Transform external signals into sub-EPI structure through metabolism.

    This function implements the "digestion" phase of THOL's vibrational metabolism.
    It combines internal acceleration (d²EPI/dt²) with external network pressure
    to compute the magnitude of emergent sub-EPI.

    Parameters
    ----------
    parent_epi : float
        Current EPI magnitude of parent node
    signals : dict | None
        Network signals captured from environment (from capture_network_signals).
        If None, falls back to internal bifurcation only.
    d2_epi : float
        Internal structural acceleration (∂²EPI/∂t²)
    scaling_factor : float, default 0.25
        Canonical THOL sub-EPI scaling (0.25 = 25% of parent)
    gradient_weight : float, default 0.15
        Weight for external EPI gradient contribution
    complexity_weight : float, default 0.10
        Weight for phase variance complexity bonus

    Returns
    -------
    float
        Metabolized sub-EPI magnitude, bounded to [0, 1.0]

    Notes
    -----
    TNFR Metabolic Formula:

    sub-EPI = base_internal + network_contribution + complexity_bonus

    Where:
    - base_internal: parent_epi * scaling_factor (internal bifurcation)
    - network_contribution: epi_gradient * weight (external pressure)
    - complexity_bonus: phase_variance * weight (field complexity)

    This reflects canonical principle: "THOL reorganizes external experience
    into internal structure without external instruction" (Manual TNFR, p. 112).

    Examples
    --------
    >>> # Internal bifurcation only (isolated node)
    >>> metabolize_signals_into_subepi(0.60, None, d2_epi=0.15)
    0.15

    >>> # Metabolizing network pressure
    >>> signals = {"epi_gradient": 0.20, "phase_variance": 0.10, ...}
    >>> metabolize_signals_into_subepi(0.60, signals, d2_epi=0.15)
    0.21  # Enhanced by network context
    """
    np = get_numpy()

    # Base: Internal bifurcation (existing behavior)
    base_sub_epi = parent_epi * scaling_factor

    # If isolated, return internal bifurcation only
    if signals is None:
        return float(np.clip(base_sub_epi, 0.0, 1.0))

    # Network contribution: EPI gradient pressure
    network_contribution = signals["epi_gradient"] * gradient_weight

    # Complexity bonus: Phase variance indicates rich field to metabolize
    complexity_bonus = signals["phase_variance"] * complexity_weight

    # Combine internal + external
    metabolized_epi = base_sub_epi + network_contribution + complexity_bonus

    # Structural bounds [0, 1]
    return float(np.clip(metabolized_epi, 0.0, 1.0))


def propagate_subepi_to_network(
    G: TNFRGraph,
    parent_node: NodeId,
    sub_epi_record: dict[str, Any],
) -> list[tuple[NodeId, float]]:
    """Propagate emergent sub-EPI to coupled neighbors through resonance.

    Implements canonical THOL network dynamics: bifurcation creates structures
    that propagate through coupled nodes, triggering potential cascades.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the network
    parent_node : NodeId
        Node where sub-EPI originated (bifurcation source)
    sub_epi_record : dict
        Sub-EPI record from bifurcation, containing:
        - "epi": sub-EPI magnitude
        - "vf": inherited structural frequency
        - "timestamp": creation time

    Returns
    -------
    list of (NodeId, float)
        List of (neighbor_id, injected_epi) tuples showing propagation results.
        Empty list if no propagation occurred.

    Notes
    -----
    TNFR Principle: "Sub-EPIs propagate to coupled neighbors, triggering their
    own bifurcations when ∂²EPI/∂t² > τ" (canonical THOL dynamics).

    **AGENTS.md Invariant #5**: No coupling is valid without explicit phase
    verification. This function enforces phase compatibility before propagation:
    - Computes coupling_strength = 1.0 - (|Δθ| / π) using angle_diff()
    - Rejects neighbors with coupling_strength < threshold (antiphase blocked)
    - Ensures resonance physics: only phase-aligned nodes receive sub-EPIs

    Propagation mechanism:
    1. Select neighbors with sufficient coupling (phase alignment)
    2. Compute attenuation based on coupling strength
    3. Inject attenuated sub-EPI influence into neighbor's EPI
    4. Record propagation in graph telemetry

    Attenuation prevents unbounded growth while enabling cascades.

    Examples
    --------
    >>> # Create coupled network
    >>> G = nx.Graph()
    >>> G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
    >>> G.add_node(1, epi=0.40, vf=1.0, theta=0.12)  # Phase-aligned
    >>> G.add_edge(0, 1)
    >>> sub_epi = {"epi": 0.15, "vf": 1.1, "timestamp": 10}
    >>> propagations = propagate_subepi_to_network(G, node=0, sub_epi_record=sub_epi)
    >>> len(propagations)  # Number of neighbors reached
    1
    >>> propagations[0]  # (neighbor_id, injected_epi)
    (1, 0.105)  # 70% attenuation
    """
    from ..alias import set_attr

    neighbors = list(G.neighbors(parent_node))
    if not neighbors:
        return []

    # Configuration
    min_coupling_strength = float(G.graph.get("THOL_MIN_COUPLING_FOR_PROPAGATION", 0.5))
    attenuation_factor = float(G.graph.get("THOL_PROPAGATION_ATTENUATION", 0.7))

    parent_theta = float(get_attr(G.nodes[parent_node], ALIAS_THETA, 0.0))
    sub_epi_magnitude = sub_epi_record["epi"]

    propagations = []

    for neighbor in neighbors:
        neighbor_theta = float(get_attr(G.nodes[neighbor], ALIAS_THETA, 0.0))

        # INVARIANT #5: Phase verification before coupling
        # Compute coupling strength based on phase alignment
        # coupling_strength ∈ [0, 1]: 1 = in-phase, 0 = antiphase
        phase_diff = abs(angle_diff(neighbor_theta, parent_theta))
        coupling_strength = 1.0 - (phase_diff / math.pi)

        # Propagate only if sufficiently coupled (phase-aligned)
        # Antiphase neighbors (Δθ ≈ π) have coupling_strength ≈ 0, blocked by threshold
        if coupling_strength >= min_coupling_strength:
            # Attenuate sub-EPI based on distance and coupling
            attenuated_epi = sub_epi_magnitude * attenuation_factor * coupling_strength

            # Inject into neighbor's EPI
            neighbor_epi = float(get_attr(G.nodes[neighbor], ALIAS_EPI, 0.0))
            new_neighbor_epi = neighbor_epi + attenuated_epi

            # Boundary check
            from ..dynamics.structural_clip import structural_clip

            epi_max = float(G.graph.get("EPI_MAX", 1.0))
            new_neighbor_epi = structural_clip(new_neighbor_epi, lo=0.0, hi=epi_max)

            set_attr(G.nodes[neighbor], ALIAS_EPI, new_neighbor_epi)

            propagations.append((neighbor, attenuated_epi))

            # Update neighbor's EPI history for potential subsequent bifurcation
            history = G.nodes[neighbor].get("epi_history", [])
            history.append(new_neighbor_epi)
            G.nodes[neighbor]["epi_history"] = history[-10:]  # Keep last 10

    return propagations


def compute_cascade_depth(G: TNFRGraph, node: NodeId) -> int:
    """Compute maximum hierarchical depth of bifurcation cascade.

    Recursively measures how many levels of nested sub-EPIs exist,
    where each sub-EPI can itself bifurcate into deeper levels.

    With architectural refactor: sub-EPIs are now independent NFR nodes,
    enabling true recursive depth computation.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing bifurcation history
    node : NodeId
        Root node of cascade analysis

    Returns
    -------
    int
        Maximum cascade depth (0 if no bifurcation occurred)

    Examples
    --------
    >>> # Single-level bifurcation
    >>> compute_cascade_depth(G, node)
    1

    >>> # Multi-level cascade (sub-EPIs bifurcated further)
    >>> compute_cascade_depth(G_complex, node)
    3

    Notes
    -----
    TNFR Principle: Cascade depth measures the hierarchical complexity
    of emergent self-organization. Depth = 1 indicates direct bifurcation;
    depth > 1 indicates recursive, multi-scale emergence.

    ARCHITECTURAL: Now supports true recursive bifurcation with independent
    sub-nodes. If a node has no independent sub-nodes, falls back to
    metadata-based depth for backward compatibility.
    """
    # Primary path: Check for independent sub-nodes (new architecture)
    sub_nodes = G.nodes[node].get("sub_nodes", [])
    if sub_nodes:
        # Recursive computation with actual nodes
        max_depth = 0
        for sub_node_id in sub_nodes:
            if sub_node_id in G.nodes:
                # Recurse into child's depth
                child_depth = compute_cascade_depth(G, sub_node_id)
                max_depth = max(max_depth, 1 + child_depth)
            else:
                # Child node exists in list but not in graph (shouldn't happen)
                max_depth = max(max_depth, 1)
        return max_depth

    # Fallback path: Legacy metadata-based depth
    sub_epis = G.nodes[node].get("sub_epis", [])
    if not sub_epis:
        return 0

    max_depth = 1
    for sub in sub_epis:
        # Check if sub-EPI has node_id (new architecture with metadata)
        if "node_id" in sub and sub["node_id"] in G.nodes:
            # Recurse into independent node
            child_depth = compute_cascade_depth(G, sub["node_id"])
            max_depth = max(max_depth, 1 + child_depth)
        else:
            # Legacy metadata-only mode
            nested_depth = sub.get("cascade_depth", 0)
            max_depth = max(max_depth, 1 + nested_depth)

    return max_depth


def compute_hierarchical_depth(G: TNFRGraph, node: NodeId) -> int:
    """Compute maximum bifurcation depth from node using recursive traversal.

    Traverses sub-EPI hierarchy recursively to find the maximum bifurcation_level
    across all nested branches. This provides accurate hierarchical telemetry for
    nested THOL bifurcations, supporting operational fractality analysis.

    Parameters
    ----------
    G : TNFRGraph
        Network graph
    node : NodeId
        Root node to measure depth from

    Returns
    -------
    int
        Maximum bifurcation depth (0 = no bifurcations, 1 = single-level, etc.)

    Notes
    -----
    TNFR Principle: Hierarchical depth reflects operational fractality
    (Invariant #7) - the ability of sub-EPIs to bifurcate recursively,
    creating multi-scale emergent structures.

    This function recursively traverses all branches to find the deepest
    bifurcation_level, providing precise depth tracking for:
    - Debugging complex nested structures
    - Validating depth limits
    - Analyzing bifurcation patterns
    - Performance monitoring

    Examples
    --------
    >>> # Node with no bifurcations
    >>> compute_hierarchical_depth(G, node)
    0

    >>> # Node with single-level bifurcation
    >>> compute_hierarchical_depth(G, node_with_subs)
    1

    >>> # Node with 2-level nested bifurcation
    >>> compute_hierarchical_depth(G, node_nested)
    2
    """
    sub_epis = G.nodes[node].get("sub_epis", [])

    if not sub_epis:
        return 0

    # Recursively find the maximum bifurcation_level across all branches
    max_level = 0
    for sub_epi in sub_epis:
        # Get this sub-EPI's bifurcation level
        level = sub_epi.get("bifurcation_level", 1)
        max_level = max(max_level, level)

        # Recurse into sub-node if it exists to find deeper levels
        sub_node_id = sub_epi.get("node_id")
        if sub_node_id and sub_node_id in G.nodes:
            # Recursively check this sub-node's depth
            sub_depth = compute_hierarchical_depth(G, sub_node_id)
            # If sub-node has bifurcations, its depth represents deeper levels
            if sub_depth > 0:
                max_level = max(max_level, sub_depth)

    return max_level


def compute_propagation_radius(G: TNFRGraph) -> int:
    """Count total unique nodes affected by THOL cascades.

    Parameters
    ----------
    G : TNFRGraph
        Graph with THOL propagation history

    Returns
    -------
    int
        Number of nodes reached by at least one propagation event

    Notes
    -----
    TNFR Principle: Propagation radius measures the spatial extent
    of cascade effects across the network. High radius indicates
    network-wide self-organization.

    Examples
    --------
    >>> # Local cascade (few nodes)
    >>> compute_propagation_radius(G_local)
    3

    >>> # Network-wide cascade
    >>> compute_propagation_radius(G_wide)
    15
    """
    propagations = G.graph.get("thol_propagations", [])
    affected_nodes = set()

    for prop in propagations:
        affected_nodes.add(prop["source_node"])
        for target, _ in prop["propagations"]:
            affected_nodes.add(target)

    return len(affected_nodes)


def compute_subepi_collective_coherence(G: TNFRGraph, node: NodeId) -> float:
    """Calculate coherence of sub-EPI ensemble.

    Measures how structurally aligned the emergent sub-EPIs are.
    Low variance = high coherence = stable emergence.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node with sub-EPIs to analyze

    Returns
    -------
    float
        Coherence metric [0, 1] where 1 = perfect alignment

    Notes
    -----
    Uses variance-based coherence:
    C_sub = 1 / (1 + var(sub_epi_magnitudes))

    TNFR Principle: Coherent bifurcation produces sub-EPIs with similar
    structural magnitudes, indicating controlled emergence vs chaotic
    fragmentation.

    Examples
    --------
    >>> # Coherent bifurcation (similar sub-EPIs)
    >>> compute_subepi_collective_coherence(G, node)
    0.85

    >>> # Chaotic fragmentation (varied sub-EPIs)
    >>> compute_subepi_collective_coherence(G_chaotic, node)
    0.23
    """
    np = get_numpy()

    sub_epis = G.nodes[node].get("sub_epis", [])
    if len(sub_epis) < 2:
        return 0.0  # Need ≥2 sub-EPIs to measure coherence

    epi_values = [sub["epi"] for sub in sub_epis]
    variance = float(np.var(epi_values))

    # Coherence: inverse relationship with variance
    coherence = 1.0 / (1.0 + variance)
    return coherence


def compute_metabolic_activity_index(G: TNFRGraph, node: NodeId) -> float:
    """Measure proportion of sub-EPIs generated through network metabolism.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to analyze

    Returns
    -------
    float
        Ratio [0, 1] of metabolized sub-EPIs to total sub-EPIs
        1.0 = all sub-EPIs included network context
        0.0 = all sub-EPIs were purely internal bifurcations

    Notes
    -----
    TNFR Principle: Metabolic activity measures how much network context
    influenced bifurcation. High index indicates external pressure drove
    emergence; low index indicates internal acceleration dominated.

    Examples
    --------
    >>> # Network-driven bifurcation
    >>> compute_metabolic_activity_index(G_coupled, node)
    0.90

    >>> # Internal-only bifurcation
    >>> compute_metabolic_activity_index(G_isolated, node)
    0.0
    """
    sub_epis = G.nodes[node].get("sub_epis", [])
    if not sub_epis:
        return 0.0

    metabolized_count = sum(1 for sub in sub_epis if sub.get("metabolized", False))
    return metabolized_count / len(sub_epis)
