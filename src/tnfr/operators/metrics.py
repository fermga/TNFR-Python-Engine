"""Operator-specific metrics collection for TNFR structural operators.

Each operator produces characteristic metrics that reflect its structural
effects on nodes. This module provides metric collectors for telemetry
and analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr, get_attr_str
from ..constants.aliases import (
    ALIAS_EPI,
    ALIAS_VF,
    ALIAS_DNFR,
    ALIAS_THETA,
    ALIAS_D2EPI,
    ALIAS_EMISSION_TIMESTAMP,
)

__all__ = [
    "emission_metrics",
    "reception_metrics",
    "coherence_metrics",
    "dissonance_metrics",
    "coupling_metrics",
    "resonance_metrics",
    "silence_metrics",
    "expansion_metrics",
    "contraction_metrics",
    "self_organization_metrics",
    "mutation_metrics",
    "transition_metrics",
    "recursivity_metrics",
]


def _get_node_attr(
    G: TNFRGraph, node: NodeId, aliases: tuple[str, ...], default: float = 0.0
) -> float:
    """Get node attribute using alias fallback."""
    return float(get_attr(G.nodes[node], aliases, default))


def emission_metrics(
    G: TNFRGraph, node: NodeId, epi_before: float, vf_before: float
) -> dict[str, Any]:
    """AL - Emission metrics with structural fidelity indicators.

    Collects emission-specific metrics that reflect canonical AL effects:
    - EPI: Increments (form activation)
    - vf: Activates/increases (Hz_str)
    - DELTA_NFR: Initializes positive reorganization
    - theta: Influences phase alignment

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    epi_before : float
        EPI value before operator application
    vf_before : float
        νf value before operator application

    Returns
    -------
    dict
        Emission-specific metrics including:
        - Core deltas (delta_epi, delta_vf, dnfr_initialized, theta_current)
        - AL-specific quality indicators:
          - emission_quality: "valid" if both EPI and νf increased, else "weak"
          - activation_from_latency: True if node was latent (EPI < 0.3)
          - form_emergence_magnitude: Absolute EPI increment
          - frequency_activation: True if νf increased
          - reorganization_positive: True if ΔNFR > 0
        - Traceability markers:
          - emission_timestamp: ISO UTC timestamp of activation
          - irreversibility_marker: True if node was activated
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    theta = _get_node_attr(G, node, ALIAS_THETA)

    # Fetch emission timestamp using alias system
    emission_timestamp = None
    try:
        emission_timestamp = get_attr_str(
            G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None
        )
    except (AttributeError, KeyError, ImportError):
        # Fallback if alias system unavailable or node lacks timestamp
        emission_timestamp = G.nodes[node].get("emission_timestamp")

    # Compute deltas
    delta_epi = epi_after - epi_before
    delta_vf = vf_after - vf_before

    # AL-specific quality indicators
    emission_quality = "valid" if (delta_epi > 0 and delta_vf > 0) else "weak"
    activation_from_latency = epi_before < 0.3  # Latency threshold
    frequency_activation = delta_vf > 0
    reorganization_positive = dnfr > 0

    # Irreversibility marker
    irreversibility_marker = G.nodes[node].get("_emission_activated", False)

    return {
        "operator": "Emission",
        "glyph": "AL",
        # Core metrics (existing)
        "delta_epi": delta_epi,
        "delta_vf": delta_vf,
        "dnfr_initialized": dnfr,
        "theta_current": theta,
        # Legacy compatibility
        "epi_final": epi_after,
        "vf_final": vf_after,
        "dnfr_final": dnfr,
        "activation_strength": delta_epi,
        "is_activated": epi_after > 0.5,
        # AL-specific (NEW)
        "emission_quality": emission_quality,
        "activation_from_latency": activation_from_latency,
        "form_emergence_magnitude": delta_epi,
        "frequency_activation": frequency_activation,
        "reorganization_positive": reorganization_positive,
        # Traceability (NEW)
        "emission_timestamp": emission_timestamp,
        "irreversibility_marker": irreversibility_marker,
    }


def reception_metrics(G: TNFRGraph, node: NodeId, epi_before: float) -> dict[str, Any]:
    """EN - Reception metrics: EPI integration, source tracking, integration efficiency.

    Extended metrics for Reception (EN) operator that track emission sources,
    phase compatibility, and integration efficiency as specified in TNFR.pdf
    §2.2.1 (EN - Structural reception).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    epi_before : float
        EPI value before operator application

    Returns
    -------
    dict
        Reception-specific metrics including:
        - Core metrics: delta_epi, epi_final, dnfr_after
        - Legacy metrics: neighbor_count, neighbor_epi_mean, integration_strength
        - EN-specific (NEW):
          - num_sources: Number of detected emission sources
          - integration_efficiency: Ratio of integrated to available coherence
          - most_compatible_source: Most phase-compatible source node
          - phase_compatibility_avg: Average phase compatibility with sources
          - coherence_received: Total coherence integrated (delta_epi)
          - stabilization_effective: Whether ΔNFR reduced below threshold
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)

    # Legacy neighbor metrics (backward compatibility)
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Calculate mean neighbor EPI
    neighbor_epi_sum = 0.0
    for n in neighbors:
        neighbor_epi_sum += _get_node_attr(G, n, ALIAS_EPI)
    neighbor_epi_mean = neighbor_epi_sum / neighbor_count if neighbor_count > 0 else 0.0

    # Compute delta EPI (coherence received)
    delta_epi = epi_after - epi_before

    # EN-specific: Source tracking and integration efficiency
    sources = G.nodes[node].get("_reception_sources", [])
    num_sources = len(sources)

    # Calculate total available coherence from sources
    total_available_coherence = sum(strength for _, _, strength in sources)

    # Integration efficiency: ratio of integrated to available coherence
    # Only meaningful if coherence was actually available
    integration_efficiency = (
        delta_epi / total_available_coherence if total_available_coherence > 0 else 0.0
    )

    # Most compatible source (first in sorted list)
    most_compatible_source = sources[0][0] if sources else None

    # Average phase compatibility across all sources
    phase_compatibility_avg = (
        sum(compat for _, compat, _ in sources) / num_sources
        if num_sources > 0
        else 0.0
    )

    # Stabilization effectiveness (ΔNFR reduced?)
    stabilization_effective = dnfr_after < 0.1

    return {
        "operator": "Reception",
        "glyph": "EN",
        # Core metrics
        "delta_epi": delta_epi,
        "epi_final": epi_after,
        "dnfr_after": dnfr_after,
        # Legacy metrics (backward compatibility)
        "neighbor_count": neighbor_count,
        "neighbor_epi_mean": neighbor_epi_mean,
        "integration_strength": abs(delta_epi),
        # EN-specific (NEW)
        "num_sources": num_sources,
        "integration_efficiency": integration_efficiency,
        "most_compatible_source": most_compatible_source,
        "phase_compatibility_avg": phase_compatibility_avg,
        "coherence_received": delta_epi,
        "stabilization_effective": stabilization_effective,
    }


def coherence_metrics(G: TNFRGraph, node: NodeId, dnfr_before: float) -> dict[str, Any]:
    """IL - Coherence metrics: ΔC(t), stability gain, ΔNFR reduction, phase alignment.

    Extended to include ΔNFR reduction percentage, C(t) coherence metrics,
    phase alignment quality, and telemetry from the explicit reduction mechanism
    implemented in the Coherence operator.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    dnfr_before : float
        ΔNFR value before operator application

    Returns
    -------
    dict
        Coherence-specific metrics including:
        - dnfr_before: ΔNFR value before operator
        - dnfr_after: ΔNFR value after operator
        - dnfr_reduction: Absolute reduction (before - after)
        - dnfr_reduction_pct: Percentage reduction relative to before
        - stability_gain: Improvement in stability (reduction of |ΔNFR|)
        - is_stabilized: Whether node reached stable state (|ΔNFR| < 0.1)
        - C_global: Global network coherence (current)
        - C_local: Local neighborhood coherence (current)
        - phase_alignment: Local phase alignment quality (Kuramoto order parameter)
        - phase_coherence_quality: Alias for phase_alignment (for clarity)
        - stabilization_quality: Combined metric (C_local * (1.0 - dnfr_after))
        - epi_final, vf_final: Final structural state
    """
    # Import here to avoid circular import
    from ..metrics.coherence import compute_global_coherence, compute_local_coherence
    from ..metrics.phase_coherence import compute_phase_alignment

    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    vf = _get_node_attr(G, node, ALIAS_VF)

    # Compute reduction metrics
    dnfr_reduction = dnfr_before - dnfr_after
    dnfr_reduction_pct = (
        (dnfr_reduction / dnfr_before * 100.0) if dnfr_before > 0 else 0.0
    )

    # Compute coherence metrics
    C_global = compute_global_coherence(G)
    C_local = compute_local_coherence(G, node)

    # Compute phase alignment (Kuramoto order parameter)
    phase_alignment = compute_phase_alignment(G, node)

    return {
        "operator": "Coherence",
        "glyph": "IL",
        "dnfr_before": dnfr_before,
        "dnfr_after": dnfr_after,
        "dnfr_reduction": dnfr_reduction,
        "dnfr_reduction_pct": dnfr_reduction_pct,
        "dnfr_final": dnfr_after,
        "stability_gain": abs(dnfr_before) - abs(dnfr_after),
        "C_global": C_global,
        "C_local": C_local,
        "phase_alignment": phase_alignment,
        "phase_coherence_quality": phase_alignment,  # Alias for clarity
        "stabilization_quality": C_local * (1.0 - dnfr_after),  # Combined metric
        "epi_final": epi,
        "vf_final": vf,
        "is_stabilized": abs(dnfr_after) < 0.1,  # Configurable threshold
    }


def dissonance_metrics(
    G: TNFRGraph, node: NodeId, dnfr_before: float, theta_before: float
) -> dict[str, Any]:
    """OZ - Comprehensive dissonance and bifurcation metrics.

    Collects extended metrics for the Dissonance (OZ) operator, including
    quantitative bifurcation analysis, topological disruption measures, and
    viable path identification. This aligns with TNFR canonical theory (§2.3.3)
    that OZ introduces **topological dissonance**, not just numerical instability.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    dnfr_before : float
        ΔNFR value before operator application
    theta_before : float
        Phase value before operator application

    Returns
    -------
    dict
        Comprehensive dissonance metrics with keys:

        **Quantitative dynamics:**

        - dnfr_increase: Magnitude of introduced instability
        - dnfr_final: Post-OZ ΔNFR value
        - theta_shift: Phase exploration degree
        - theta_final: Post-OZ phase value
        - d2epi: Structural acceleration (bifurcation indicator)

        **Bifurcation analysis:**

        - bifurcation_score: Quantitative potential [0,1]
        - bifurcation_active: Boolean threshold indicator (score > 0.5)
        - viable_paths: List of viable operator glyph values
        - viable_path_count: Number of viable paths
        - mutation_readiness: Boolean indicator for ZHIR viability

        **Topological effects:**

        - topological_asymmetry_delta: Change in structural asymmetry
        - symmetry_disrupted: Boolean (|delta| > 0.1)

        **Network impact:**

        - neighbor_count: Total neighbors
        - impacted_neighbors: Count with |ΔNFR| > 0.1
        - network_impact_radius: Ratio of impacted neighbors

        **Recovery guidance:**

        - recovery_estimate_IL: Estimated IL applications needed
        - dissonance_level: |ΔNFR| magnitude
        - critical_dissonance: Boolean (|ΔNFR| > 0.8)

    Notes
    -----
    **Enhanced metrics vs original:**

    The original implementation (lines 326-342) provided:
    - Basic ΔNFR change
    - Boolean bifurcation_risk
    - Simple d2epi reading

    This enhanced version adds:
    - Quantitative bifurcation_score [0,1]
    - Viable path identification
    - Topological asymmetry measurement
    - Network impact analysis
    - Recovery estimation

    **Topological asymmetry:**

    Measures structural disruption in the node's ego-network using degree
    and clustering heterogeneity. This captures the canonical effect that
    OZ introduces **topological disruption**, not just numerical change.

    **Viable paths:**

    Identifies which operators can structurally resolve the dissonance:
    - IL (Coherence): Always viable (universal resolution)
    - ZHIR (Mutation): If νf > 0.8 (controlled transformation)
    - NUL (Contraction): If EPI < 0.5 (safe collapse window)
    - THOL (Self-organization): If degree >= 2 (network support)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Dissonance, Coherence
    >>>
    >>> G, node = create_nfr("test", epi=0.5, vf=1.2)
    >>> # Add neighbors for network analysis
    >>> for i in range(3):
    ...     G.add_node(f"n{i}")
    ...     G.add_edge(node, f"n{i}")
    >>>
    >>> # Enable metrics collection
    >>> G.graph['COLLECT_OPERATOR_METRICS'] = True
    >>>
    >>> # Apply Coherence to stabilize, then Dissonance to disrupt
    >>> Coherence()(G, node)
    >>> Dissonance()(G, node)
    >>>
    >>> # Retrieve enhanced metrics
    >>> metrics = G.graph['operator_metrics'][-1]
    >>> print(f"Bifurcation score: {metrics['bifurcation_score']:.2f}")
    >>> print(f"Viable paths: {metrics['viable_paths']}")
    >>> print(f"Network impact: {metrics['network_impact_radius']:.1%}")
    >>> print(f"Recovery estimate: {metrics['recovery_estimate_IL']} IL")

    See Also
    --------
    tnfr.dynamics.bifurcation.compute_bifurcation_score : Bifurcation scoring
    tnfr.topology.asymmetry.compute_topological_asymmetry : Asymmetry measurement
    tnfr.dynamics.bifurcation.get_bifurcation_paths : Viable path identification
    """
    from ..dynamics.bifurcation import compute_bifurcation_score, get_bifurcation_paths
    from ..topology.asymmetry import compute_topological_asymmetry
    from .nodal_equation import compute_d2epi_dt2

    # Get post-OZ node state
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    theta_after = _get_node_attr(G, node, ALIAS_THETA)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)

    # 1. Compute d2epi actively during OZ
    d2epi = compute_d2epi_dt2(G, node)

    # 2. Quantitative bifurcation score (not just boolean)
    bifurcation_threshold = float(G.graph.get("OZ_BIFURCATION_THRESHOLD", 0.5))
    bifurcation_score = compute_bifurcation_score(
        d2epi=d2epi,
        dnfr=dnfr_after,
        vf=vf_after,
        epi=epi_after,
        tau=bifurcation_threshold,
    )

    # 3. Topological asymmetry introduced by OZ
    # Note: We measure asymmetry after OZ. In a full implementation, we'd also
    # capture before state, but for metrics collection we focus on post-state.
    # The delta is captured conceptually (OZ introduces disruption).
    asymmetry_after = compute_topological_asymmetry(G, node)

    # For now, we'll estimate delta based on the assumption that OZ increases asymmetry
    # In a future enhancement, this could be computed by storing asymmetry_before
    asymmetry_delta = asymmetry_after  # Simplified: assume OZ caused current asymmetry

    # 4. Analyze viable post-OZ paths
    # Set bifurcation_ready flag if score exceeds threshold
    if bifurcation_score > 0.5:
        G.nodes[node]["_bifurcation_ready"] = True

    viable_paths = get_bifurcation_paths(G, node)

    # 5. Network impact (neighbors affected by dissonance)
    neighbors = list(G.neighbors(node))
    impacted_neighbors = 0

    if neighbors:
        # Count neighbors with significant |ΔNFR|
        impact_threshold = 0.1
        for n in neighbors:
            neighbor_dnfr = abs(_get_node_attr(G, n, ALIAS_DNFR))
            if neighbor_dnfr > impact_threshold:
                impacted_neighbors += 1

    # 6. Recovery estimate (how many IL needed to resolve)
    # Assumes ~15% ΔNFR reduction per IL application
    il_reduction_rate = 0.15
    recovery_estimate = (
        int(abs(dnfr_after) / il_reduction_rate) + 1 if dnfr_after != 0 else 1
    )

    # 7. Propagation analysis (if propagation occurred)
    propagation_data = {}
    propagation_events = G.graph.get("_oz_propagation_events", [])
    if propagation_events:
        latest_event = propagation_events[-1]
        if latest_event["source"] == node:
            propagation_data = {
                "propagation_occurred": True,
                "affected_neighbors": latest_event["affected_count"],
                "propagation_magnitude": latest_event["magnitude"],
                "affected_nodes": latest_event["affected_nodes"],
            }
        else:
            propagation_data = {"propagation_occurred": False}
    else:
        propagation_data = {"propagation_occurred": False}

    # 8. Compute network dissonance field (if propagation module available)
    field_data = {}
    try:
        from ..dynamics.propagation import compute_network_dissonance_field

        field = compute_network_dissonance_field(G, node, radius=2)
        field_data = {
            "dissonance_field_radius": len(field),
            "max_field_strength": max(field.values()) if field else 0.0,
            "mean_field_strength": sum(field.values()) / len(field) if field else 0.0,
        }
    except (ImportError, Exception):
        # Gracefully handle if propagation module not available
        field_data = {
            "dissonance_field_radius": 0,
            "max_field_strength": 0.0,
            "mean_field_strength": 0.0,
        }

    return {
        "operator": "Dissonance",
        "glyph": "OZ",
        # Quantitative dynamics
        "dnfr_increase": dnfr_after - dnfr_before,
        "dnfr_final": dnfr_after,
        "theta_shift": abs(theta_after - theta_before),
        "theta_final": theta_after,
        "d2epi": d2epi,
        # Bifurcation analysis
        "bifurcation_score": bifurcation_score,
        "bifurcation_active": bifurcation_score > 0.5,
        "viable_paths": [str(g.value) for g in viable_paths],
        "viable_path_count": len(viable_paths),
        "mutation_readiness": any(g.value == "ZHIR" for g in viable_paths),
        # Topological effects
        "topological_asymmetry_delta": asymmetry_delta,
        "symmetry_disrupted": abs(asymmetry_delta) > 0.1,
        # Network impact
        "neighbor_count": len(neighbors),
        "impacted_neighbors": impacted_neighbors,
        "network_impact_radius": (
            impacted_neighbors / len(neighbors) if neighbors else 0.0
        ),
        # Recovery guidance
        "recovery_estimate_IL": recovery_estimate,
        "dissonance_level": abs(dnfr_after),
        "critical_dissonance": abs(dnfr_after) > 0.8,
        # Network propagation
        **propagation_data,
        **field_data,
    }


def coupling_metrics(
    G: TNFRGraph,
    node: NodeId,
    theta_before: float,
    dnfr_before: float = None,
    vf_before: float = None,
    edges_before: int = None,
    epi_before: float = None,
) -> dict[str, Any]:
    """UM - Coupling metrics: phase alignment, link formation, synchrony, ΔNFR reduction.

    Extended metrics for Coupling (UM) operator that track structural changes,
    network formation, and synchronization effectiveness.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    theta_before : float
        Phase value before operator application
    dnfr_before : float, optional
        ΔNFR value before operator application (for reduction tracking)
    vf_before : float, optional
        Structural frequency (νf) before operator application
    edges_before : int, optional
        Number of edges before operator application
    epi_before : float, optional
        EPI value before operator application (for invariance verification)

    Returns
    -------
    dict
        Coupling-specific metrics including:

        **Phase metrics:**

        - theta_shift: Absolute phase change
        - theta_final: Post-coupling phase
        - mean_neighbor_phase: Average phase of neighbors
        - phase_alignment: Alignment with neighbors [0,1]
        - phase_dispersion: Standard deviation of phases in local cluster
        - is_synchronized: Boolean indicating strong synchronization (alignment > 0.8)

        **Frequency metrics:**

        - delta_vf: Change in structural frequency (νf)
        - vf_final: Post-coupling structural frequency

        **Reorganization metrics:**

        - delta_dnfr: Change in ΔNFR
        - dnfr_stabilization: Reduction of reorganization pressure (positive if stabilized)
        - dnfr_final: Post-coupling ΔNFR
        - dnfr_reduction: Absolute reduction (before - after)
        - dnfr_reduction_pct: Percentage reduction

        **EPI Invariance metrics:**

        - epi_before: EPI value before coupling
        - epi_after: EPI value after coupling
        - epi_drift: Absolute difference between before and after
        - epi_preserved: Boolean indicating EPI invariance (drift < 1e-9)

        **Network metrics:**

        - neighbor_count: Number of neighbors after coupling
        - new_edges_count: Number of edges added
        - total_edges: Total edges after coupling
        - coupling_strength_total: Sum of coupling weights on edges
        - local_coherence: Kuramoto order parameter of local subgraph

    Notes
    -----
    The extended metrics align with TNFR canonical theory (§2.2.2) that UM creates
    structural links through phase synchronization (φᵢ(t) ≈ φⱼ(t)). The metrics
    capture both the synchronization quality and the network structural changes
    resulting from coupling.

    **EPI Invariance**: UM MUST preserve EPI identity. The epi_preserved metric
    validates this fundamental invariant. If epi_preserved is False, it indicates
    a violation of TNFR canonical requirements.

    See Also
    --------
    operators.definitions.Coupling : UM operator implementation
    metrics.phase_coherence.compute_phase_alignment : Phase alignment computation
    """
    import math
    import statistics

    theta_after = _get_node_attr(G, node, ALIAS_THETA)
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Calculate phase coherence with neighbors
    if neighbor_count > 0:
        phase_sum = sum(_get_node_attr(G, n, ALIAS_THETA) for n in neighbors)
        mean_neighbor_phase = phase_sum / neighbor_count
        phase_alignment = 1.0 - abs(theta_after - mean_neighbor_phase) / math.pi
    else:
        mean_neighbor_phase = theta_after
        phase_alignment = 0.0

    # Base metrics (always present)
    metrics = {
        "operator": "Coupling",
        "glyph": "UM",
        "theta_shift": abs(theta_after - theta_before),
        "theta_final": theta_after,
        "neighbor_count": neighbor_count,
        "mean_neighbor_phase": mean_neighbor_phase,
        "phase_alignment": max(0.0, phase_alignment),
    }

    # Structural frequency metrics (if vf_before provided)
    if vf_before is not None:
        delta_vf = vf_after - vf_before
        metrics.update(
            {
                "delta_vf": delta_vf,
                "vf_final": vf_after,
            }
        )

    # ΔNFR reduction metrics (if dnfr_before provided)
    if dnfr_before is not None:
        dnfr_reduction = dnfr_before - dnfr_after
        dnfr_reduction_pct = (dnfr_reduction / (abs(dnfr_before) + 1e-9)) * 100.0
        dnfr_stabilization = dnfr_before - dnfr_after  # Positive if stabilized
        metrics.update(
            {
                "dnfr_before": dnfr_before,
                "dnfr_after": dnfr_after,
                "delta_dnfr": dnfr_after - dnfr_before,
                "dnfr_reduction": dnfr_reduction,
                "dnfr_reduction_pct": dnfr_reduction_pct,
                "dnfr_stabilization": dnfr_stabilization,
                "dnfr_final": dnfr_after,
            }
        )

    # EPI invariance verification (if epi_before provided)
    # CRITICAL: UM MUST preserve EPI identity per TNFR canonical theory
    if epi_before is not None:
        epi_after = _get_node_attr(G, node, ALIAS_EPI)
        epi_drift = abs(epi_after - epi_before)
        metrics.update(
            {
                "epi_before": epi_before,
                "epi_after": epi_after,
                "epi_drift": epi_drift,
                "epi_preserved": epi_drift < 1e-9,  # Should ALWAYS be True
            }
        )

    # Edge/network formation metrics (if edges_before provided)
    edges_after = G.degree(node)
    if edges_before is not None:
        new_edges_count = edges_after - edges_before
        metrics.update(
            {
                "new_edges_count": new_edges_count,
                "total_edges": edges_after,
            }
        )
    else:
        # Still provide total_edges even without edges_before
        metrics["total_edges"] = edges_after

    # Coupling strength (sum of edge weights)
    coupling_strength_total = 0.0
    for neighbor in neighbors:
        edge_data = G.get_edge_data(node, neighbor)
        if edge_data and isinstance(edge_data, dict):
            coupling_strength_total += edge_data.get("coupling", 0.0)
    metrics["coupling_strength_total"] = coupling_strength_total

    # Phase dispersion (standard deviation of local phases)
    if neighbor_count > 1:
        phases = [theta_after] + [_get_node_attr(G, n, ALIAS_THETA) for n in neighbors]
        phase_std = statistics.stdev(phases)
        metrics["phase_dispersion"] = phase_std
    else:
        metrics["phase_dispersion"] = 0.0

    # Local coherence (Kuramoto order parameter of subgraph)
    if neighbor_count > 0:
        from ..metrics.phase_coherence import compute_phase_alignment

        local_coherence = compute_phase_alignment(G, node, radius=1)
        metrics["local_coherence"] = local_coherence
    else:
        metrics["local_coherence"] = 0.0

    # Synchronization indicator
    metrics["is_synchronized"] = phase_alignment > 0.8

    return metrics


def resonance_metrics(
    G: TNFRGraph,
    node: NodeId,
    epi_before: float,
    vf_before: float | None = None,
) -> dict[str, Any]:
    """RA - Resonance metrics: EPI propagation, νf amplification, phase strengthening.

    Canonical TNFR resonance metrics include:
    - EPI propagation effectiveness
    - νf amplification (structural frequency increase)
    - Phase alignment strengthening
    - Identity preservation validation
    - Network coherence contribution

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    epi_before : float
        EPI value before operator application
    vf_before : float | None
        νf value before operator application (for amplification tracking)

    Returns
    -------
    dict
        Resonance-specific metrics including:
        - EPI propagation metrics
        - νf amplification ratio (canonical effect)
        - Phase alignment quality
        - Identity preservation status
        - Network coherence contribution
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Calculate resonance strength based on neighbor coupling
    if neighbor_count > 0:
        neighbor_epi_sum = sum(_get_node_attr(G, n, ALIAS_EPI) for n in neighbors)
        neighbor_epi_mean = neighbor_epi_sum / neighbor_count
        resonance_strength = abs(epi_after - epi_before) * neighbor_count

        # Canonical νf amplification tracking
        if vf_before is not None and vf_before > 0:
            vf_amplification = vf_after / vf_before
        else:
            vf_amplification = 1.0

        # Phase alignment quality (measure coherence with neighbors)
        from ..metrics.phase_coherence import compute_phase_alignment

        phase_alignment = compute_phase_alignment(G, node)
    else:
        neighbor_epi_mean = 0.0
        resonance_strength = 0.0
        vf_amplification = 1.0
        phase_alignment = 0.0

    # Identity preservation check (sign should be preserved)
    identity_preserved = epi_before * epi_after >= 0

    return {
        "operator": "Resonance",
        "glyph": "RA",
        "delta_epi": epi_after - epi_before,
        "epi_final": epi_after,
        "epi_before": epi_before,
        "neighbor_count": neighbor_count,
        "neighbor_epi_mean": neighbor_epi_mean,
        "resonance_strength": resonance_strength,
        "propagation_successful": neighbor_count > 0
        and abs(epi_after - neighbor_epi_mean) < 0.5,
        # Canonical TNFR effects
        "vf_amplification": vf_amplification,  # Canonical: νf increases through resonance
        "vf_before": vf_before if vf_before is not None else vf_after,
        "vf_after": vf_after,
        "phase_alignment": phase_alignment,  # Canonical: phase strengthens
        "identity_preserved": identity_preserved,  # Canonical: EPI identity maintained
    }


def _compute_epi_variance(G: TNFRGraph, node: NodeId) -> float:
    """Compute EPI variance during silence period.

    Measures the standard deviation of EPI values recorded during silence,
    validating effective preservation (variance ≈ 0).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to compute variance for

    Returns
    -------
    float
        Standard deviation of EPI during silence period
    """
    import numpy as np

    epi_history = G.nodes[node].get("epi_history_during_silence", [])
    if len(epi_history) < 2:
        return 0.0
    return float(np.std(epi_history))


def _compute_preservation_integrity(preserved_epi: float, epi_after: float) -> float:
    """Compute preservation integrity ratio.

    Measures structural preservation quality as:
        integrity = 1 - |EPI_after - EPI_preserved| / EPI_preserved

    Interpretation:
    - integrity = 1.0: Perfect preservation
    - integrity < 0.95: Significant degradation
    - integrity < 0.8: Preservation failure

    Parameters
    ----------
    preserved_epi : float
        EPI value that was preserved at silence start
    epi_after : float
        Current EPI value

    Returns
    -------
    float
        Preservation integrity in [0, 1]
    """
    if preserved_epi == 0:
        return 1.0 if epi_after == 0 else 0.0

    integrity = 1.0 - abs(epi_after - preserved_epi) / abs(preserved_epi)
    return max(0.0, integrity)


def _compute_reactivation_readiness(G: TNFRGraph, node: NodeId) -> float:
    """Compute readiness score for reactivation from silence.

    Evaluates if the node can reactivate effectively based on:
    - νf residual (must be recoverable)
    - EPI preserved (must be coherent)
    - Silence duration (not excessive)
    - Network connectivity (active neighbors)

    Score in [0, 1]:
    - 1.0: Fully ready to reactivate
    - 0.5-0.8: Moderate readiness
    - < 0.3: Risky reactivation

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to compute readiness for

    Returns
    -------
    float
        Reactivation readiness score in [0, 1]
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    duration = G.nodes[node].get("silence_duration", 0.0)

    # Count active neighbors
    active_neighbors = 0
    if G.has_node(node):
        for n in G.neighbors(node):
            if _get_node_attr(G, n, ALIAS_VF) > 0.1:
                active_neighbors += 1

    # Scoring components
    vf_score = min(vf / 0.5, 1.0)  # νf recoverable
    epi_score = min(epi / 0.3, 1.0)  # EPI coherent
    duration_score = 1.0 / (1.0 + duration * 0.1)  # Penalize long silence
    network_score = min(active_neighbors / 3.0, 1.0)  # Network support

    return (vf_score + epi_score + duration_score + network_score) / 4.0


def _estimate_time_to_collapse(G: TNFRGraph, node: NodeId) -> float:
    """Estimate time until nodal collapse during silence.

    Estimates how long silence can be maintained before structural collapse
    based on observed drift rate or default degradation model.

    Model:
        t_collapse ≈ EPI_preserved / |DRIFT_RATE|

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to estimate collapse time for

    Returns
    -------
    float
        Estimated time steps until collapse (inf if no degradation)
    """
    preserved_epi = G.nodes[node].get("preserved_epi", 0.0)
    drift_rate = G.nodes[node].get("epi_drift_rate", 0.0)

    if abs(drift_rate) < 1e-10:
        # No observed degradation - return large value
        return float("inf")

    if preserved_epi <= 0:
        # Already at or below collapse threshold
        return 0.0

    # Estimate time until EPI reaches zero
    return abs(preserved_epi / drift_rate)


def silence_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """SHA - Silence metrics: νf reduction, EPI preservation, duration tracking.

    Extended metrics for deep analysis of structural preservation effectiveness.
    Collects silence-specific metrics that reflect canonical SHA effects including
    latency state management as specified in TNFR.pdf §2.3.10.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    vf_before : float
        νf value before operator application
    epi_before : float
        EPI value before operator application

    Returns
    -------
    dict
        Silence-specific metrics including:

        **Core metrics (existing):**

        - operator: "Silence"
        - glyph: "SHA"
        - vf_reduction: Absolute reduction in νf
        - vf_final: Post-silence νf value
        - epi_preservation: Absolute EPI change (should be ≈ 0)
        - epi_final: Post-silence EPI value
        - is_silent: Boolean indicating silent state (νf < 0.1)

        **Latency state tracking:**

        - latent: Boolean latency flag
        - silence_duration: Time in silence state (steps or structural time)

        **Extended metrics (NEW):**

        - epi_variance: Standard deviation of EPI during silence
        - preservation_integrity: Quality metric [0, 1] for preservation
        - reactivation_readiness: Readiness score [0, 1] for reactivation
        - time_to_collapse: Estimated time until nodal collapse

    Notes
    -----
    Extended metrics enable:
    - Detection of excessive silence (collapse risk)
    - Validation of preservation quality
    - Analysis of consolidation patterns (memory, learning)
    - Strategic pause effectiveness (biomedical, cognitive, social domains)

    See Also
    --------
    _compute_epi_variance : EPI variance computation
    _compute_preservation_integrity : Preservation quality metric
    _compute_reactivation_readiness : Reactivation readiness score
    _estimate_time_to_collapse : Collapse time estimation
    """
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    preserved_epi = G.nodes[node].get("preserved_epi")

    # Core metrics (existing)
    core = {
        "operator": "Silence",
        "glyph": "SHA",
        "vf_reduction": vf_before - vf_after,
        "vf_final": vf_after,
        "epi_preservation": abs(epi_after - epi_before),
        "epi_final": epi_after,
        "is_silent": vf_after < 0.1,
    }

    # Latency state tracking metrics
    core["latent"] = G.nodes[node].get("latent", False)
    core["silence_duration"] = G.nodes[node].get("silence_duration", 0.0)

    # Extended metrics (new)
    extended = {
        "epi_variance": _compute_epi_variance(G, node),
        "preservation_integrity": (
            _compute_preservation_integrity(preserved_epi, epi_after)
            if preserved_epi is not None
            else 1.0 - abs(epi_after - epi_before)
        ),
        "reactivation_readiness": _compute_reactivation_readiness(G, node),
        "time_to_collapse": _estimate_time_to_collapse(G, node),
    }

    return {**core, **extended}


def expansion_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """VAL - Enhanced expansion metrics with structural indicators (Issue #2724).

    Captures comprehensive metrics reflecting canonical VAL effects:
    - Basic growth metrics (Δνf, ΔEPI)
    - Bifurcation risk (∂²EPI/∂t²)
    - Coherence preservation (local C(t))
    - Fractality indicators (growth ratios)
    - Network impact (phase coherence with neighbors)
    - Structural stability (ΔNFR bounds)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    vf_before : float
        νf value before operator application
    epi_before : float
        EPI value before operator application

    Returns
    -------
    dict
        Comprehensive expansion metrics including:

        **Core Metrics (existing)**:
        - operator, glyph: Identification
        - vf_increase, vf_final: Frequency changes
        - delta_epi, epi_final: EPI changes
        - expansion_factor: Relative νf increase

        **Structural Stability (NEW)**:
        - dnfr_final: Final reorganization gradient
        - dnfr_positive: True if ΔNFR > 0 (required for expansion)
        - dnfr_stable: True if 0 < ΔNFR < 1.0 (bounded growth)

        **Bifurcation Risk (ENHANCED)**:
        - d2epi: EPI acceleration (∂²EPI/∂t²)
        - bifurcation_risk: True when |∂²EPI/∂t²| > threshold
        - bifurcation_magnitude: Ratio of d2epi to threshold
        - bifurcation_threshold: Configurable threshold value

        **Coherence Preservation (ENHANCED)**:
        - coherence_local: Local coherence measurement [0,1]
        - coherence_preserved: True when C_local > threshold

        **Fractality Indicators (ENHANCED)**:
        - epi_growth_rate: Relative EPI growth
        - vf_growth_rate: Relative νf growth
        - growth_ratio: vf_growth_rate / epi_growth_rate
        - fractal_preserved: True when ratio in valid range [0.5, 2.0]

        **Network Impact (NEW)**:
        - neighbor_count: Number of neighbors
        - phase_coherence_neighbors: Phase alignment with neighbors [0,1]
        - network_coupled: True if neighbors exist and phase_coherence > 0.5
        - theta_final: Final phase value

        **Overall Health (NEW)**:
        - expansion_healthy: Combined indicator of all health metrics

    Notes
    -----
    Key indicators:
    - bifurcation_risk: True when |∂²EPI/∂t²| > threshold
    - fractal_preserved: True when growth rates maintain scaling relationship
    - coherence_preserved: True when local C(t) remains above threshold
    - dnfr_positive: True when ΔNFR > 0 (required for expansion)

    Thresholds are configurable via graph metadata:
    - VAL_BIFURCATION_THRESHOLD (default: 0.3)
    - VAL_MIN_COHERENCE (default: 0.5)
    - VAL_FRACTAL_RATIO_MIN (default: 0.5)
    - VAL_FRACTAL_RATIO_MAX (default: 2.0)

    Examples
    --------
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Expansion
    >>>
    >>> G, node = create_nfr("test", epi=0.4, vf=1.0)
    >>> G.graph["COLLECT_OPERATOR_METRICS"] = True
    >>> run_sequence(G, node, [Expansion()])
    >>>
    >>> metrics = G.graph["operator_metrics"][-1]
    >>> if metrics["bifurcation_risk"]:
    ...     print(f"WARNING: Bifurcation risk! d2epi={metrics['d2epi']:.3f}")
    >>> if not metrics["coherence_preserved"]:
    ...     print(f"WARNING: Coherence degraded! C={metrics['coherence_local']:.3f}")

    See Also
    --------
    Expansion : VAL operator that produces these metrics
    validate_expansion : Preconditions ensuring valid expansion
    """
    import math

    # Basic state
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    d2epi = _get_node_attr(G, node, ALIAS_D2EPI)
    theta = _get_node_attr(G, node, ALIAS_THETA)

    # Network context
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Thresholds (configurable)
    bifurcation_threshold = float(G.graph.get("VAL_BIFURCATION_THRESHOLD", 0.3))
    coherence_threshold = float(G.graph.get("VAL_MIN_COHERENCE", 0.5))
    fractal_ratio_min = float(G.graph.get("VAL_FRACTAL_RATIO_MIN", 0.5))
    fractal_ratio_max = float(G.graph.get("VAL_FRACTAL_RATIO_MAX", 2.0))

    # Growth deltas
    delta_epi = epi_after - epi_before
    delta_vf = vf_after - vf_before

    # Growth rates (relative to initial values)
    epi_growth_rate = (delta_epi / epi_before) if epi_before > 1e-9 else 0.0
    vf_growth_rate = (delta_vf / vf_before) if vf_before > 1e-9 else 0.0
    growth_ratio = (
        vf_growth_rate / epi_growth_rate if abs(epi_growth_rate) > 1e-9 else 0.0
    )

    # Coherence preservation
    try:
        from ..metrics.coherence import compute_local_coherence

        c_local = compute_local_coherence(G, node)
    except (ImportError, AttributeError):
        # Fallback if coherence module not available
        c_local = 0.0

    # Phase coherence with neighbors
    if neighbor_count > 0:
        neighbor_theta_sum = sum(_get_node_attr(G, n, ALIAS_THETA) for n in neighbors)
        mean_neighbor_theta = neighbor_theta_sum / neighbor_count
        phase_diff = abs(theta - mean_neighbor_theta)
        # Normalize to [0, 1], 1 = perfect alignment
        phase_coherence_neighbors = 1.0 - min(phase_diff, math.pi) / math.pi
    else:
        phase_coherence_neighbors = 0.0

    # Bifurcation magnitude (ratio to threshold)
    bifurcation_magnitude = (
        abs(d2epi) / bifurcation_threshold if bifurcation_threshold > 0 else 0.0
    )

    # Boolean indicators
    bifurcation_risk = abs(d2epi) > bifurcation_threshold
    coherence_preserved = c_local > coherence_threshold
    dnfr_positive = dnfr > 0
    dnfr_stable = 0 < dnfr < 1.0
    fractal_preserved = (
        fractal_ratio_min < growth_ratio < fractal_ratio_max
        if abs(epi_growth_rate) > 1e-9
        else True
    )
    network_coupled = neighbor_count > 0 and phase_coherence_neighbors > 0.5

    # Overall health indicator
    expansion_healthy = (
        dnfr_positive
        and not bifurcation_risk
        and coherence_preserved
        and fractal_preserved
    )

    return {
        # Core identification
        "operator": "Expansion",
        "glyph": "VAL",
        # Existing basic metrics
        "vf_increase": delta_vf,
        "vf_final": vf_after,
        "delta_epi": delta_epi,
        "epi_final": epi_after,
        "expansion_factor": vf_after / vf_before if vf_before > 1e-9 else 1.0,
        # NEW: Structural stability
        "dnfr_final": dnfr,
        "dnfr_positive": dnfr_positive,
        "dnfr_stable": dnfr_stable,
        # NEW: Bifurcation risk (enhanced)
        "d2epi": d2epi,
        "bifurcation_risk": bifurcation_risk,
        "bifurcation_magnitude": bifurcation_magnitude,
        "bifurcation_threshold": bifurcation_threshold,
        # NEW: Coherence preservation
        "coherence_local": c_local,
        "coherence_preserved": coherence_preserved,
        # NEW: Fractality indicators
        "epi_growth_rate": epi_growth_rate,
        "vf_growth_rate": vf_growth_rate,
        "growth_ratio": growth_ratio,
        "fractal_preserved": fractal_preserved,
        # NEW: Network impact
        "neighbor_count": neighbor_count,
        "phase_coherence_neighbors": max(0.0, phase_coherence_neighbors),
        "network_coupled": network_coupled,
        "theta_final": theta,
        # NEW: Overall health
        "expansion_healthy": expansion_healthy,
        # Metadata
        "metrics_version": "3.0_canonical",
    }


def contraction_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """NUL - Contraction metrics: νf decrease, core concentration, ΔNFR densification.

    Collects comprehensive contraction metrics including structural density dynamics
    that validate canonical NUL behavior and enable early warning for over-compression.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    vf_before : float
        νf value before operator application
    epi_before : float
        EPI value before operator application

    Returns
    -------
    dict
        Contraction-specific metrics including:

        **Basic metrics:**

        - operator: "Contraction"
        - glyph: "NUL"
        - vf_decrease: Absolute reduction in νf
        - vf_final: Post-contraction νf
        - delta_epi: EPI change
        - epi_final: Post-contraction EPI
        - dnfr_final: Post-contraction ΔNFR
        - contraction_factor: Ratio of vf_after / vf_before

        **Densification metrics (if available):**

        - densification_factor: ΔNFR amplification factor (typically 1.35)
        - dnfr_densified: Boolean indicating densification occurred
        - dnfr_before: ΔNFR value before contraction
        - dnfr_increase: Absolute ΔNFR change (dnfr_after - dnfr_before)

        **Structural density metrics (NEW):**

        - density_before: |ΔNFR| / max(EPI, ε) before contraction
        - density_after: |ΔNFR| / max(EPI, ε) after contraction
        - densification_ratio: density_after / density_before
        - is_critical_density: Warning flag (density > threshold)

    Notes
    -----
    **Structural Density**: Defined as ρ = |ΔNFR| / max(EPI, ε) where ε = 1e-9.
    This captures the concentration of reorganization pressure per unit structure.

    **Critical Density**: When density exceeds CRITICAL_DENSITY_THRESHOLD (default: 5.0),
    it indicates over-compression risk where the node may become unstable.

    **Densification Ratio**: Quantifies how much density increased during contraction.
    Canonical NUL should produce densification_ratio ≈ densification_factor / contraction_factor.

    See Also
    --------
    Contraction : NUL operator implementation
    validate_contraction : Preconditions for safe contraction
    """
    # Small epsilon for numerical stability
    EPSILON = 1e-9
    
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)

    # Extract densification telemetry if available
    densification_log = G.graph.get("nul_densification_log", [])
    densification_factor = None
    dnfr_before = None
    if densification_log:
        # Get the most recent densification entry for this node
        last_entry = densification_log[-1]
        densification_factor = last_entry.get("densification_factor")
        dnfr_before = last_entry.get("dnfr_before")

    # Calculate structural density before and after
    # Density = |ΔNFR| / max(EPI, ε)
    density_before = abs(dnfr_before) / max(abs(epi_before), EPSILON) if dnfr_before is not None else 0.0
    density_after = abs(dnfr_after) / max(abs(epi_after), EPSILON)
    
    # Calculate densification ratio (how much density increased)
    densification_ratio = density_after / density_before if density_before > EPSILON else float('inf')
    
    # Get critical density threshold from graph config or use default
    critical_density_threshold = float(G.graph.get("CRITICAL_DENSITY_THRESHOLD", 5.0))
    is_critical_density = density_after > critical_density_threshold

    metrics = {
        "operator": "Contraction",
        "glyph": "NUL",
        "vf_decrease": vf_before - vf_after,
        "vf_final": vf_after,
        "delta_epi": epi_after - epi_before,
        "epi_final": epi_after,
        "dnfr_final": dnfr_after,
        "contraction_factor": vf_after / vf_before if vf_before > 0 else 1.0,
    }
    
    # Add densification metrics if available
    if densification_factor is not None:
        metrics["densification_factor"] = densification_factor
        metrics["dnfr_densified"] = True
    if dnfr_before is not None:
        metrics["dnfr_before"] = dnfr_before
        metrics["dnfr_increase"] = dnfr_after - dnfr_before if dnfr_before else 0.0
    
    # Add NEW structural density metrics
    metrics["density_before"] = density_before
    metrics["density_after"] = density_after
    metrics["densification_ratio"] = densification_ratio
    metrics["is_critical_density"] = is_critical_density
    
    return metrics


def self_organization_metrics(
    G: TNFRGraph, node: NodeId, epi_before: float, vf_before: float
) -> dict[str, Any]:
    """THOL - Enhanced metrics with cascade dynamics and collective coherence.

    Collects comprehensive THOL metrics including bifurcation, cascade propagation,
    collective coherence of sub-EPIs, and metabolic activity indicators.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    epi_before : float
        EPI value before operator application
    vf_before : float
        νf value before operator application

    Returns
    -------
    dict
        Self-organization-specific metrics including:

        **Base operator metrics:**

        - operator: "Self-organization"
        - glyph: "THOL"
        - delta_epi: Change in EPI
        - delta_vf: Change in νf
        - epi_final: Final EPI value
        - vf_final: Final νf value
        - d2epi: Structural acceleration
        - dnfr_final: Final ΔNFR

        **Bifurcation metrics:**

        - bifurcation_occurred: Boolean indicator
        - nested_epi_count: Number of sub-EPIs created
        - d2epi_magnitude: Absolute acceleration

        **Cascade dynamics (NEW):**

        - cascade_depth: Maximum hierarchical bifurcation depth
        - propagation_radius: Total unique nodes affected
        - cascade_detected: Boolean cascade indicator
        - affected_node_count: Nodes reached by cascade
        - total_propagations: Total propagation events

        **Collective coherence (NEW):**

        - subepi_coherence: Coherence of sub-EPI ensemble [0,1]
        - metabolic_activity_index: Network context usage [0,1]

        **Network emergence indicator (NEW):**

        - network_emergence: Combined indicator (cascade + high coherence)

    Notes
    -----
    TNFR Principle: Complete traceability of self-organization dynamics.
    These metrics enable reconstruction of entire cascade evolution,
    validation of controlled emergence, and identification of collective
    network phenomena.

    See Also
    --------
    operators.metabolism.compute_cascade_depth : Cascade depth computation
    operators.metabolism.compute_subepi_collective_coherence : Coherence metric
    operators.metabolism.compute_metabolic_activity_index : Metabolic tracking
    operators.cascade.detect_cascade : Cascade detection
    """
    from .cascade import detect_cascade
    from .metabolism import (
        compute_cascade_depth,
        compute_propagation_radius,
        compute_subepi_collective_coherence,
        compute_metabolic_activity_index,
    )

    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    d2epi = _get_node_attr(G, node, ALIAS_D2EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)

    # Track nested EPI count from node attribute or graph (backward compatibility)
    nested_epi_count = len(G.nodes[node].get("sub_epis", []))
    if nested_epi_count == 0:
        # Fallback to old location for backward compatibility
        nested_epi_count = len(G.graph.get("sub_epi", []))

    # Cascade and propagation analysis
    cascade_analysis = detect_cascade(G)

    # NEW: Enhanced cascade and emergence metrics
    cascade_depth = compute_cascade_depth(G, node)
    propagation_radius = compute_propagation_radius(G)
    subepi_coherence = compute_subepi_collective_coherence(G, node)
    metabolic_activity = compute_metabolic_activity_index(G, node)

    return {
        # Base operator metrics
        "operator": "Self-organization",
        "glyph": "THOL",
        "delta_epi": epi_after - epi_before,
        "delta_vf": vf_after - vf_before,
        "epi_final": epi_after,
        "vf_final": vf_after,
        "d2epi": d2epi,
        "dnfr_final": dnfr,
        # Bifurcation metrics
        "bifurcation_occurred": nested_epi_count > 0,
        "nested_epi_count": nested_epi_count,
        "d2epi_magnitude": abs(d2epi),
        # NEW: Cascade dynamics
        "cascade_depth": cascade_depth,
        "propagation_radius": propagation_radius,
        "cascade_detected": cascade_analysis["is_cascade"],
        "affected_node_count": len(cascade_analysis["affected_nodes"]),
        "total_propagations": cascade_analysis["total_propagations"],
        # NEW: Collective coherence
        "subepi_coherence": subepi_coherence,
        "metabolic_activity_index": metabolic_activity,
        # NEW: Network emergence indicator
        "network_emergence": (
            cascade_analysis["is_cascade"] and subepi_coherence > 0.5
        ),
    }


def mutation_metrics(
    G: TNFRGraph,
    node: NodeId,
    theta_before: float,
    epi_before: float,
    vf_before: float | None = None,
    dnfr_before: float | None = None,
) -> dict[str, Any]:
    """ZHIR - Comprehensive mutation metrics with canonical structural indicators.

    Collects extended metrics reflecting canonical ZHIR effects:
    - Threshold verification (∂EPI/∂t > ξ)
    - Phase transformation quality (θ → θ')
    - Bifurcation potential (∂²EPI/∂t² > τ)
    - Structural identity preservation
    - Network impact and propagation
    - Destabilizer context (R4 Extended)
    - Grammar validation status

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    theta_before : float
        Phase value before operator application
    epi_before : float
        EPI value before operator application
    vf_before : float, optional
        νf before mutation (for frequency shift tracking)
    dnfr_before : float, optional
        ΔNFR before mutation (for pressure tracking)

    Returns
    -------
    dict
        Comprehensive mutation metrics organized by category:

        **Core metrics (existing):**

        - operator, glyph: Identification
        - theta_shift, theta_final: Phase changes
        - delta_epi, epi_final: EPI changes
        - phase_change: Boolean indicator

        **Threshold verification (ENHANCED):**

        - depi_dt: Structural velocity (∂EPI/∂t)
        - threshold_xi: Configured threshold
        - threshold_met: Boolean (∂EPI/∂t > ξ)
        - threshold_ratio: depi_dt / ξ
        - threshold_exceeded_by: max(0, depi_dt - ξ)

        **Phase transformation (ENHANCED):**

        - theta_regime_before: Initial phase regime [0-3]
        - theta_regime_after: Final phase regime [0-3]
        - regime_changed: Boolean regime transition
        - theta_shift_direction: +1 (forward) or -1 (backward)
        - phase_transformation_magnitude: Normalized shift [0, 1]

        **Bifurcation analysis (NEW):**

        - d2epi: Structural acceleration
        - bifurcation_threshold_tau: Configured τ
        - bifurcation_potential: Boolean (∂²EPI/∂t² > τ)
        - bifurcation_score: Quantitative potential [0, 1]
        - bifurcation_triggered: Boolean (event recorded)
        - bifurcation_event_count: Number of bifurcation events

        **Structural preservation (NEW):**

        - epi_kind_before: Identity before mutation
        - epi_kind_after: Identity after mutation
        - identity_preserved: Boolean (must be True)
        - delta_vf: Change in structural frequency
        - vf_final: Final νf
        - delta_dnfr: Change in reorganization pressure
        - dnfr_final: Final ΔNFR

        **Network impact (NEW):**

        - neighbor_count: Number of neighbors
        - impacted_neighbors: Count with phase shift detected
        - network_impact_radius: Ratio of impacted neighbors
        - phase_coherence_neighbors: Phase alignment after mutation

        **Destabilizer context (NEW - R4 Extended):**

        - destabilizer_type: "strong"/"moderate"/"weak"/None
        - destabilizer_operator: Glyph that enabled mutation
        - destabilizer_distance: Operators since destabilizer
        - recent_history: Last 4 operators

        **Grammar validation (NEW):**

        - grammar_u4b_satisfied: Boolean (IL precedence + destabilizer)
        - il_precedence_found: Boolean (IL in history)
        - destabilizer_recent: Boolean (within window)

    Examples
    --------
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence, Dissonance, Mutation
    >>>
    >>> G, node = create_nfr("test", epi=0.5, vf=1.2)
    >>> G.graph["COLLECT_OPERATOR_METRICS"] = True
    >>>
    >>> # Apply canonical sequence (IL → OZ → ZHIR)
    >>> run_sequence(G, node, [Coherence(), Dissonance(), Mutation()])
    >>>
    >>> # Retrieve comprehensive metrics
    >>> metrics = G.graph["operator_metrics"][-1]
    >>> print(f"Threshold met: {metrics['threshold_met']}")
    >>> print(f"Bifurcation score: {metrics['bifurcation_score']:.2f}")
    >>> print(f"Identity preserved: {metrics['identity_preserved']}")
    >>> print(f"Grammar satisfied: {metrics['grammar_u4b_satisfied']}")

    See Also
    --------
    operators.definitions.Mutation : ZHIR operator implementation
    dynamics.bifurcation.compute_bifurcation_score : Bifurcation scoring
    operators.preconditions.validate_mutation : Precondition validation with context tracking
    """
    import math

    # === GET POST-MUTATION STATE ===
    theta_after = _get_node_attr(G, node, ALIAS_THETA)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    d2epi = _get_node_attr(G, node, ALIAS_D2EPI, 0.0)

    # === THRESHOLD VERIFICATION ===
    # Compute ∂EPI/∂t from history
    epi_history = G.nodes[node].get("epi_history") or G.nodes[node].get(
        "_epi_history", []
    )
    if len(epi_history) >= 2:
        depi_dt = abs(epi_history[-1] - epi_history[-2])
    else:
        depi_dt = 0.0

    xi = float(G.graph.get("ZHIR_THRESHOLD_XI", 0.1))
    threshold_met = depi_dt >= xi
    threshold_ratio = depi_dt / xi if xi > 0 else 0.0

    # === PHASE TRANSFORMATION ===
    # Extract transformation telemetry from glyph storage
    theta_shift_stored = G.nodes[node].get("_zhir_theta_shift", None)
    regime_changed = G.nodes[node].get("_zhir_regime_changed", False)
    regime_before_stored = G.nodes[node].get("_zhir_regime_before", None)
    regime_after_stored = G.nodes[node].get("_zhir_regime_after", None)
    fixed_mode = G.nodes[node].get("_zhir_fixed_mode", False)

    # Compute theta shift
    theta_shift = theta_after - theta_before
    theta_shift_magnitude = abs(theta_shift)

    # Compute regimes if not stored
    regime_before = (
        regime_before_stored
        if regime_before_stored is not None
        else int(theta_before // (math.pi / 2))
    )
    regime_after = (
        regime_after_stored
        if regime_after_stored is not None
        else int(theta_after // (math.pi / 2))
    )

    # Normalized phase transformation magnitude [0, 1]
    phase_transformation_magnitude = min(theta_shift_magnitude / math.pi, 1.0)

    # === BIFURCATION ANALYSIS ===
    tau = float(
        G.graph.get(
            "BIFURCATION_THRESHOLD_TAU", G.graph.get("ZHIR_BIFURCATION_THRESHOLD", 0.5)
        )
    )
    bifurcation_potential = d2epi > tau

    # Compute bifurcation score using canonical formula
    from ..dynamics.bifurcation import compute_bifurcation_score

    bifurcation_score = compute_bifurcation_score(
        d2epi=d2epi, dnfr=dnfr_after, vf=vf_after, epi=epi_after, tau=tau
    )

    # Check if bifurcation was triggered (event recorded)
    bifurcation_events = G.graph.get("zhir_bifurcation_events", [])
    bifurcation_triggered = len(bifurcation_events) > 0
    bifurcation_event_count = len(bifurcation_events)

    # === STRUCTURAL PRESERVATION ===
    epi_kind_before = G.nodes[node].get("_epi_kind_before")
    epi_kind_after = G.nodes[node].get("epi_kind")
    identity_preserved = (
        epi_kind_before == epi_kind_after if epi_kind_before is not None else True
    )

    # Track frequency and pressure changes
    delta_vf = vf_after - vf_before if vf_before is not None else 0.0
    delta_dnfr = dnfr_after - dnfr_before if dnfr_before is not None else 0.0

    # === NETWORK IMPACT ===
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Count neighbors that experienced phase shifts
    # This is a simplified heuristic - we check if neighbors have recent phase changes
    impacted_neighbors = 0
    phase_impact_threshold = 0.1

    if neighbor_count > 0:
        # Check neighbors for phase alignment/disruption
        for n in neighbors:
            neighbor_theta = _get_node_attr(G, n, ALIAS_THETA)
            # Simplified: check if neighbor is in similar phase regime after mutation
            phase_diff = abs(neighbor_theta - theta_after)
            # If phase diff is large, neighbor might be impacted
            if phase_diff > phase_impact_threshold:
                # Check if neighbor has changed recently (has history)
                neighbor_theta_history = G.nodes[n].get("theta_history", [])
                if len(neighbor_theta_history) >= 2:
                    neighbor_change = abs(
                        neighbor_theta_history[-1] - neighbor_theta_history[-2]
                    )
                    if neighbor_change > 0.05:  # Neighbor experienced change
                        impacted_neighbors += 1

        # Phase coherence with neighbors after mutation
        from ..metrics.phase_coherence import compute_phase_alignment

        phase_coherence = compute_phase_alignment(G, node, radius=1)
    else:
        phase_coherence = 0.0

    # === DESTABILIZER CONTEXT (R4 Extended) ===
    mutation_context = G.nodes[node].get("_mutation_context", {})
    destabilizer_type = mutation_context.get("destabilizer_type")
    destabilizer_operator = mutation_context.get("destabilizer_operator")
    destabilizer_distance = mutation_context.get("destabilizer_distance")
    recent_history = mutation_context.get("recent_history", [])

    # === GRAMMAR VALIDATION (U4b) ===
    # Check if U4b satisfied (IL precedence + recent destabilizer)
    glyph_history = G.nodes[node].get("glyph_history", [])

    # Look for IL in history
    il_precedence_found = any("IL" in str(g) for g in glyph_history)

    # Check if destabilizer is recent (within ~3 operators)
    destabilizer_recent = (
        destabilizer_distance is not None and destabilizer_distance <= 3
    )

    grammar_u4b_satisfied = il_precedence_found and destabilizer_recent

    # === RETURN COMPREHENSIVE METRICS ===
    return {
        # === CORE (existing) ===
        "operator": "Mutation",
        "glyph": "ZHIR",
        "theta_shift": theta_shift_magnitude,
        "theta_shift_signed": (
            theta_shift_stored if theta_shift_stored is not None else theta_shift
        ),
        "theta_before": theta_before,
        "theta_after": theta_after,
        "theta_final": theta_after,
        "phase_change": theta_shift_magnitude > 0.5,  # Configurable threshold
        "transformation_mode": "fixed" if fixed_mode else "canonical",
        # === THRESHOLD VERIFICATION (ENHANCED) ===
        "depi_dt": depi_dt,
        "threshold_xi": xi,
        "threshold_met": threshold_met,
        "threshold_ratio": threshold_ratio,
        "threshold_exceeded_by": max(0.0, depi_dt - xi),
        "threshold_warning": G.nodes[node].get("_zhir_threshold_warning", False),
        "threshold_validated": G.nodes[node].get("_zhir_threshold_met", False),
        "threshold_unknown": G.nodes[node].get("_zhir_threshold_unknown", False),
        # === PHASE TRANSFORMATION (ENHANCED) ===
        "theta_regime_before": regime_before,
        "theta_regime_after": regime_after,
        "regime_changed": regime_changed or (regime_before != regime_after),
        "theta_regime_change": regime_changed or (regime_before != regime_after),  # Backwards compat
        "regime_before": regime_before,  # Backwards compat
        "regime_after": regime_after,  # Backwards compat
        "theta_shift_direction": math.copysign(1.0, theta_shift),
        "phase_transformation_magnitude": phase_transformation_magnitude,
        # === BIFURCATION ANALYSIS (NEW) ===
        "d2epi": d2epi,
        "bifurcation_threshold_tau": tau,
        "bifurcation_potential": bifurcation_potential,
        "bifurcation_score": bifurcation_score,
        "bifurcation_triggered": bifurcation_triggered,
        "bifurcation_event_count": bifurcation_event_count,
        # === EPI METRICS ===
        "delta_epi": epi_after - epi_before,
        "epi_before": epi_before,
        "epi_after": epi_after,
        "epi_final": epi_after,
        # === STRUCTURAL PRESERVATION (NEW) ===
        "epi_kind_before": epi_kind_before,
        "epi_kind_after": epi_kind_after,
        "identity_preserved": identity_preserved,
        "delta_vf": delta_vf,
        "vf_before": vf_before if vf_before is not None else vf_after,
        "vf_final": vf_after,
        "delta_dnfr": delta_dnfr,
        "dnfr_before": dnfr_before if dnfr_before is not None else dnfr_after,
        "dnfr_final": dnfr_after,
        # === NETWORK IMPACT (NEW) ===
        "neighbor_count": neighbor_count,
        "impacted_neighbors": impacted_neighbors,
        "network_impact_radius": (
            impacted_neighbors / neighbor_count if neighbor_count > 0 else 0.0
        ),
        "phase_coherence_neighbors": phase_coherence,
        # === DESTABILIZER CONTEXT (NEW - R4 Extended) ===
        "destabilizer_type": destabilizer_type,
        "destabilizer_operator": destabilizer_operator,
        "destabilizer_distance": destabilizer_distance,
        "recent_history": recent_history,
        # === GRAMMAR VALIDATION (NEW) ===
        "grammar_u4b_satisfied": grammar_u4b_satisfied,
        "il_precedence_found": il_precedence_found,
        "destabilizer_recent": destabilizer_recent,
        # === METADATA ===
        "metrics_version": "2.0_canonical",
    }


def transition_metrics(
    G: TNFRGraph,
    node: NodeId,
    dnfr_before: float,
    vf_before: float,
    theta_before: float,
    epi_before: float | None = None,
) -> dict[str, Any]:
    """NAV - Transition metrics: regime classification, phase shift, frequency scaling.

    Collects comprehensive transition metrics including regime origin/destination,
    phase shift magnitude (properly wrapped), transition type classification, and
    structural preservation ratios as specified in TNFR.pdf Table 2.3.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    dnfr_before : float
        ΔNFR value before operator application
    vf_before : float
        νf value before operator application
    theta_before : float
        Phase value before operator application
    epi_before : float, optional
        EPI value before operator application (for preservation tracking)

    Returns
    -------
    dict
        Transition-specific metrics including:

        **Core metrics (existing)**:

        - operator: "Transition"
        - glyph: "NAV"
        - delta_theta: Signed phase change
        - delta_vf: Change in νf
        - delta_dnfr: Change in ΔNFR
        - dnfr_final: Final ΔNFR value
        - vf_final: Final νf value
        - theta_final: Final phase value
        - transition_complete: Boolean (|ΔNFR| < |νf|)

        **Regime classification (NEW)**:

        - regime_origin: "latent" | "active" | "resonant"
        - regime_destination: "latent" | "active" | "resonant"
        - transition_type: "reactivation" | "phase_shift" | "regime_change"

        **Phase metrics (NEW)**:

        - phase_shift_magnitude: Absolute phase change (radians, 0-π)
        - phase_shift_signed: Signed phase change (radians, wrapped to [-π, π])

        **Structural scaling (NEW)**:

        - vf_scaling_factor: vf_after / vf_before
        - dnfr_damping_ratio: dnfr_after / dnfr_before
        - epi_preservation: epi_after / epi_before (if epi_before provided)

        **Latency tracking (NEW)**:

        - latency_duration: Time in silence (seconds) if transitioning from SHA

    Notes
    -----
    **Regime Classification**:
    
    - **Latent**: latent flag set OR νf < 0.05
    - **Active**: Default operational state
    - **Resonant**: EPI > 0.5 AND νf > 0.8

    **Transition Type**:

    - **reactivation**: From latent state (SHA → NAV flow)
    - **phase_shift**: Significant phase change (|Δθ| > 0.3 rad)
    - **regime_change**: Regime switch without significant phase shift

    **Phase Shift Wrapping**:

    Phase shifts are properly wrapped to [-π, π] range to handle 0-2π boundary
    crossings correctly, ensuring accurate phase change measurement.

    Examples
    --------
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Silence, Transition
    >>> 
    >>> # Example: SHA → NAV reactivation
    >>> G, node = create_nfr("test", epi=0.5, vf=0.8)
    >>> G.graph["COLLECT_OPERATOR_METRICS"] = True
    >>> run_sequence(G, node, [Silence(), Transition()])
    >>> 
    >>> metrics = G.graph["operator_metrics"][-1]
    >>> assert metrics["operator"] == "Transition"
    >>> assert metrics["transition_type"] == "reactivation"
    >>> assert metrics["regime_origin"] == "latent"
    >>> assert metrics["latency_duration"] is not None

    See Also
    --------
    operators.definitions.Transition : NAV operator implementation
    operators.definitions.Transition._detect_regime : Regime detection logic
    """
    import math

    # Get current state (after transformation)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    theta_after = _get_node_attr(G, node, ALIAS_THETA)

    # === REGIME CLASSIFICATION ===
    # Get regime origin from node attribute (stored by Transition operator before super().__call__)
    regime_origin = G.nodes[node].get("_regime_before", None)
    if regime_origin is None:
        # Fallback: detect regime from before state
        regime_origin = _detect_regime_from_state(
            epi_before or epi_after, vf_before, False  # Cannot access latent flag from before
        )

    # Detect destination regime
    regime_destination = _detect_regime_from_state(
        epi_after, vf_after, G.nodes[node].get("latent", False)
    )

    # === TRANSITION TYPE CLASSIFICATION ===
    # Calculate phase shift (properly wrapped)
    phase_shift_raw = theta_after - theta_before
    if phase_shift_raw > math.pi:
        phase_shift_raw -= 2 * math.pi
    elif phase_shift_raw < -math.pi:
        phase_shift_raw += 2 * math.pi

    # Classify transition type
    if regime_origin == "latent":
        transition_type = "reactivation"
    elif abs(phase_shift_raw) > 0.3:
        transition_type = "phase_shift"
    else:
        transition_type = "regime_change"

    # === STRUCTURAL SCALING FACTORS ===
    vf_scaling = vf_after / vf_before if vf_before > 0 else 1.0
    dnfr_damping = dnfr_after / dnfr_before if abs(dnfr_before) > 1e-9 else 1.0

    # === EPI PRESERVATION ===
    epi_preservation = None
    if epi_before is not None and epi_before > 0:
        epi_preservation = epi_after / epi_before

    # === LATENCY DURATION ===
    # Get from node if transitioning from silence
    latency_duration = G.nodes[node].get("silence_duration", None)

    return {
        # === CORE (existing, preserved) ===
        "operator": "Transition",
        "glyph": "NAV",
        "delta_theta": phase_shift_raw,
        "delta_vf": vf_after - vf_before,
        "delta_dnfr": dnfr_after - dnfr_before,
        "dnfr_final": dnfr_after,
        "vf_final": vf_after,
        "theta_final": theta_after,
        "transition_complete": abs(dnfr_after) < abs(vf_after),
        # Legacy compatibility
        "dnfr_change": abs(dnfr_after - dnfr_before),
        "vf_change": abs(vf_after - vf_before),
        "theta_shift": abs(phase_shift_raw),
        # === REGIME CLASSIFICATION (NEW) ===
        "regime_origin": regime_origin,
        "regime_destination": regime_destination,
        "transition_type": transition_type,
        # === PHASE METRICS (NEW) ===
        "phase_shift_magnitude": abs(phase_shift_raw),
        "phase_shift_signed": phase_shift_raw,
        # === STRUCTURAL SCALING (NEW) ===
        "vf_scaling_factor": vf_scaling,
        "dnfr_damping_ratio": dnfr_damping,
        "epi_preservation": epi_preservation,
        # === LATENCY TRACKING (NEW) ===
        "latency_duration": latency_duration,
    }


def _detect_regime_from_state(epi: float, vf: float, latent: bool) -> str:
    """Detect structural regime from node state.

    Helper function for transition_metrics to classify regime without
    accessing the Transition operator directly.

    Parameters
    ----------
    epi : float
        EPI value
    vf : float
        νf value
    latent : bool
        Latent flag

    Returns
    -------
    str
        Regime classification: "latent", "active", or "resonant"

    Notes
    -----
    Matches logic in Transition._detect_regime (definitions.py).
    """
    if latent or vf < 0.05:
        return "latent"
    elif epi > 0.5 and vf > 0.8:
        return "resonant"
    else:
        return "active"


def recursivity_metrics(
    G: TNFRGraph, node: NodeId, epi_before: float, vf_before: float
) -> dict[str, Any]:
    """REMESH - Recursivity metrics: fractal propagation, multi-scale coherence.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    epi_before : float
        EPI value before operator application
    vf_before : float
        νf value before operator application

    Returns
    -------
    dict
        Recursivity-specific metrics including fractal pattern indicators
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)

    # Track echo traces if graph maintains them
    echo_traces = G.graph.get("echo_trace", [])
    echo_count = len(echo_traces)

    return {
        "operator": "Recursivity",
        "glyph": "REMESH",
        "delta_epi": epi_after - epi_before,
        "delta_vf": vf_after - vf_before,
        "epi_final": epi_after,
        "vf_final": vf_after,
        "echo_count": echo_count,
        "fractal_depth": echo_count,
        "multi_scale_active": echo_count > 0,
    }
