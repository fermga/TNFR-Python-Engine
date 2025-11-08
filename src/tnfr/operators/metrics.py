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
    §2.2.1 (EN - Recepción estructural).

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
        sum(compat for _, compat, _ in sources) / num_sources if num_sources > 0 else 0.0
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
    recovery_estimate = int(abs(dnfr_after) / il_reduction_rate) + 1 if dnfr_after != 0 else 1
    
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
        "network_impact_radius": impacted_neighbors / len(neighbors) if neighbors else 0.0,
        
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
        metrics.update({
            "delta_vf": delta_vf,
            "vf_final": vf_after,
        })

    # ΔNFR reduction metrics (if dnfr_before provided)
    if dnfr_before is not None:
        dnfr_reduction = dnfr_before - dnfr_after
        dnfr_reduction_pct = (dnfr_reduction / (abs(dnfr_before) + 1e-9)) * 100.0
        dnfr_stabilization = dnfr_before - dnfr_after  # Positive if stabilized
        metrics.update({
            "dnfr_before": dnfr_before,
            "dnfr_after": dnfr_after,
            "delta_dnfr": dnfr_after - dnfr_before,
            "dnfr_reduction": dnfr_reduction,
            "dnfr_reduction_pct": dnfr_reduction_pct,
            "dnfr_stabilization": dnfr_stabilization,
            "dnfr_final": dnfr_after,
        })

    # EPI invariance verification (if epi_before provided)
    # CRITICAL: UM MUST preserve EPI identity per TNFR canonical theory
    if epi_before is not None:
        epi_after = _get_node_attr(G, node, ALIAS_EPI)
        epi_drift = abs(epi_after - epi_before)
        metrics.update({
            "epi_before": epi_before,
            "epi_after": epi_after,
            "epi_drift": epi_drift,
            "epi_preserved": epi_drift < 1e-9,  # Should ALWAYS be True
        })

    # Edge/network formation metrics (if edges_before provided)
    edges_after = G.degree(node)
    if edges_before is not None:
        new_edges_count = edges_after - edges_before
        metrics.update({
            "new_edges_count": new_edges_count,
            "total_edges": edges_after,
        })
    else:
        # Still provide total_edges even without edges_before
        metrics["total_edges"] = edges_after

    # Coupling strength (sum of edge weights)
    coupling_strength_total = 0.0
    for neighbor in neighbors:
        edge_data = G.get_edge_data(node, neighbor)
        if edge_data and isinstance(edge_data, dict):
            coupling_strength_total += edge_data.get('coupling', 0.0)
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
    identity_preserved = (epi_before * epi_after >= 0)

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


def silence_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """SHA - Silence metrics: νf reduction, EPI preservation, and latency tracking.

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
        - Core metrics: vf_reduction, epi_preservation
        - Latency state: latent flag, silence_duration
        - Integrity metrics: preservation_integrity, epi_variance
    """
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)

    # Basic SHA metrics
    metrics = {
        "operator": "Silence",
        "glyph": "SHA",
        "vf_reduction": vf_before - vf_after,
        "vf_final": vf_after,
        "epi_preservation": abs(epi_after - epi_before),
        "epi_final": epi_after,
        "is_silent": vf_after < 0.1,  # Configurable threshold
    }

    # Latency state tracking metrics
    metrics["latent"] = G.nodes[node].get("latent", False)
    metrics["silence_duration"] = G.nodes[node].get("silence_duration", 0.0)

    # Preservation integrity: measures EPI variance during silence
    preserved_epi = G.nodes[node].get("preserved_epi")
    if preserved_epi is not None:
        preservation_integrity = abs(epi_after - preserved_epi) / max(
            abs(preserved_epi), 1e-10
        )
        metrics["preservation_integrity"] = preservation_integrity
    else:
        metrics["preservation_integrity"] = 0.0

    # EPI variance during silence (relative to preserved value)
    if preserved_epi is not None:
        epi_variance = abs(epi_after - preserved_epi)
        metrics["epi_variance_during_silence"] = epi_variance
    else:
        metrics["epi_variance_during_silence"] = abs(epi_after - epi_before)

    return metrics


def expansion_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """VAL - Expansion metrics: νf increase, volume exploration.

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
        Expansion-specific metrics including structural dilation
    """
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)

    return {
        "operator": "Expansion",
        "glyph": "VAL",
        "vf_increase": vf_after - vf_before,
        "vf_final": vf_after,
        "delta_epi": epi_after - epi_before,
        "epi_final": epi_after,
        "expansion_factor": vf_after / vf_before if vf_before > 0 else 1.0,
    }


def contraction_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """NUL - Contraction metrics: νf decrease, core concentration.

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
        Contraction-specific metrics including structural compression
    """
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)

    return {
        "operator": "Contraction",
        "glyph": "NUL",
        "vf_decrease": vf_before - vf_after,
        "vf_final": vf_after,
        "delta_epi": epi_after - epi_before,
        "epi_final": epi_after,
        "dnfr_final": dnfr,
        "contraction_factor": vf_after / vf_before if vf_before > 0 else 1.0,
    }


def self_organization_metrics(
    G: TNFRGraph, node: NodeId, epi_before: float, vf_before: float
) -> dict[str, Any]:
    """THOL - Self-organization metrics: nested EPI generation, cascade formation.

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
        Self-organization-specific metrics including cascade indicators
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    d2epi = _get_node_attr(G, node, ALIAS_D2EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)

    # Track nested EPI count if graph maintains it
    nested_epi_count = len(G.graph.get("sub_epi", []))

    return {
        "operator": "Self-organization",
        "glyph": "THOL",
        "delta_epi": epi_after - epi_before,
        "delta_vf": vf_after - vf_before,
        "epi_final": epi_after,
        "vf_final": vf_after,
        "d2epi": d2epi,
        "dnfr_final": dnfr,
        "nested_epi_count": nested_epi_count,
        "cascade_active": abs(d2epi) > 0.1,  # Configurable threshold
    }


def mutation_metrics(
    G: TNFRGraph, node: NodeId, theta_before: float, epi_before: float
) -> dict[str, Any]:
    """ZHIR - Mutation metrics: phase transition, structural regime change.

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

    Returns
    -------
    dict
        Mutation-specific metrics including phase change indicators
    """
    theta_after = _get_node_attr(G, node, ALIAS_THETA)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)

    return {
        "operator": "Mutation",
        "glyph": "ZHIR",
        "theta_shift": abs(theta_after - theta_before),
        "theta_final": theta_after,
        "delta_epi": epi_after - epi_before,
        "epi_final": epi_after,
        "phase_change": abs(theta_after - theta_before) > 0.5,  # Configurable threshold
    }


def transition_metrics(
    G: TNFRGraph,
    node: NodeId,
    dnfr_before: float,
    vf_before: float,
    theta_before: float,
) -> dict[str, Any]:
    """NAV - Transition metrics: regime handoff, ΔNFR rebalancing.

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

    Returns
    -------
    dict
        Transition-specific metrics including handoff success
    """
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    theta_after = _get_node_attr(G, node, ALIAS_THETA)

    return {
        "operator": "Transition",
        "glyph": "NAV",
        "dnfr_change": abs(dnfr_after - dnfr_before),
        "dnfr_final": dnfr_after,
        "vf_change": abs(vf_after - vf_before),
        "vf_final": vf_after,
        "theta_shift": abs(theta_after - theta_before),
        "theta_final": theta_after,
        # Transition complete when ΔNFR magnitude is bounded by νf magnitude
        # indicating structural frequency dominates reorganization dynamics
        "transition_complete": abs(dnfr_after) < abs(vf_after),
    }


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
