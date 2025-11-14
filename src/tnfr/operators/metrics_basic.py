"""Operator metrics: basic operators."""

from __future__ import annotations

from typing import Any

from .metrics_core import (
    get_node_attr as _get_node_attr,
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
    HAS_EMISSION_TIMESTAMP_ALIAS as _HAS_EMISSION_TIMESTAMP_ALIAS,
    EMISSION_TIMESTAMP_TUPLE as _ALIAS_EMISSION_TIMESTAMP_TUPLE,
)
from ..alias import get_attr_str


def emission_metrics(G, node, epi_before: float, vf_before: float) -> dict[str, Any]:
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

    # Emission timestamp via alias system with guarded fallback
    emission_timestamp = None
    if _HAS_EMISSION_TIMESTAMP_ALIAS and _ALIAS_EMISSION_TIMESTAMP_TUPLE:
        try:
            emission_timestamp = get_attr_str(
                G.nodes[node], _ALIAS_EMISSION_TIMESTAMP_TUPLE, default=None
            )
        except Exception:
            pass
    if emission_timestamp is None:
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


def reception_metrics(G, node, epi_before: float) -> dict[str, Any]:
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


def coherence_metrics(G, node, dnfr_before: float) -> dict[str, Any]:
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
    # Import minimal dependencies (avoid unavailable symbols)
    from ..metrics.phase_coherence import compute_phase_alignment
    from ..metrics.common import compute_coherence as _compute_global_coherence
    from ..metrics.local_coherence import compute_local_coherence_fallback

    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    vf = _get_node_attr(G, node, ALIAS_VF)

    # Compute reduction metrics
    dnfr_reduction = dnfr_before - dnfr_after
    dnfr_reduction_pct = (dnfr_reduction / dnfr_before * 100.0) if dnfr_before > 0 else 0.0

    # Compute global coherence using shared common implementation
    C_global = _compute_global_coherence(G)

    # Local coherence via extracted helper
    C_local = compute_local_coherence_fallback(G, node)

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


def dissonance_metrics(G, node, dnfr_before, theta_before):
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
        "network_impact_radius": (impacted_neighbors / len(neighbors) if neighbors else 0.0),
        # Recovery guidance
        "recovery_estimate_IL": recovery_estimate,
        "dissonance_level": abs(dnfr_after),
        "critical_dissonance": abs(dnfr_after) > 0.8,
        # Network propagation
        **propagation_data,
        **field_data,
    }


