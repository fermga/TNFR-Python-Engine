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
    """IL - Coherence metrics: ΔC(t), stability gain, ΔNFR reduction.

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
        Coherence-specific metrics including stability improvement
    """
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    vf = _get_node_attr(G, node, ALIAS_VF)

    return {
        "operator": "Coherence",
        "glyph": "IL",
        "dnfr_reduction": dnfr_before - dnfr_after,
        "dnfr_final": dnfr_after,
        "stability_gain": abs(dnfr_before) - abs(dnfr_after),
        "epi_final": epi,
        "vf_final": vf,
        "is_stabilized": abs(dnfr_after) < 0.1,  # Configurable threshold
    }


def dissonance_metrics(
    G: TNFRGraph, node: NodeId, dnfr_before: float, theta_before: float
) -> dict[str, Any]:
    """OZ - Dissonance metrics: ΔNFR increase, bifurcation risk, phase shift.

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
        Dissonance-specific metrics including bifurcation indicators
    """
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    theta_after = _get_node_attr(G, node, ALIAS_THETA)
    d2epi = _get_node_attr(G, node, ALIAS_D2EPI)

    # Bifurcation threshold - configurable
    bifurcation_threshold = float(G.graph.get("OZ_BIFURCATION_THRESHOLD", 0.5))

    return {
        "operator": "Dissonance",
        "glyph": "OZ",
        "dnfr_increase": dnfr_after - dnfr_before,
        "dnfr_final": dnfr_after,
        "theta_shift": abs(theta_after - theta_before),
        "d2epi": d2epi,
        "bifurcation_risk": abs(d2epi) > bifurcation_threshold,
        "dissonance_level": abs(dnfr_after),
    }


def coupling_metrics(G: TNFRGraph, node: NodeId, theta_before: float) -> dict[str, Any]:
    """UM - Coupling metrics: phase alignment, link formation, synchrony.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to collect metrics from
    theta_before : float
        Phase value before operator application

    Returns
    -------
    dict
        Coupling-specific metrics including phase synchronization
    """
    import math

    theta_after = _get_node_attr(G, node, ALIAS_THETA)
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

    return {
        "operator": "Coupling",
        "glyph": "UM",
        "theta_shift": abs(theta_after - theta_before),
        "theta_final": theta_after,
        "neighbor_count": neighbor_count,
        "mean_neighbor_phase": mean_neighbor_phase,
        "phase_alignment": max(0.0, phase_alignment),
    }


def resonance_metrics(G: TNFRGraph, node: NodeId, epi_before: float) -> dict[str, Any]:
    """RA - Resonance metrics: EPI propagation, affected neighbors, resonance strength.

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
        Resonance-specific metrics including propagation effectiveness
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Calculate resonance strength based on neighbor coupling
    if neighbor_count > 0:
        neighbor_epi_sum = sum(_get_node_attr(G, n, ALIAS_EPI) for n in neighbors)
        neighbor_epi_mean = neighbor_epi_sum / neighbor_count
        resonance_strength = abs(epi_after - epi_before) * neighbor_count
    else:
        neighbor_epi_mean = 0.0
        resonance_strength = 0.0

    return {
        "operator": "Resonance",
        "glyph": "RA",
        "delta_epi": epi_after - epi_before,
        "epi_final": epi_after,
        "neighbor_count": neighbor_count,
        "neighbor_epi_mean": neighbor_epi_mean,
        "resonance_strength": resonance_strength,
        "propagation_successful": neighbor_count > 0
        and abs(epi_after - neighbor_epi_mean) < 0.5,
    }


def silence_metrics(
    G: TNFRGraph, node: NodeId, vf_before: float, epi_before: float
) -> dict[str, Any]:
    """SHA - Silence metrics: νf reduction, EPI preservation.

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
        Silence-specific metrics including frequency reduction
    """
    vf_after = _get_node_attr(G, node, ALIAS_VF)
    epi_after = _get_node_attr(G, node, ALIAS_EPI)

    return {
        "operator": "Silence",
        "glyph": "SHA",
        "vf_reduction": vf_before - vf_after,
        "vf_final": vf_after,
        "epi_preservation": abs(epi_after - epi_before),
        "epi_final": epi_after,
        "is_silent": vf_after < 0.1,  # Configurable threshold
    }


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
