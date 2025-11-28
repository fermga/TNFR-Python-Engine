"""Operator metrics: network operators."""

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

__all__ = [
    "coupling_metrics",
    "resonance_metrics",
    "silence_metrics",
    # Private helpers exposed for testing backward compatibility
    "_compute_epi_variance",
    "_compute_preservation_integrity",
    "_compute_reactivation_readiness",
    "_estimate_time_to_collapse",
]



def _compute_epi_variance(G, node) -> float:
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


def _compute_reactivation_readiness(G, node) -> float:
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


def _estimate_time_to_collapse(G, node) -> float:
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




def coupling_metrics(
    G,
    node,
    theta_before,
    dnfr_before=None,
    vf_before=None,
    edges_before=None,
    epi_before=None,
):
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
    G,
    node,
    epi_before,
    vf_before=None,
):
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
        "propagation_successful": neighbor_count > 0 and abs(epi_after - neighbor_epi_mean) < 0.5,
        # Canonical TNFR effects
        "vf_amplification": vf_amplification,  # Canonical: νf increases through resonance
        "vf_before": vf_before if vf_before is not None else vf_after,
        "vf_after": vf_after,
        "phase_alignment": phase_alignment,  # Canonical: phase strengthens
        "identity_preserved": identity_preserved,  # Canonical: EPI identity maintained
    }


def _compute_epi_variance(G, node) -> float:
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


def _compute_reactivation_readiness(G, node) -> float:
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


def _estimate_time_to_collapse(G, node) -> float:
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


def silence_metrics(G, node, vf_before, epi_before):
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


