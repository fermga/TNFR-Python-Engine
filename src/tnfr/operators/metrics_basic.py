"""Operator metrics: basic operators."""

from __future__ import annotations

import math
from typing import Any

from ..alias import get_attr_str
from ..utils import angle_diff
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .metrics_core import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from .metrics_core import EMISSION_TIMESTAMP_TUPLE as _ALIAS_EMISSION_TIMESTAMP_TUPLE
from .metrics_core import HAS_EMISSION_TIMESTAMP_ALIAS as _HAS_EMISSION_TIMESTAMP_ALIAS
from .metrics_core import get_node_attr as _get_node_attr
from ._diagnostic_scores import (
    mean_unit_score,
    nonnegative_magnitude,
    sum_nonnegative_magnitudes,
)
from ._reception_kernel import (
    RECEPTION_PRE_STATE_BOUNDARY,
    RECEPTION_PRESSURE_OBSERVATION_BOUNDARY,
    ReceptionReadSnapshot,
    capture_reception_read_snapshot,
)

_COARSE_PRESSURE_MAGNITUDE_THRESHOLD = 0.1


def _normalize_stored_reception_sources(
    value: Any,
) -> tuple[tuple[Any, float, float], ...] | None:
    """Normalize valid legacy EN metadata without trusting opaque payloads."""

    if type(value) not in (list, tuple):
        return None
    normalized: list[tuple[Any, float, float]] = []
    for source in value:
        if type(source) not in (list, tuple) or len(source) != 3:
            return None
        compatibility = source[1]
        activity = source[2]
        if (
            type(compatibility) is not float
            or not math.isfinite(compatibility)
            or not 0.0 <= compatibility <= 1.0
            or type(activity) is not float
            or not math.isfinite(activity)
            or activity < 0.0
        ):
            return None
        normalized.append((source[0], compatibility, activity))
    return tuple(normalized)


def emission_metrics(G, node, epi_before: float, vf_before: float) -> dict[str, Any]:
    """AL - Emission metrics with structural fidelity indicators.

    Collects emission-specific metrics for AL's EPI-only contract:
    - EPI: receives a positive source proposal and cannot decrease
    - vf: remains at its pre-existing basal value
    - DELTA_NFR and theta: observed as context, not written by AL

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
          - emission_quality: "valid" when EPI did not decrease
          - emission_effective: True when the bounded EPI source increased EPI
          - activation_from_latency: True if node was latent (EPI < 0.35)
          - form_emergence_magnitude: signed EPI source increment
          - frequency_preserved: True when AL left νf unchanged
          - capacity_active: True when the retained νf is positive
          - frequency_activation: legacy νf-change flag, expected False for AL
          - reorganization_positive: observed ΔNFR sign, not an AL effect
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

    # AL's source proposal can saturate at the configured EPI boundary.
    # Non-decrease is therefore the runtime contract; νf is a read-only
    # precondition rather than a success signal.
    tolerance = 1e-12
    epi_contract_satisfied = delta_epi >= -tolerance
    emission_quality = "valid" if epi_contract_satisfied else "invalid"
    emission_effective = delta_epi > tolerance

    latency_threshold = 0.35  # ≈ 0.357 (latency)
    activation_from_latency = epi_before < latency_threshold
    frequency_preserved = abs(delta_vf) <= tolerance
    capacity_active = vf_after > 0.0
    # Backward-compatible observation: canonical AL leaves this False.
    frequency_activation = delta_vf > tolerance
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
        "emission_effective": emission_effective,
        "activation_from_latency": activation_from_latency,
        "form_emergence_magnitude": delta_epi,
        "frequency_preserved": frequency_preserved,
        "capacity_active": capacity_active,
        "frequency_activation": frequency_activation,
        "reorganization_positive": reorganization_positive,
        # Traceability (NEW)
        "emission_timestamp": emission_timestamp,
        "irreversibility_marker": irreversibility_marker,
    }


def reception_metrics(G, node, epi_before: float) -> dict[str, Any]:
    """Collect EN metrics from values observable at this call boundary."""

    return _reception_metrics_impl(G, node, epi_before, read_snapshot=None)


def _reception_metrics_from_snapshot(
    G,
    node,
    epi_before: float,
    *,
    read_snapshot: ReceptionReadSnapshot,
) -> dict[str, Any]:
    """Consume executor-owned pre-EN evidence through an internal boundary."""

    if type(read_snapshot) is not ReceptionReadSnapshot:
        raise TypeError("read_snapshot must be a ReceptionReadSnapshot")
    if (
        not read_snapshot._proof_fields_are_intact()
        or read_snapshot._metric_consumer_graph_owner is not G
        or read_snapshot._metric_consumer_graph_identity != id(G)
    ):
        raise ValueError("Reception metrics snapshot belongs to another graph")
    return _reception_metrics_impl(
        G,
        node,
        epi_before,
        read_snapshot=read_snapshot,
    )


def _reception_metrics_impl(
    G,
    node,
    epi_before: float,
    *,
    read_snapshot: ReceptionReadSnapshot | None,
) -> dict[str, Any]:
    """EN EPI-intake, source-activity and phase-compatibility diagnostics.

    These operational readouts include a signed activity ratio and a bounded
    phase score. They do not read dEPI and DeltaNFR as the shared C(t) kernel
    requires, so none certifies canonical structural coherence.

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
          - total_source_emission_activity: Unbounded activity sum
          - epi_delta_per_source_activity: Signed, unbounded intake ratio
          - most_compatible_source: Most phase-compatible source node
          - mean_phase_compatibility_score: Bounded score mean
          - legacy compatibility aliases for the renamed fields
          - pressure_magnitude_below_effectiveness_threshold: Whether the
            stored post-EN |ΔNFR| is below the coarse threshold
          - stabilization_effective: compatibility alias for that predicate
          - explicit read and pressure-observation boundaries
    """
    epi_after = _get_node_attr(G, node, ALIAS_EPI)
    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    storage = G.nodes[node]
    stored_source_metadata_present = "_reception_sources" in storage
    normalized_stored_sources = (
        _normalize_stored_reception_sources(storage["_reception_sources"])
        if stored_source_metadata_present
        else None
    )
    stored_source_metadata_valid = (
        stored_source_metadata_present
        and normalized_stored_sources is not None
    )

    if read_snapshot is None:
        # Standalone metrics observe the live call boundary, but their EN input
        # domain still comes from the canonical snapshot policy: incoming arcs
        # on directed support, target substitution for missing neighbour EPI,
        # and no effective neighbour set when none has explicit EPI.
        live_read = capture_reception_read_snapshot(
            G,
            node,
            track_sources=False,
        )
        neighbors = live_read.neighbors
        neighbor_count = len(neighbors)
        neighbor_epi_mean = live_read.neighbor_epi_mean
        sources = normalized_stored_sources or ()
        read_boundary = "metrics_call_live_state"
        pressure_boundary = "metrics_call_live_state"
        source_tracking_enabled = None
        source_max_distance = None
        stored_source_metadata_boundary = "metrics_call_live_state"
    else:
        if type(read_snapshot) is not ReceptionReadSnapshot:
            raise TypeError("read_snapshot must be a ReceptionReadSnapshot")
        if (
            not read_snapshot._proof_fields_are_intact()
            or read_snapshot._metric_consumer_graph_owner is not G
            or read_snapshot._metric_consumer_graph_identity != id(G)
        ):
            raise ValueError("Reception metrics snapshot belongs to another graph")
        if not proof_stamps_are_identical(
            structural_proof_signature(read_snapshot.node),
            structural_proof_signature(node),
        ):
            raise ValueError("Reception metrics snapshot target changed")
        if epi_before != read_snapshot.target_epi:
            raise ValueError("Reception metrics pre-state contradicts its snapshot")
        neighbors = read_snapshot.neighbors
        neighbor_count = len(neighbors)
        neighbor_epi_mean = read_snapshot.neighbor_epi_mean
        sources = read_snapshot.reception_sources or ()
        read_boundary = RECEPTION_PRE_STATE_BOUNDARY
        pressure_boundary = RECEPTION_PRESSURE_OBSERVATION_BOUNDARY
        source_tracking_enabled = read_snapshot.source_tracking_enabled
        source_max_distance = read_snapshot.source_max_distance
        stored_source_metadata_boundary = (
            RECEPTION_PRESSURE_OBSERVATION_BOUNDARY
        )

    # Compute the signed EPI change. This is form change, not C(t).
    delta_epi = epi_after - epi_before

    # EN-specific: Source tracking and integration efficiency
    num_sources = len(sources)

    source_activities = tuple(
        nonnegative_magnitude(
            activity, label="EN source emission activity"
        )
        for _, _, activity in sources
    )
    total_source_emission_activity = sum_nonnegative_magnitudes(
        source_activities,
        label="EN total source emission activity",
    )
    epi_delta_per_source_activity = (
        delta_epi / total_source_emission_activity
        if total_source_emission_activity > 0.0
        else 0.0
    )

    # Source detection is sorted, while standalone legacy metadata need not be.
    # ``max`` retains the first record when compatibility scores tie.
    most_compatible_source = (
        max(sources, key=lambda source: source[1])[0]
        if sources
        else None
    )

    mean_phase_compatibility_score = (
        mean_unit_score(
            (score for _, score, _ in sources),
            label="EN mean phase compatibility",
        )
        if num_sources > 0
        else 0.0
    )

    pressure_magnitude_below_threshold = (
        abs(dnfr_after) < _COARSE_PRESSURE_MAGNITUDE_THRESHOLD
    )

    return {
        "operator": "Reception",
        "glyph": "EN",
        # Core metrics
        "delta_epi": delta_epi,
        "epi_final": epi_after,
        "dnfr_after": dnfr_after,
        "observed_dnfr": dnfr_after,
        "reception_read_boundary": read_boundary,
        "neighbor_state_observation_boundary": read_boundary,
        "source_state_observation_boundary": read_boundary,
        "stored_source_metadata_observation_boundary": (
            stored_source_metadata_boundary
        ),
        "dnfr_observation_boundary": pressure_boundary,
        # Legacy metrics (backward compatibility)
        "neighbor_count": neighbor_count,
        "neighbor_epi_mean": neighbor_epi_mean,
        "integration_strength": abs(delta_epi),
        # EN-specific (NEW)
        "num_sources": num_sources,
        "source_tracking_enabled": source_tracking_enabled,
        "source_max_distance": source_max_distance,
        "sources_observed": (
            source_tracking_enabled
            if source_tracking_enabled is not None
            else None
        ),
        "source_absence_observed": (
            not sources if source_tracking_enabled else None
        ),
        "stored_source_metadata_present": stored_source_metadata_present,
        "stored_source_metadata_valid": stored_source_metadata_valid,
        "total_source_emission_activity": total_source_emission_activity,
        "epi_delta_per_source_activity": epi_delta_per_source_activity,
        "most_compatible_source": most_compatible_source,
        "mean_phase_compatibility_score": mean_phase_compatibility_score,
        "canonical_coherence_certified": False,
        # Public compatibility aliases. None denotes structural C(t).
        "integration_efficiency": epi_delta_per_source_activity,
        "phase_compatibility_avg": mean_phase_compatibility_score,
        "coherence_received": delta_epi,
        "pressure_magnitude_below_effectiveness_threshold": (
            pressure_magnitude_below_threshold
        ),
        "pressure_effectiveness_threshold": (
            _COARSE_PRESSURE_MAGNITUDE_THRESHOLD
        ),
        "stabilization_effective": pressure_magnitude_below_threshold,
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
        - dnfr_reduction: Reduction in |ΔNFR|
        - dnfr_reduction_pct: Percentage reduction relative to |before|
        - stability_gain: Improvement in stability (reduction of |ΔNFR|)
        - is_stabilized: Coarse operator-effectiveness flag (|ΔNFR| < 0.1) --
          NOT the structural-equilibrium fixed point (|ΔNFR| <= 1e-3; see
          metrics.common.is_structural_equilibrium)
        - C_global: Global network coherence (current)
        - C_local: Local neighborhood coherence (current)
        - phase_alignment: Local phase alignment quality (Kuramoto order parameter)
        - phase_coherence_quality: Alias for phase_alignment (for clarity)
        - stabilization_quality: Combined metric (C_local * (1.0 - dnfr_after))
        - epi_final, vf_final: Final structural state
    """
    # Import minimal dependencies (avoid unavailable symbols)
    from ..metrics.common import compute_coherence as _compute_global_coherence
    from ..metrics.local_coherence import compute_local_coherence_fallback
    from ..metrics.phase_coherence import compute_phase_alignment

    dnfr_after = _get_node_attr(G, node, ALIAS_DNFR)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    vf = _get_node_attr(G, node, ALIAS_VF)

    # IL contracts pressure magnitude while preserving its sign.
    magnitude_before = abs(dnfr_before)
    magnitude_after = abs(dnfr_after)
    dnfr_reduction = magnitude_before - magnitude_after
    dnfr_reduction_pct = (
        dnfr_reduction / magnitude_before * 100.0
        if magnitude_before > 0.0
        else 0.0
    )

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
        "stability_gain": dnfr_reduction,
        "C_global": C_global,
        "C_local": C_local,
        "phase_alignment": phase_alignment,
        "phase_coherence_quality": phase_alignment,  # Alias for clarity
        "stabilization_quality": C_local
        * max(0.0, 1.0 - magnitude_after),
        "epi_final": epi,
        "vf_final": vf,
        # Coarse operator-effectiveness flag, NOT structural equilibrium (the
        # canonical fixed point is |ΔNFR| <= 1e-3; see is_structural_equilibrium)
        "is_stabilized": (
            abs(dnfr_after) < _COARSE_PRESSURE_MAGNITUDE_THRESHOLD
        ),
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
        - viable_paths: list of viable operator glyph values
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
    # set bifurcation_ready flag if score exceeds threshold
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
        "theta_shift": abs(angle_diff(theta_after, theta_before)),
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
