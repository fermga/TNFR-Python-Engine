"""Operator metrics: structural operators."""

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




def expansion_metrics(G, node, vf_before: float, epi_before: float) -> dict[str, Any]:
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
    growth_ratio = vf_growth_rate / epi_growth_rate if abs(epi_growth_rate) > 1e-9 else 0.0

    # Coherence preservation
    # Local coherence via extracted helper
    from ..metrics.local_coherence import compute_local_coherence_fallback

    c_local = compute_local_coherence_fallback(G, node)

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
    bifurcation_magnitude = abs(d2epi) / bifurcation_threshold if bifurcation_threshold > 0 else 0.0

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
        dnfr_positive and not bifurcation_risk and coherence_preserved and fractal_preserved
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


def contraction_metrics(G, node, vf_before, epi_before):
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
    density_before = (
        abs(dnfr_before) / max(abs(epi_before), EPSILON) if dnfr_before is not None else 0.0
    )
    density_after = abs(dnfr_after) / max(abs(epi_after), EPSILON)

    # Calculate densification ratio (how much density increased)
    densification_ratio = (
        density_after / density_before if density_before > EPSILON else float("inf")
    )

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


def self_organization_metrics(G, node, epi_before, vf_before):
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
        "network_emergence": (cascade_analysis["is_cascade"] and subepi_coherence > 0.5),
    }


def mutation_metrics(
    G,
    node,
    theta_before,
    epi_before,
    vf_before=None,
    dnfr_before=None,
):
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
    epi_history = G.nodes[node].get("epi_history") or G.nodes[node].get("_epi_history", [])
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
        G.graph.get("BIFURCATION_THRESHOLD_TAU", G.graph.get("ZHIR_BIFURCATION_THRESHOLD", 0.5))
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
    identity_preserved = epi_kind_before == epi_kind_after if epi_kind_before is not None else True

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
                    neighbor_change = abs(neighbor_theta_history[-1] - neighbor_theta_history[-2])
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
    destabilizer_recent = destabilizer_distance is not None and destabilizer_distance <= 3

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
        "theta_regime_change": regime_changed
        or (regime_before != regime_after),  # Backwards compat
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
    G,
    node,
    dnfr_before,
    vf_before,
    theta_before,
    epi_before=None,
):
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


def recursivity_metrics(G, node, epi_before, vf_before):
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



