"""Strict precondition validation for RA (Resonance) operator.

This module implements canonical precondition validation for the Resonance (RA)
structural operator according to TNFR theory. RA requires specific structural
conditions to maintain TNFR operational fidelity:

1. **Coherent source EPI**: Node must have sufficient structural form for propagation
2. **Network connectivity**: Edges must exist for resonance to propagate through
3. **Phase compatibility**: Node must be synchronized with neighbors (coupling)
4. **Controlled dissonance**: ΔNFR must not be excessive (stable resonance)
5. **Sufficient νf**: Structural frequency must support propagation dynamics

These validations protect structural integrity by ensuring RA is only applied to
nodes in the appropriate state for coherence propagation through the network.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_resonance_strict", "diagnose_resonance_readiness"]


def validate_resonance_strict(
    G: TNFRGraph,
    node: Any,
    *,
    min_epi: float | None = None,
    require_coupling: bool = True,
    max_dissonance: float | None = None,
    warn_phase_misalignment: bool = True,
) -> None:
    """Validate strict canonical preconditions for RA (Resonance) operator.

    According to TNFR theory, Resonance (RA - Resonancia) requires:

    1. **Coherent source**: EPI >= threshold (sufficient structure to propagate)
    2. **Network connectivity**: degree > 0 (edges for propagation)
    3. **Phase compatibility**: alignment with neighbors (synchronization)
    4. **Controlled dissonance**: |ΔNFR| < threshold (stable for resonance)
    5. **Sufficient νf**: νf > threshold (capacity for propagation dynamics)

    Canonical sequences that satisfy preconditions:
    - **UM → RA**: Coupling establishes connections, then resonance propagates
    - **AL → RA**: Emission activates source, then resonance broadcasts
    - **IL → RA**: Coherence stabilizes, then propagates stable form

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation
    min_epi : float, optional
        Minimum EPI magnitude for resonance source
        Default: Uses G.graph["RA_MIN_SOURCE_EPI"] or 0.1
    require_coupling : bool, default True
        If True, validates that node has edges (connectivity)
    max_dissonance : float, optional
        Maximum allowed |ΔNFR| for resonance
        Default: Uses G.graph["RA_MAX_DISSONANCE"] or 0.5
    warn_phase_misalignment : bool, default True
        If True, warns when phase difference with neighbors is high

    Raises
    ------
    ValueError
        If EPI < min_epi (insufficient structure to propagate)
        If require_coupling=True and node has no edges
        If |ΔNFR| > max_dissonance (too unstable for resonance)
        If νf < threshold (insufficient structural frequency)

    Warnings
    --------
    UserWarning
        If phase misalignment with neighbors exceeds threshold (suboptimal resonance)
        If node is isolated but require_coupling=False

    Notes
    -----
    Thresholds are configurable via graph metadata:
    - ``RA_MIN_SOURCE_EPI``: Minimum EPI for source (default: 0.1)
    - ``RA_MAX_DISSONANCE``: Maximum |ΔNFR| (default: 0.5)
    - ``RA_MAX_PHASE_DIFF``: Maximum phase difference in radians (default: 1.0)
    - ``RA_MIN_VF``: Minimum structural frequency (default: 0.01)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.resonance import validate_resonance_strict
    >>>
    >>> # Valid node for resonance
    >>> G, node = create_nfr("source", epi=0.8, vf=0.9)
    >>> neighbor = "neighbor"
    >>> G.add_node(neighbor, epi=0.5, vf=0.8, theta=0.1, dnfr=0.05, epi_kind="seed")
    >>> G.add_edge(node, neighbor)
    >>> G.nodes[node]["dnfr"] = 0.1
    >>> validate_resonance_strict(G, node)  # OK

    >>> # Invalid: EPI too low
    >>> G2, node2 = create_nfr("weak_source", epi=0.05, vf=0.9)
    >>> neighbor2 = "neighbor2"
    >>> G2.add_node(neighbor2, epi=0.5, vf=0.8, theta=0.1, dnfr=0.05, epi_kind="seed")
    >>> G2.add_edge(node2, neighbor2)
    >>> validate_resonance_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: RA requires coherent source with EPI >= 0.1 (current: 0.050). Apply IL or THOL first.

    >>> # Invalid: No connectivity
    >>> G3, node3 = create_nfr("isolated", epi=0.8, vf=0.9)
    >>> validate_resonance_strict(G3, node3)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: RA requires network connectivity (node has no edges). Apply UM (Coupling) first.

    See Also
    --------
    tnfr.operators.definitions.Resonance : Resonance operator implementation
    tnfr.operators.definitions.Coupling : Establishes connectivity for RA
    diagnose_resonance_readiness : Diagnostic function for RA readiness
    """
    from ...alias import get_attr
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
    from ...utils.numeric import angle_diff

    # Get configuration with defensive fallbacks
    if min_epi is None:
        min_epi = float(G.graph.get("RA_MIN_SOURCE_EPI", 0.1))
    if max_dissonance is None:
        max_dissonance = float(G.graph.get("RA_MAX_DISSONANCE", 0.5))
    min_vf = float(G.graph.get("RA_MIN_VF", 0.01))
    max_phase_diff = float(G.graph.get("RA_MAX_PHASE_DIFF", 1.0))  # ~60 degrees

    # 1. Validate coherent source EPI
    epi = abs(float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)))
    if epi < min_epi:
        raise ValueError(
            f"RA requires coherent source with EPI >= {min_epi:.1f} "
            f"(current: {epi:.3f}). Apply IL or THOL first."
        )

    # 2. Validate network connectivity
    neighbors = list(G.neighbors(node))
    if require_coupling:
        if not neighbors:
            raise ValueError(
                "RA requires network connectivity (node has no edges). "
                "Apply UM (Coupling) first to establish resonant links."
            )
    elif not neighbors:
        # Node is isolated but require_coupling=False - issue warning
        warnings.warn(
            f"Node {node} is isolated - RA will have no propagation effect. "
            "Consider applying UM (Coupling) first.",
            UserWarning,
            stacklevel=3,
        )

    # 3. Validate sufficient structural frequency
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    if vf < min_vf:
        raise ValueError(
            f"RA requires sufficient structural frequency νf >= {min_vf:.2f} "
            f"(current: {vf:.3f}). Apply AL (Emission) or VAL (Expansion) first."
        )

    # 4. Validate controlled dissonance
    dnfr = abs(float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)))
    if dnfr > max_dissonance:
        raise ValueError(
            f"RA requires controlled dissonance with |ΔNFR| <= {max_dissonance:.1f} "
            f"(current: {dnfr:.3f}). Apply IL (Coherence) first to stabilize."
        )

    # 5. Validate phase compatibility (warning only, neighbors exist)
    if warn_phase_misalignment and neighbors:
        try:
            from ...metrics.trig import neighbor_phase_mean

            theta_node = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
            theta_neighbors = neighbor_phase_mean(G, node)
            phase_diff = abs(angle_diff(theta_neighbors, theta_node))

            if phase_diff > max_phase_diff:
                warnings.warn(
                    f"RA phase misalignment: Δφ = {phase_diff:.2f} > {max_phase_diff:.2f}. "
                    "Consider applying UM (Coupling) first for better resonance.",
                    UserWarning,
                    stacklevel=3,
                )
        except Exception:
            # Phase validation is optional, don't fail if unavailable
            pass


def diagnose_resonance_readiness(G: TNFRGraph, node: Any) -> dict[str, Any]:
    """Diagnose node readiness for RA (Resonance) operator.

    Provides comprehensive diagnostic report with readiness status and
    actionable recommendations for RA operator application.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : Any
        Node to diagnose

    Returns
    -------
    dict
        Diagnostic report with:
        - ``ready``: bool - overall readiness status
        - ``checks``: dict - individual check results (passed/failed/warning)
        - ``values``: dict - current node state values
        - ``recommendations``: list - actionable steps to achieve readiness
        - ``canonical_sequences``: list - suggested operator sequences

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.resonance import diagnose_resonance_readiness
    >>>
    >>> # Diagnose weak source
    >>> G, node = create_nfr("weak", epi=0.05, vf=0.9)
    >>> diag = diagnose_resonance_readiness(G, node)
    >>> diag["ready"]
    False
    >>> "coherent_source" in diag["checks"]
    True
    >>> diag["checks"]["coherent_source"]
    'failed'
    >>> "Apply IL (Coherence) or THOL (Self-organization)" in diag["recommendations"][0]
    True

    See Also
    --------
    validate_resonance_strict : Strict precondition validator
    """
    from ...alias import get_attr
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
    from ...utils.numeric import angle_diff

    # Get thresholds
    min_epi = float(G.graph.get("RA_MIN_SOURCE_EPI", 0.1))
    max_dissonance = float(G.graph.get("RA_MAX_DISSONANCE", 0.5))
    min_vf = float(G.graph.get("RA_MIN_VF", 0.01))
    max_phase_diff = float(G.graph.get("RA_MAX_PHASE_DIFF", 1.0))

    # Get current state
    epi = abs(float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    dnfr = abs(float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)))
    theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
    neighbors = list(G.neighbors(node))
    neighbor_count = len(neighbors)

    # Initialize checks
    checks = {}
    recommendations = []

    # Check 1: Coherent source
    if epi >= min_epi:
        checks["coherent_source"] = "passed"
    else:
        checks["coherent_source"] = "failed"
        recommendations.append(
            f"Apply IL (Coherence) or THOL (Self-organization) to increase EPI "
            f"from {epi:.3f} to >= {min_epi:.1f}"
        )

    # Check 2: Network connectivity
    if neighbor_count > 0:
        checks["network_connectivity"] = "passed"
    else:
        checks["network_connectivity"] = "failed"
        recommendations.append("Apply UM (Coupling) to establish network connections before RA")

    # Check 3: Structural frequency
    if vf >= min_vf:
        checks["structural_frequency"] = "passed"
    else:
        checks["structural_frequency"] = "failed"
        recommendations.append(
            f"Apply AL (Emission) or VAL (Expansion) to increase νf "
            f"from {vf:.3f} to >= {min_vf:.2f}"
        )

    # Check 4: Controlled dissonance
    if dnfr <= max_dissonance:
        checks["controlled_dissonance"] = "passed"
    else:
        checks["controlled_dissonance"] = "failed"
        recommendations.append(
            f"Apply IL (Coherence) to reduce |ΔNFR| from {dnfr:.3f} " f"to <= {max_dissonance:.1f}"
        )

    # Check 5: Phase alignment (warning only)
    phase_diff = None
    if neighbor_count > 0:
        try:
            from ...metrics.trig import neighbor_phase_mean

            theta_neighbors = neighbor_phase_mean(G, node)
            phase_diff = abs(angle_diff(theta_neighbors, theta))

            if phase_diff <= max_phase_diff:
                checks["phase_alignment"] = "passed"
            else:
                checks["phase_alignment"] = "warning"
                recommendations.append(
                    f"Consider applying UM (Coupling) to improve phase alignment "
                    f"(current: Δφ = {phase_diff:.2f}, optimal: <= {max_phase_diff:.2f})"
                )
        except Exception:
            checks["phase_alignment"] = "unavailable"
    else:
        checks["phase_alignment"] = "n/a"

    # Determine overall readiness
    critical_checks = [
        "coherent_source",
        "network_connectivity",
        "structural_frequency",
        "controlled_dissonance",
    ]
    ready = all(checks.get(check) == "passed" for check in critical_checks)

    # Canonical sequences
    canonical_sequences = [
        "UM → RA (Coupling then Resonance)",
        "AL → RA (Emission then Resonance)",
        "IL → RA (Coherence then Resonance)",
        "AL → EN → IL → UM → RA (Full activation sequence)",
    ]

    return {
        "ready": ready,
        "checks": checks,
        "values": {
            "epi": epi,
            "vf": vf,
            "dnfr": dnfr,
            "theta": theta,
            "neighbor_count": neighbor_count,
            "phase_diff": phase_diff,
        },
        "recommendations": recommendations,
        "canonical_sequences": canonical_sequences,
        "thresholds": {
            "min_epi": min_epi,
            "max_dissonance": max_dissonance,
            "min_vf": min_vf,
            "max_phase_diff": max_phase_diff,
        },
    }
