"""Configured precondition validation for RA (Resonance) operator.

The optional strict policy and its readiness diagnostic share admitted state
and threshold inputs. These configurable bounds are not laws derived from the
nodal equation. The separate U3 gate remains mandatory during execution:

1. **Coherent source EPI**: Node must have sufficient structural form for propagation
2. **Network connectivity**: Edges must exist for resonance to propagate through
3. **Mean-phase observation**: Misalignment warns; a zero resultant is unavailable
4. **Controlled dissonance**: ΔNFR must not be excessive (stable resonance)
5. **Sufficient νf**: Structural frequency must support propagation dynamics

These configured checks do not establish synchronization, propagation or
persistence. The readiness report is not execution authorization.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ...alias import get_attr
from ...config.thresholds import DNFR_RA_MAX, EPI_RA_MIN, VF_RA_MIN
from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ...constants.canonical import DELTA_PHI_MAX
from ...errors import TNFRValueError
from ...utils.numeric import angle_diff
from .._argument_validation import finite_node_real, finite_real, strict_bool
from .._epi_domain import require_real_scalar_epi

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_resonance_strict", "diagnose_resonance_readiness"]


def _resonance_thresholds(
    G: TNFRGraph,
    *,
    min_epi: Any = None,
    max_dissonance: Any = None,
) -> tuple[float, ...]:
    """Resolve nonnegative policy magnitudes before any comparison or coercion."""
    inputs = (
        (
            "RA_MIN_SOURCE_EPI",
            (
                G.graph.get("RA_MIN_SOURCE_EPI", EPI_RA_MIN)
                if min_epi is None
                else min_epi
            ),
        ),
        (
            "RA_MAX_DISSONANCE",
            (
                G.graph.get("RA_MAX_DISSONANCE", DNFR_RA_MAX)
                if max_dissonance is None
                else max_dissonance
            ),
        ),
        ("RA_MIN_VF", G.graph.get("RA_MIN_VF", VF_RA_MIN)),
        ("RA_MAX_PHASE_DIFF", G.graph.get("RA_MAX_PHASE_DIFF", DELTA_PHI_MAX)),
    )
    return tuple(
        finite_real(value, operator="Resonance", label=key, lower=0.0)
        for key, value in inputs
    )


def _resonance_state(G: TNFRGraph, node: Any) -> tuple[float, float, float]:
    """Read finite source magnitude, nonnegative capacity and pressure magnitude."""
    data = G.nodes[node]
    epi = require_real_scalar_epi(
        get_attr(data, ALIAS_EPI, 0.0, strict=True, conv=lambda value: value),
        operator="Resonance",
        label="EPI state",
    )
    vf = finite_node_real(
        data, ALIAS_VF, 0.0, operator="Resonance", label="nu_f state", lower=0.0
    )
    dnfr = finite_node_real(
        data, ALIAS_DNFR, 0.0, operator="Resonance", label="DeltaNFR state"
    )
    return abs(epi), vf, abs(dnfr)


def _phase_alignment_difference(
    G: TNFRGraph, neighbors: list[Any], theta: float
) -> float | None:
    """Observe admitted neighbor phases; joint-zero components have no direction."""
    from ...mathematics.phasor_resultant import reduce_phasor_components
    from ...metrics.trig_cache import compute_theta_trig

    trig = compute_theta_trig((neighbor, G.nodes[neighbor]) for neighbor in neighbors)
    resultant = reduce_phasor_components(
        (trig.cos[neighbor], trig.sin[neighbor]) for neighbor in neighbors
    )
    if resultant.angle is None:
        return None
    return abs(angle_diff(resultant.angle, theta))


def _capacity_advice(vf: float, minimum: float) -> str:
    """Avoid promising that a multiplicative operator activates zero capacity."""
    required = f"Supply admitted capacity νf >= {minimum:.2f} (current: {vf:.3f})."
    if vf > 0.0:
        return required + " VAL (Expansion) can scale positive capacity if admitted."
    return required + " Multiplicative scaling preserves zero capacity."


def validate_resonance_strict(
    G: TNFRGraph,
    node: Any,
    *,
    min_epi: float | None = None,
    require_coupling: bool = True,
    max_dissonance: float | None = None,
    warn_phase_misalignment: bool = True,
) -> None:
    """Validate the configured strict policy for RA (Resonance).

    According to TNFR theory, Resonance (RA - Resonancia) requires:

    1. **Coherent source**: EPI >= threshold (sufficient structure to propagate)
    2. **Network connectivity**: degree > 0 (edges for propagation)
    3. **Phase compatibility**: alignment with neighbors (synchronization)
    4. **Controlled dissonance**: |ΔNFR| < threshold (stable for resonance)
    5. **Sufficient νf**: νf > threshold (capacity for propagation dynamics)

    Example compositions; every step still requires live admission:
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
    OperatorPreconditionError
        If an active threshold or scalar input is invalid. Thresholds must be
        finite nonnegative real magnitudes; zero is admitted.
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
    - ``RA_MAX_PHASE_DIFF``: Mean-based warning threshold in radians
      (default: ``DELTA_PHI_MAX``, pi/2); distinct from the U3 hard gate
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
    ValueError: RA requires coherent source with EPI >= 0.1 (current: 0.050). Supply a source meeting the configured EPI magnitude bound.

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
    min_epi, max_dissonance, min_vf, max_phase_diff = _resonance_thresholds(
        G, min_epi=min_epi, max_dissonance=max_dissonance
    )
    require_coupling = strict_bool(
        require_coupling, operator="Resonance", label="require_coupling"
    )
    warn_phase_misalignment = strict_bool(
        warn_phase_misalignment,
        operator="Resonance",
        label="warn_phase_misalignment",
    )
    epi, vf, dnfr = _resonance_state(G, node)

    # 1. Validate coherent source EPI
    if epi < min_epi:
        raise TNFRValueError(
            f"RA requires coherent source with EPI >= {min_epi:.1f} "
            f"(current: {epi:.3f}). "
            "Supply a source meeting the configured EPI magnitude bound.",
            context={"epi": epi, "min_epi": min_epi},
            suggestion="Supply a source meeting the configured EPI magnitude bound.",
        )

    # 2. Validate network connectivity
    neighbors = list(G.neighbors(node))
    if require_coupling:
        if not neighbors:
            raise TNFRValueError(
                "RA requires network connectivity (node has no edges). "
                "Apply UM (Coupling) first to establish resonant links.",
                suggestion="Apply UM (Coupling) first to establish resonant links.",
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
    if vf < min_vf:
        advice = _capacity_advice(vf, min_vf)
        raise TNFRValueError(
            f"RA requires sufficient structural frequency. {advice}",
            context={"vf": vf, "min_vf": min_vf},
            suggestion=advice,
        )

    # 4. Validate controlled dissonance
    if dnfr > max_dissonance:
        raise TNFRValueError(
            f"RA requires controlled dissonance with |ΔNFR| <= {max_dissonance:.1f} "
            f"(current: {dnfr:.3f}). Apply IL (Coherence) first to stabilize.",
            context={"dnfr": dnfr, "max_dissonance": max_dissonance},
            suggestion="Apply IL (Coherence) first to stabilize.",
        )

    # 5. Validate phase compatibility (suboptimality warning; the U3 hard gate
    #    is the enforcer that raises on genuine incompatibility)
    if warn_phase_misalignment and neighbors:
        theta_node = finite_node_real(
            G.nodes[node],
            ALIAS_THETA,
            0.0,
            operator="Resonance",
            label="theta state",
        )
        phase_diff = _phase_alignment_difference(G, neighbors, theta_node)
        if phase_diff is None:
            warnings.warn(
                "RA phase alignment unavailable: zero represented neighbor resultant.",
                UserWarning,
                stacklevel=3,
            )
        elif phase_diff > max_phase_diff:
            warnings.warn(
                f"RA phase misalignment: Δφ = {phase_diff:.2f} > "
                f"{max_phase_diff:.2f}. "
                "Consider applying UM (Coupling) first for better resonance.",
                UserWarning,
                stacklevel=3,
            )


def diagnose_resonance_readiness(G: TNFRGraph, node: Any) -> dict[str, Any]:
    """Diagnose node readiness for RA (Resonance) operator.

    Provides a configured-policy diagnostic with readiness status and
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

    Notes
    -----
    Invalid consumed state or thresholds raise their scalar-admission error;
    they cannot be interpreted as successful checks. ``ready`` covers this
    configured policy only, not all live operator contracts. The mean-phase
    warning is distinct from mandatory U3 admission during execution.
    Exactly cancelling represented neighbor phasors make only that warning
    unavailable; invalid phases raise instead. No graph caches are written.

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
    >>> "Supply a source with |EPI|" in diag["recommendations"][0]
    True

    See Also
    --------
    validate_resonance_strict : Strict precondition validator
    """
    min_epi, max_dissonance, min_vf, max_phase_diff = _resonance_thresholds(G)
    epi, vf, dnfr = _resonance_state(G, node)
    theta = finite_node_real(
        G.nodes[node],
        ALIAS_THETA,
        0.0,
        operator="Resonance",
        label="theta state",
    )
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
            f"Supply a source with |EPI| >= {min_epi:.3f} " f"(current: {epi:.3f})"
        )

    # Check 2: Network connectivity
    if neighbor_count > 0:
        checks["network_connectivity"] = "passed"
    else:
        checks["network_connectivity"] = "failed"
        recommendations.append(
            "Apply UM (Coupling) to establish network connections before RA"
        )

    # Check 3: Structural frequency
    if vf >= min_vf:
        checks["structural_frequency"] = "passed"
    else:
        checks["structural_frequency"] = "failed"
        recommendations.append(_capacity_advice(vf, min_vf))

    # Check 4: Controlled dissonance
    if dnfr <= max_dissonance:
        checks["controlled_dissonance"] = "passed"
    else:
        checks["controlled_dissonance"] = "failed"
        recommendations.append(
            f"Apply IL (Coherence) to reduce |ΔNFR| from {dnfr:.3f} "
            f"to <= {max_dissonance:.1f}"
        )

    # Check 5: Phase alignment (warning only)
    phase_diff = None
    if neighbor_count > 0:
        phase_diff = _phase_alignment_difference(G, neighbors, theta)
        if phase_diff is None:
            checks["phase_alignment"] = "unavailable"
            recommendations.append(
                "Mean-phase alignment unavailable: zero represented neighbor resultant."
            )
        elif phase_diff <= max_phase_diff:
            checks["phase_alignment"] = "passed"
        else:
            checks["phase_alignment"] = "warning"
            recommendations.append(
                f"Consider applying UM (Coupling) to improve phase alignment "
                f"(current: Δφ = {phase_diff:.2f}, optimal: <= {max_phase_diff:.2f})"
            )
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
