"""Strict precondition validation for IL (Coherence) operator.

This module implements canonical precondition validation for the Coherence (IL)
structural operator according to TNFR.pdf §2.2.1. IL requires specific structural
conditions to maintain TNFR operational fidelity:

1. **Active EPI**: Node must have non-zero structural form (EPI > 0)
2. **Active νf**: Structural frequency must exceed the minimum threshold
3. **Pressure magnitude**: Either sign of ΔNFR is a valid IL input
4. **Network coupling**: Connections enable optional phase locking

These validations protect structural integrity by ensuring IL is only applied to
nodes in the appropriate state for coherence stabilization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = [
    "coherence_precondition_warnings",
    "diagnose_coherence_readiness",
    "emit_coherence_precondition_warnings",
    "validate_coherence_strict",
]


def validate_coherence_strict(
    G: TNFRGraph,
    node: Any,
    *,
    emit_warnings: bool = True,
) -> None:
    """Validate strict canonical preconditions for IL (Coherence) operator.

    According to TNFR.pdf §2.2.1, Coherence (IL - Coherencia estructural) requires:

    1. **Active EPI**: EPI > 0 (node must have active structural form)
    2. **Active νf**: νf > threshold (sufficient structural frequency)
    3. **Pressure magnitude**: positive and negative ΔNFR are stabilized alike
    4. **Network coupling**: degree > 0 (connections for phase locking)

    IL does not write EPI, so EPI headroom is not an admission condition.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation

    Raises
    ------
    ValueError
        If EPI <= 0 (no structural form to stabilize)
        If νf <= 0 (no structural frequency - consider VAL/Expansion or NAV/Transition)

    Warnings
    --------
    UserWarning
        If ΔNFR == 0 (no reorganization pressure - IL may be redundant)
        If |ΔNFR| > critical threshold (high pressure - repeat IL or use THOL)
        If node is isolated (no connections - phase locking will have no effect)

    Notes
    -----
    Thresholds are configurable via:
    - Graph metadata: ``G.graph["IL_PRECONDITIONS"]``
    - Module defaults: :data:`tnfr.config.thresholds.EPI_IL_MIN`, etc.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.coherence import validate_coherence_strict
    >>> G, node = create_nfr("test", epi=0.5, vf=0.9)
    >>> G.nodes[node]["dnfr"] = 0.1
    >>> validate_coherence_strict(G, node)  # OK - active state with reorganization pressure

    >>> G2, node2 = create_nfr("inactive", epi=0.0, vf=0.9)
    >>> validate_coherence_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: IL precondition failed: EPI=0.000 <= 0.0. IL requires active structural form.

    See Also
    --------
    tnfr.config.thresholds : Configurable threshold constants
    tnfr.operators.preconditions : Base precondition validators
    tnfr.operators.definitions.Coherence : Coherence operator implementation
    diagnose_coherence_readiness : Diagnostic function for IL readiness
    """
    from ...alias import get_attr
    from ...config.thresholds import EPI_IL_MIN, VF_IL_MIN
    from ...constants.aliases import ALIAS_EPI, ALIAS_VF

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    # Get configurable thresholds (allow override via graph metadata)
    config = G.graph.get("IL_PRECONDITIONS", {})
    min_epi = float(config.get("min_epi", EPI_IL_MIN))
    min_vf = float(config.get("min_vf", VF_IL_MIN))

    # Precondition 1: EPI must be active (non-zero structural form)
    # IL stabilizes existing structure - requires structure to exist
    if epi <= min_epi:
        raise ValueError(
            f"IL precondition failed: EPI={epi:.3f} <= {min_epi:.3f}. "
            f"IL requires active structural form (non-zero EPI). "
            f"Suggestion: Apply AL (Emission) first to seed structural form."
        )

    # Precondition 2: νf must be active (sufficient structural frequency)
    # IL requires active reorganization capacity to effect stabilization
    if vf <= min_vf:
        raise ValueError(
            f"IL precondition failed: νf={vf:.3f} <= {min_vf:.3f}. "
            f"Structural frequency too low for coherence stabilization. "
            f"Suggestion: Apply VAL (Expansion) or NAV (Transition) to raise νf first."
        )

    if emit_warnings:
        emit_coherence_precondition_warnings(G, node)


def coherence_precondition_warnings(
    G: TNFRGraph, node: Any
) -> tuple[str, ...]:
    """Return strict IL warnings without emitting or mutating graph state."""

    from ...alias import get_attr
    from ...config.thresholds import DNFR_IL_CRITICAL
    from ...constants.aliases import ALIAS_DNFR

    config = G.graph.get("IL_PRECONDITIONS", {})
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
    magnitude = abs(dnfr)
    critical = float(
        config.get("dnfr_critical_threshold", DNFR_IL_CRITICAL)
    )
    messages: list[str] = []
    if bool(config.get("warn_zero_dnfr", True)) and magnitude == 0.0:
        messages.append(
            f"IL warning: Node {node!r} has |ΔNFR|=0. "
            "No reorganization pressure to stabilize. "
            "IL application may be redundant in this state."
        )
    if magnitude > critical:
        messages.append(
            f"IL warning: Node {node!r} has |ΔNFR|={magnitude:.3f} "
            f"> {critical:.3f}. High reorganization pressure may require "
            "repeated IL or THOL stabilization."
        )
    if (
        bool(config.get("warn_isolated", True))
        and G.degree(node) == 0
        and len(G) > 1
    ):
        messages.append(
            f"IL warning: Node {node!r} isolated (degree=0). "
            "Phase locking will have no effect. "
            "Consider applying UM (Coupling) first to connect the node."
        )
    return tuple(messages)


def emit_coherence_precondition_warnings(
    G: TNFRGraph, node: Any
) -> None:
    """Emit the pure warning set for one accepted direct IL request."""

    import warnings

    for message in coherence_precondition_warnings(G, node):
        warnings.warn(message, UserWarning, stacklevel=3)


def diagnose_coherence_readiness(G: TNFRGraph, node: Any) -> dict:
    """Diagnose node readiness for IL (Coherence) operator.

    Performs all canonical precondition checks and returns a diagnostic report
    with readiness status and actionable recommendations.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : Any
        Node identifier for diagnosis

    Returns
    -------
    dict
        Diagnostic report with the following structure:

        - ``node``: Node identifier
        - ``ready``: bool - Overall readiness (all critical checks passed)
        - ``checks``: dict - Individual check results

          - ``epi_active``: bool - EPI > 0
          - ``vf_active``: bool - νf > 0
          - ``dnfr_present``: bool - |ΔNFR| > 0 (warning only)
          - ``dnfr_not_critical``: bool - |ΔNFR| < critical (warning only)
          - ``has_connections``: bool - degree > 0 (warning only)

        - ``values``: dict - Current node attribute values

          - ``epi``: Current EPI value
          - ``vf``: Current νf value
          - ``dnfr``: Current ΔNFR value
          - ``degree``: Node degree

        - ``recommendations``: list[str] - Actionable suggestions

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.coherence import diagnose_coherence_readiness
    >>> G, node = create_nfr("test", epi=0.5, vf=0.9)
    >>> G.nodes[node]["dnfr"] = 0.1
    >>> report = diagnose_coherence_readiness(G, node)
    >>> report["ready"]
    True
    >>> "✓ Node ready" in report["recommendations"][0]
    True

    See Also
    --------
    validate_coherence_strict : Strict precondition validator
    """
    from ...alias import get_attr
    from ...config.thresholds import DNFR_IL_CRITICAL, EPI_IL_MIN, VF_IL_MIN
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
    degree = G.degree(node)

    # Get configurable thresholds
    config = G.graph.get("IL_PRECONDITIONS", {})
    min_epi = float(config.get("min_epi", EPI_IL_MIN))
    min_vf = float(config.get("min_vf", VF_IL_MIN))
    dnfr_critical = float(config.get("dnfr_critical_threshold", DNFR_IL_CRITICAL))

    # Perform checks
    checks = {
        "epi_active": epi > min_epi,
        "vf_active": vf > min_vf,
        "dnfr_present": abs(dnfr) > 0.0,
        "dnfr_not_critical": abs(dnfr) <= dnfr_critical,
        "has_connections": degree > 0,
    }

    # Critical checks (hard failures)
    critical_checks = ["epi_active", "vf_active"]
    all_critical_passed = all(checks[key] for key in critical_checks)

    # Generate recommendations
    recommendations = []

    if not checks["epi_active"]:
        recommendations.append("Apply AL (Emission) to seed structural form")

    if not checks["vf_active"]:
        recommendations.append(
            "Apply VAL (Expansion) or NAV (Transition) to raise νf"
        )

    if not checks["dnfr_present"]:
        recommendations.append("⚠ ΔNFR=0 - IL may be redundant")

    if not checks["dnfr_not_critical"]:
        recommendations.append("⚠ High |ΔNFR| - consider repeated IL or THOL")

    if not checks["has_connections"]:
        recommendations.append(
            "⚠ Isolated node - consider UM (Coupling) to enable phase locking"
        )

    if all_critical_passed:
        recommendations.insert(0, "✓ Node ready for IL (Coherence)")

    return {
        "node": node,
        "ready": all_critical_passed,
        "checks": checks,
        "values": {
            "epi": epi,
            "vf": vf,
            "dnfr": dnfr,
            "degree": degree,
        },
        "recommendations": recommendations,
    }
