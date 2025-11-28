"""Strict precondition validation for IL (Coherence) operator.

This module implements canonical precondition validation for the Coherence (IL)
structural operator according to TNFR.pdf §2.2.1. IL requires specific structural
conditions to maintain TNFR operational fidelity:

1. **Active EPI**: Node must have non-zero structural form (EPI > 0)
2. **Non-saturated EPI**: EPI must be below maximum (leave room for stabilization increment)
3. **Active νf**: Structural frequency must exceed minimum threshold
4. **ΔNFR presence**: While IL reduces ΔNFR, some reorganization pressure should exist
5. **ΔNFR not critical**: Excessive ΔNFR may require OZ (Dissonance) first
6. **Network coupling**: Connections enable phase locking

These validations protect structural integrity by ensuring IL is only applied to
nodes in the appropriate state for coherence stabilization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_coherence_strict", "diagnose_coherence_readiness"]


def validate_coherence_strict(G: TNFRGraph, node: Any) -> None:
    """Validate strict canonical preconditions for IL (Coherence) operator.

    According to TNFR.pdf §2.2.1, Coherence (IL - Coherencia estructural) requires:

    1. **Active EPI**: EPI > 0 (node must have active structural form)
    2. **Non-saturated EPI**: EPI < maximum (leave room for stabilization)
    3. **Active νf**: νf > threshold (sufficient structural frequency)
    4. **ΔNFR presence**: ΔNFR > 0 (reorganization pressure to stabilize)
    5. **ΔNFR not critical**: ΔNFR < critical threshold (manageable instability)
    6. **Network coupling**: degree > 0 (connections for phase locking)

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
        If EPI >= maximum (node saturated - consider NUL/Contraction)
        If νf <= 0 (no structural frequency - consider AL/Emission or NAV/Transition)

    Warnings
    --------
    UserWarning
        If ΔNFR == 0 (no reorganization pressure - IL may be redundant)
        If ΔNFR > critical threshold (high instability - consider OZ/Dissonance first)
        If node is isolated (no connections - phase locking will have no effect)

    Notes
    -----
    Thresholds are configurable via:
    - Graph metadata: ``G.graph["IL_PRECONDITIONS"]``
    - Module defaults: :data:`tnfr.config.thresholds.EPI_IL_MAX`, etc.

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
    import warnings

    from ...alias import get_attr
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
    from ...config.thresholds import (
        DNFR_IL_CRITICAL,
        EPI_IL_MAX,
        EPI_IL_MIN,
        VF_IL_MIN,
    )

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

    # Get configurable thresholds (allow override via graph metadata)
    config = G.graph.get("IL_PRECONDITIONS", {})
    min_epi = float(config.get("min_epi", EPI_IL_MIN))
    max_epi = float(config.get("max_epi", EPI_IL_MAX))
    min_vf = float(config.get("min_vf", VF_IL_MIN))
    dnfr_critical = float(config.get("dnfr_critical_threshold", DNFR_IL_CRITICAL))
    warn_isolated = bool(config.get("warn_isolated", True))
    warn_zero_dnfr = bool(config.get("warn_zero_dnfr", True))

    # Precondition 1: EPI must be active (non-zero structural form)
    # IL stabilizes existing structure - requires structure to exist
    if epi <= min_epi:
        raise ValueError(
            f"IL precondition failed: EPI={epi:.3f} <= {min_epi:.3f}. "
            f"IL requires active structural form (non-zero EPI). "
            f"Suggestion: Apply AL (Emission) first to activate node."
        )

    # Precondition 2: EPI must not be saturated (leave room for stabilization)
    # IL may increment EPI slightly during stabilization
    if epi >= max_epi:
        raise ValueError(
            f"IL precondition failed: EPI={epi:.3f} >= {max_epi:.3f}. "
            f"Node saturated, cannot stabilize further. "
            f"Suggestion: Consider NUL (Contraction) to consolidate structure."
        )

    # Precondition 3: νf must be active (sufficient structural frequency)
    # IL requires active reorganization capacity to effect stabilization
    if vf <= min_vf:
        raise ValueError(
            f"IL precondition failed: νf={vf:.3f} <= {min_vf:.3f}. "
            f"Structural frequency too low for coherence stabilization. "
            f"Suggestion: Apply AL (Emission) or NAV (Transition) to activate νf first."
        )

    # Precondition 4: Check for ΔNFR presence (warning only - not hard failure)
    # IL reduces ΔNFR, but if ΔNFR is already zero, IL is redundant
    if warn_zero_dnfr and dnfr == 0.0:
        warnings.warn(
            f"IL warning: Node {node!r} has ΔNFR=0. "
            f"No reorganization pressure to stabilize. "
            f"IL application may be redundant in this state.",
            UserWarning,
            stacklevel=3,
        )

    # Precondition 5: Check for critically high ΔNFR (warning only)
    # Excessive instability may require controlled dissonance (OZ) before stabilization
    if dnfr > dnfr_critical:
        warnings.warn(
            f"IL warning: Node {node!r} has ΔNFR={dnfr:.3f} > {dnfr_critical:.3f}. "
            f"High reorganization pressure - stabilization may be difficult. "
            f"Consider applying OZ (Dissonance) → IL sequence for better control.",
            UserWarning,
            stacklevel=3,
        )

    # Precondition 6: Check for network connections (warning only)
    # Isolated nodes can stabilize but phase locking will have no effect
    degree = G.degree(node)
    network_size = len(G)
    if warn_isolated and degree == 0 and network_size > 1:
        warnings.warn(
            f"IL warning: Node {node!r} isolated (degree=0). "
            f"Phase locking will have no effect. "
            f"Consider applying UM (Coupling) first to connect node to network.",
            UserWarning,
            stacklevel=3,
        )


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
          - ``epi_not_saturated``: bool - EPI < maximum
          - ``vf_active``: bool - νf > 0
          - ``dnfr_present``: bool - ΔNFR > 0 (warning only)
          - ``dnfr_not_critical``: bool - ΔNFR < critical (warning only)
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
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
    from ...config.thresholds import (
        DNFR_IL_CRITICAL,
        EPI_IL_MAX,
        EPI_IL_MIN,
        VF_IL_MIN,
    )

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
    degree = G.degree(node)

    # Get configurable thresholds
    config = G.graph.get("IL_PRECONDITIONS", {})
    min_epi = float(config.get("min_epi", EPI_IL_MIN))
    max_epi = float(config.get("max_epi", EPI_IL_MAX))
    min_vf = float(config.get("min_vf", VF_IL_MIN))
    dnfr_critical = float(config.get("dnfr_critical_threshold", DNFR_IL_CRITICAL))

    # Perform checks
    checks = {
        "epi_active": epi > min_epi,
        "epi_not_saturated": epi < max_epi,
        "vf_active": vf > min_vf,
        "dnfr_present": dnfr > 0.0,
        "dnfr_not_critical": dnfr < dnfr_critical,
        "has_connections": degree > 0,
    }

    # Critical checks (hard failures)
    critical_checks = ["epi_active", "epi_not_saturated", "vf_active"]
    all_critical_passed = all(checks[key] for key in critical_checks)

    # Generate recommendations
    recommendations = []

    if not checks["epi_active"]:
        recommendations.append("Apply AL (Emission) to activate node")

    if checks["epi_active"] and not checks["epi_not_saturated"]:
        recommendations.append("Apply NUL (Contraction) to consolidate saturated structure")

    if not checks["vf_active"]:
        recommendations.append("Apply AL (Emission) or NAV (Transition) to activate νf")

    if checks["epi_active"] and not checks["dnfr_present"]:
        recommendations.append("⚠ ΔNFR=0 - IL may be redundant")

    if checks["dnfr_present"] and not checks["dnfr_not_critical"]:
        recommendations.append("⚠ High ΔNFR - consider OZ (Dissonance) → IL sequence")

    if not checks["has_connections"]:
        recommendations.append("⚠ Isolated node - consider UM (Coupling) to enable phase locking")

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
