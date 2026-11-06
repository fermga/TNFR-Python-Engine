"""Configured strict admission policy for OZ (Dissonance).

The policy imposes a signed EPI floor, pressure-magnitude ceiling, capacity
floor, a bounded unresolved-OZ history check and a degree warning. These
operational thresholds do not prove productive reorganization or exclude a
future collapse. They are separate from live grammar and its persistent U2 debt.

The retained May TNFR.pdf, printed/PDF pages 89-93 and 151, motivates critical
perturbation and resolution of accumulated dissonance. It does not derive
these numerical thresholds, the five-event window or the IL/THOL/NUL resolver
set. Page 151's premature-closure warning concerns SHA, not an OZ requirement.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ...errors import TNFRValueError

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_dissonance_strict"]


def validate_dissonance_strict(G: TNFRGraph, node: Any) -> None:
    """Validate the enabled strict OZ policy before its event.

    The configured admission conditions are:

    1. **Form floor**: signed EPI >= configured threshold
    2. **Pressure ceiling**: |ΔNFR| <= configured inclusive limit
    3. **Capacity floor**: νf >= configured threshold
    4. **No overload**: < 2 consecutive OZ without resolution (IL/THOL/NUL)
    5. **Network connectivity**: configured degree minimum (warning only)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation

    Raises
    ------
    ValueError
        If signed EPI or capacity is below its configured minimum
        If |ΔNFR| exceeds its configured maximum
        If overload detected (>= 2 OZ without resolver operators)

    Warnings
    --------
    UserWarning
        If node degree is below the configured warning threshold

    Notes
    -----
    Thresholds are configurable via graph metadata:
    - ``G.graph["OZ_MIN_EPI"]``: Minimum EPI (default 0.2)
    - ``G.graph["OZ_MAX_DNFR"]``: Maximum |ΔNFR| (default 0.8)
    - ``G.graph["OZ_MIN_VF"]``: Minimum νf (default 0.1)
    - ``G.graph["OZ_MIN_DEGREE"]``: Minimum degree for warning (default 1)

    **Resolver operators** that integrate dissonance (reset overload counter):
    - IL (Coherence): Stabilizes structure
    - THOL (Self-organization): Creates sub-EPIs
    - NUL (Contraction): Simplifies structure

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.dissonance import validate_dissonance_strict
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.nodes[node]["dnfr"] = 0.3
    >>> validate_dissonance_strict(G, node)  # OK - configured floors satisfied

    >>> G2, node2 = create_nfr("weak", epi=0.05, vf=1.0)
    >>> validate_dissonance_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: OZ precondition failed: EPI=0.050 < 0.2. Configured EPI floor not met.

    See Also
    --------
    tnfr.operators.preconditions : Base precondition validators
    tnfr.operators.definitions.Dissonance : Dissonance operator implementation
    """
    from ...alias import get_attr
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
    from .._argument_validation import finite_node_real
    from .._epi_domain import require_real_scalar_epi
    from .._oz_config import resolve_oz_strict_policy

    # Admit authoritative values before conversion or policy comparisons.
    epi = require_real_scalar_epi(
        get_attr(G.nodes[node], ALIAS_EPI, 0.0, strict=True, conv=lambda value: value),
        operator="Dissonance",
        label="EPI state",
    )
    vf = finite_node_real(
        G.nodes[node],
        ALIAS_VF,
        0.0,
        operator="Dissonance",
        label="nu_f state",
        lower=0.0,
    )
    dnfr = finite_node_real(
        G.nodes[node],
        ALIAS_DNFR,
        0.0,
        operator="Dissonance",
        label="DeltaNFR state",
    )
    degree = G.degree(node)

    # The EPI threshold remains signed; the magnitude/capacity/degree bounds
    # are nonnegative policies. Invalid thresholds cannot certify readiness.
    policy = resolve_oz_strict_policy(G.graph)
    min_epi, max_dnfr = policy.min_epi, policy.max_dnfr
    min_vf, min_degree = policy.min_vf, policy.min_degree

    # Precondition 1: Configured signed form floor, not a coherence observable.
    if epi < min_epi:
        raise TNFRValueError(
            f"OZ precondition failed: EPI={epi:.3f} < {min_epi:.3f}. "
            f"Configured EPI floor not met. "
            f"Supply form satisfying this policy; IL does not increase EPI.",
            context={"epi": epi, "min_epi": min_epi},
            suggestion="Supply form satisfying the configured signed EPI floor.",
        )

    # Precondition 2: Configured inclusive pressure-magnitude ceiling.
    if abs(dnfr) > max_dnfr:
        raise TNFRValueError(
            f"OZ precondition failed: |ΔNFR|={abs(dnfr):.3f} > {max_dnfr:.3f}. "
            f"Configured pressure ceiling exceeded. "
            f"Suggestion: Apply IL (Coherence) first to reduce ΔNFR before introducing more dissonance.",
            context={"dnfr": dnfr, "max_dnfr": max_dnfr},
            suggestion="Apply IL (Coherence) first to reduce ΔNFR before introducing more dissonance.",
        )

    # Precondition 3: Configured nonnegative capacity floor.
    if vf < min_vf:
        raise TNFRValueError(
            f"OZ precondition failed: νf={vf:.3f} < {min_vf:.3f}. "
            f"Configured capacity floor not met. "
            f"Suggestion: Increase νf before applying OZ.",
            context={"vf": vf, "min_vf": min_vf},
            suggestion="Increase νf before applying OZ.",
        )

    # Precondition 4: Bounded chronological resolver policy; U2 is separate.
    _validate_oz_no_overload(G, node)

    # Precondition 5: Connectivity warning, not a proved bifurcation criterion.
    if degree < min_degree:
        warnings.warn(
            f"OZ warning: Node {node!r} has low connectivity (degree={degree} < {min_degree}). "
            f"Check that the supplied support matches the intended operation.",
            UserWarning,
            stacklevel=3,
        )

    # Store validation context for telemetry
    G.nodes[node]["_oz_precondition_context"] = {
        "epi": epi,
        "dnfr": dnfr,
        "vf": vf,
        "degree": degree,
        "validation_passed": True,
    }


def _validate_oz_no_overload(G: TNFRGraph, node: Any) -> None:
    """Reject an unresolved OZ in the selected recent-history window.

    Only a later IL, THOL or NUL resolves an earlier OZ for this configured
    policy. A resolver before the most recent OZ cannot cover that event.
    This bounded check neither replaces persistent U2 debt nor establishes a
    dynamical instability or entropy increase.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : Any
        Node identifier to check for overload

    Raises
    ------
    ValueError
        If >= 1 OZ application detected in recent history without resolver
        (since we're about to apply another OZ, that would make it >= 2)

    Notes
    -----
    **Resolver operators** that integrate dissonance:
    - coherence (IL): Stabilizes structure, reduces ΔNFR
    - self_organization (THOL): Creates sub-EPIs via bifurcation
    - contraction (NUL): Simplifies structure

    This check examines the last 5 operators in glyph_history to detect
    overload patterns. Since validation is called BEFORE the operator is
    applied, if we find >= 1 OZ in history without a resolver, applying
    another OZ would create overload.

    References
    ----------
    TNFR.pdf §2.3.6 - Sobrecarga disonante: "acumulación excesiva de OZ sin
    paso a mutación"
    """
    from ...operators.grammar import glyph_function_name
    from ..grammar_debt import read_grammar_state

    # Missing evidence remains unavailable; malformed stored evidence rejects.
    history = read_grammar_state(G.nodes[node]).history
    if not history:
        # No history - first operator application, cannot be overload
        return

    # Convert deque to list and examine recent history (last 5 operations)
    history_list = list(history)
    recent_history = history_list[-5:] if len(history_list) >= 5 else history_list

    resolvers = {"coherence", "self_organization", "contraction"}
    recent_names = [glyph_function_name(g) for g in recent_history]
    oz_count = 0
    for name in recent_names:
        if name in resolvers:
            oz_count = 0
        elif name == "dissonance":
            oz_count += 1

    if oz_count >= 1:
        raise TNFRValueError(
            f"OZ precondition failed: Sobrecarga disonante detected. "
            f"Found {oz_count} OZ in recent history without a later resolver. "
            f"The configured overload policy rejects another unresolved OZ. "
            f"Suggestion: Apply IL (Coherence), THOL (Self-organization), or NUL (Contraction) "
            f"to integrate dissonance before introducing more.",
            context={"oz_count": oz_count, "recent_history": recent_names},
            suggestion="Apply IL (Coherence), THOL (Self-organization), or NUL (Contraction) to integrate dissonance before introducing more.",
        )
