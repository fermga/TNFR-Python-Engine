"""Canonical precondition validators for ZHIR (Mutation) operator.

Implements comprehensive validation of mutation prerequisites including
threshold verification, grammar U4b compliance, and structural readiness.

This module provides strict, modular validation for the Mutation (ZHIR) operator,
aligning with the architectural pattern used by Coherence (IL) and Dissonance (OZ).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import NodeId, TNFRGraph
    import logging

from ...alias import get_attr
from ...config.operator_names import (
    BIFURCATION_WINDOW,
    DESTABILIZERS,
)
from ...constants.aliases import ALIAS_VF
from .._mutation_gate import (
    mutation_threshold_sample,
    validate_mutation_capacity,
    validate_mutation_threshold,
)
from ..grammar_debt import node_has_prior_coherence
from . import OperatorPreconditionError

__all__ = [
    "validate_mutation_strict",
    "validate_threshold_crossing",
    "validate_grammar_u4b",
    "record_destabilizer_context",
    "diagnose_mutation_readiness",
]


def validate_mutation_strict(G: TNFRGraph, node: NodeId) -> None:
    """Comprehensive canonical validation for ZHIR.

    Validates all TNFR requirements for mutation (AGENTS.md §11, TNFR.pdf §2.2.11):

    1. **Minimum νf**: Reorganization capacity for phase transformation
    2. **Threshold crossing**: ∂EPI/∂t > ξ (structural velocity sufficient)
    3. **Grammar U4b Part 1**: Prior IL (Coherence) for stable base
    4. **Grammar U4b Part 2**: Recent destabilizer (~3 ops) for threshold energy
    5. **Sufficient history**: EPI history for velocity calculation

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If any canonical requirement not met

    Notes
    -----
    This function implements strict validation when:
    - ``VALIDATE_OPERATOR_PRECONDITIONS=True`` (global strict mode)
    - Individual flags enabled (ZHIR_REQUIRE_IL_PRECEDENCE, etc.)

    The threshold is always a hard operator contract. U4b is checked here when
    global strict validation or either explicit U4b flag is enabled; the normal
    high-level operator path also enforces U4b through the grammar.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.mutation import validate_mutation_strict
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.nodes[node]["epi_history"] = [0.4, 0.5]
    >>> G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    >>> # This would raise if U4b not satisfied
    >>> # validate_mutation_strict(G, node)  # doctest: +SKIP
    """
    import logging

    logger = logging.getLogger(__name__)

    # 1. Minimum νf validation
    _validate_minimum_vf(G, node)

    # 2. Threshold crossing validation (∂EPI/∂t > ξ)
    validate_threshold_crossing(G, node, logger)

    # 3. Grammar U4b validation
    strict_validation = bool(G.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False))
    if strict_validation or bool(
        G.graph.get("ZHIR_REQUIRE_IL_PRECEDENCE", False)
    ) or bool(G.graph.get("ZHIR_REQUIRE_DESTABILIZER", False)):
        validate_grammar_u4b(G, node, logger)


def _validate_minimum_vf(G: TNFRGraph, node: NodeId) -> None:
    """Validate minimum structural frequency for phase transformation."""
    validate_mutation_capacity(G.nodes[node], G.graph)


def validate_threshold_crossing(
    G: TNFRGraph, node: NodeId, logger: logging.Logger | None = None
) -> None:
    """Validate ∂EPI/∂t > ξ requirement for phase transformation.

    ZHIR is a phase transformation that requires sufficient structural reorganization
    velocity to justify the transition. The threshold ξ represents the minimum rate
    of structural change needed for a phase shift to be physically meaningful.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate
    logger : logging.Logger, optional
        Logger for telemetry output

    The canonical comparison is signed and strict. Equality, negative change,
    invalid samples, and insufficient history all raise without writing node
    metadata.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.nodes[node]["epi_history"] = [0.3, 0.5]  # velocity = 0.2
    >>> G.graph["ZHIR_THRESHOLD_XI"] = 0.1
    >>> validate_threshold_crossing(G, node)  # Should pass (0.2 > 0.1)
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    sample = validate_mutation_threshold(G.nodes[node], G.graph)
    logger.info(
        "Node %r: ZHIR threshold crossed (signed dEPI/dt=%g > xi=%g)",
        node,
        sample.depi_dt,
        sample.xi,
    )


def validate_grammar_u4b(
    G: TNFRGraph, node: NodeId, logger: logging.Logger | None = None
) -> None:
    """Validate U4b: IL precedence + recent destabilizer.

    Grammar rule U4b (BIFURCATION DYNAMICS - Transformers Need Context) requires:

    1. **Prior IL (Coherence)**: Stable base for transformation
    2. **Recent destabilizer**: OZ/VAL/etc within ~3 operations for threshold energy

    This is a STRONG canonicity rule derived from bifurcation theory - phase
    transformations need both stability (IL) and elevated energy (destabilizer).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate
    logger : logging.Logger, optional
        Logger for telemetry output

    Raises
    ------
    OperatorPreconditionError
        If U4b requirements not met when strict validation enabled

    Notes
    -----
    Validation is strict when:
    - ``VALIDATE_OPERATOR_PRECONDITIONS=True`` (global)
    - ``ZHIR_REQUIRE_IL_PRECEDENCE=True`` (Part 1)
    - ``ZHIR_REQUIRE_DESTABILIZER=True`` (Part 2)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators import Coherence, Dissonance
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    >>> # Apply required sequence
    >>> Coherence()(G, node)  # IL for stable base
    >>> Dissonance()(G, node)  # OZ for destabilization
    >>> # Now validate_grammar_u4b would pass
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    # Get glyph history
    glyph_history = G.nodes[node].get("glyph_history", [])
    strict_validation = bool(G.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False))
    require_il = strict_validation or bool(
        G.graph.get("ZHIR_REQUIRE_IL_PRECEDENCE", False)
    )
    require_destabilizer = strict_validation or bool(
        G.graph.get("ZHIR_REQUIRE_DESTABILIZER", False)
    )
    if not glyph_history:
        if require_il or require_destabilizer:
            raise OperatorPreconditionError(
                "Mutation", "U4b cannot be verified without glyph history"
            )
        logger.warning(
            f"Node {node}: No glyph history available. Cannot verify U4b compliance."
        )
        return

    # Import glyph_function_name to convert glyphs to operator names
    from ..grammar import glyph_function_name

    # Convert history to operator names
    history_names = [glyph_function_name(g) for g in glyph_history]

    # Part 1: Check for prior IL (Coherence)
    il_found = node_has_prior_coherence(G.nodes[node])

    if require_il and not il_found:
        raise OperatorPreconditionError(
            "Mutation",
            "U4b violation: ZHIR requires prior IL (Coherence) for stable transformation base. "
            "Apply Coherence before mutation sequence. "
            f"Recent history: {history_names[-5:] if len(history_names) > 5 else history_names}",
        )

    if il_found:
        logger.debug(
            f"Node {node}: ZHIR IL precedence satisfied (prior Coherence found)"
        )

    # Compute the context without writing it. A rejected validation must not
    # leave metadata that falsely reports an accepted mutation path.
    context = record_destabilizer_context(G, node, logger, record=False)
    destabilizer_found = context.get("destabilizer_operator")

    if require_destabilizer and destabilizer_found is None:
        recent_history = context.get("recent_history", [])
        raise OperatorPreconditionError(
            "Mutation",
            "U4b violation: ZHIR requires recent destabilizer (OZ/VAL/etc) within ~3 ops. "
            f"Recent history: {recent_history}. "
            "Apply Dissonance or Expansion to elevate ΔNFR first.",
        )
    G.nodes[node]["_mutation_context"] = context


def record_destabilizer_context(
    G: TNFRGraph,
    node: NodeId,
    logger: logging.Logger | None = None,
    *,
    record: bool = True,
) -> dict:
    """Detect and record which destabilizer enabled the current mutation.

    This implements R4 Extended telemetry by analyzing the glyph_history
    to find the most recent destabilizer (DESTABILIZERS = {OZ, ZHIR, VAL})
    within the single structural-relaxation window BIFURCATION_WINDOW.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node being mutated
    logger : logging.Logger, optional
        Logger for telemetry output
    record : bool, default True
        Store the resolved context on the node. Diagnostics pass ``False`` to
        remain read-only.

    Returns
    -------
    dict
        Destabilizer context with keys:
        - destabilizer_operator: Name of destabilizer glyph (or None)
        - destabilizer_distance: Operations since destabilizer (or None)
        - recent_history: Last N operator names

    Notes
    -----
    The destabilizer context is stored in node['_mutation_context'] for
    structural tracing and post-hoc analysis. This enables understanding
    of bifurcation pathways without breaking TNFR structural invariants.

    The reach is a SINGLE emergent window (BIFURCATION_WINDOW): the
    structural-pressure relaxation time is topology-independent (the mean
    L_rw eigenvalue is exactly trace/N = 1), so the earlier graduated
    strong/moderate/weak split does not survive the dynamics.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators import Dissonance
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> Dissonance()(G, node)  # Apply OZ (a destabilizer)
    >>> context = record_destabilizer_context(G, node)
    >>> context["destabilizer_operator"]  # doctest: +SKIP
    'dissonance'
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    # Get glyph history from node
    history = G.nodes[node].get("glyph_history", [])
    if not history:
        # No history available, mutation enabled by external factors
        context = {
            "destabilizer_operator": None,
            "destabilizer_distance": None,
            "recent_history": [],
        }
        if record:
            G.nodes[node]["_mutation_context"] = context
        return context

    # Import glyph_function_name to convert glyphs to operator names
    from ..grammar import glyph_function_name

    # Get recent history within the single structural-relaxation window
    recent = (
        list(history)[-BIFURCATION_WINDOW:]
        if len(history) > BIFURCATION_WINDOW
        else list(history)
    )
    recent_names = [glyph_function_name(g) for g in recent]

    # Search backwards for any destabilizer ({OZ, ZHIR, VAL}) within the window
    destabilizer_found = None
    destabilizer_distance = None

    for i, op_name in enumerate(reversed(recent_names)):
        distance = i + 1  # Distance from mutation (1 = immediate predecessor)
        if op_name in DESTABILIZERS and distance <= BIFURCATION_WINDOW:
            destabilizer_found = op_name
            destabilizer_distance = distance
            break

    # Store context in node metadata for telemetry
    context = {
        "destabilizer_operator": destabilizer_found,
        "destabilizer_distance": destabilizer_distance,
        "recent_history": recent_names,
    }
    if record:
        G.nodes[node]["_mutation_context"] = context

    # Log telemetry for structural tracing
    if destabilizer_found:
        logger.info(
            f"Node {node}: ZHIR enabled by destabilizer "
            f"({destabilizer_found}) at distance {destabilizer_distance}"
        )
    else:
        logger.warning(
            f"Node {node}: ZHIR without detectable destabilizer in history. "
            f"Recent operators: {recent_names}"
        )

    return context


def diagnose_mutation_readiness(G: TNFRGraph, node: NodeId) -> dict:
    """Comprehensive diagnostic for ZHIR readiness.

    Analyzes node state and returns detailed readiness report with:
    - Overall readiness boolean
    - Individual check results
    - Recommendations for corrections

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to diagnose

    Returns
    -------
    dict
        Diagnostic report with structure:
        {
            "ready": bool,
            "checks": {
                "minimum_vf": {"passed": bool, "value": float, "threshold": float},
                "threshold_crossing": {"passed": bool, "depi_dt": float, "xi": float},
                "il_precedence": {"passed": bool, "found": bool},
                "recent_destabilizer": {"passed": bool, "operator": str|None, "distance": int|None},
                "history_length": {"passed": bool, "length": int, "required": int},
            },
            "recommendations": [str, ...]
        }

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> report = diagnose_mutation_readiness(G, node)
    >>> report["ready"]  # doctest: +SKIP
    False
    >>> report["recommendations"]  # doctest: +SKIP
    ['Apply IL (Coherence) for stable base', 'Apply OZ (Dissonance) to elevate ΔNFR', ...]
    """
    import logging

    checks = {}
    recommendations = []

    # Check 1: Minimum νf
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    min_vf = float(G.graph.get("ZHIR_MIN_VF", 0.0))
    vf_passed = (
        math.isfinite(vf)
        and math.isfinite(min_vf)
        and min_vf >= 0.0
        and vf > 0.0
        and vf >= min_vf
    )
    checks["minimum_vf"] = {
        "passed": vf_passed,
        "value": vf,
        "threshold": min_vf,
    }
    if not vf_passed:
        recommendations.append(
            f"Increase νf: current={vf:.3f}, required={min_vf:.3f}. "
            f"Apply AL (Emission) or NAV (Transition) to boost structural frequency."
        )

    # Check 2: use the same signed, strict, non-mutating sample as runtime.
    sample = None
    try:
        sample = mutation_threshold_sample(G.nodes[node], G.graph)
    except OperatorPreconditionError as exc:
        checks["threshold_crossing"] = {
            "passed": False,
            "depi_dt": None,
            "xi": G.graph.get("ZHIR_THRESHOLD_XI"),
            "reason": str(exc),
        }
        recommendations.append(str(exc))
    else:
        checks["threshold_crossing"] = {
            "passed": sample.crossed,
            "depi_dt": sample.depi_dt,
            "xi": sample.xi,
        }
        if not sample.crossed:
            recommendations.append(
                "Increase positive structural velocity: "
                f"signed dEPI/dt={sample.depi_dt:.3f} must exceed "
                f"xi={sample.xi:.3f}."
            )

    # Check 3: IL precedence
    glyph_history = G.nodes[node].get("glyph_history", [])
    if glyph_history:
        from ..grammar import glyph_function_name

        history_names = [glyph_function_name(g) for g in glyph_history]
    else:
        history_names = []

    il_found = node_has_prior_coherence(G.nodes[node])

    checks["il_precedence"] = {
        "passed": il_found,
        "found": il_found,
    }
    if not il_found:
        recommendations.append(
            "Apply IL (Coherence) for stable transformation base (U4b Part 1)."
        )

    # Check 4: Recent destabilizer
    logger = logging.getLogger(__name__)
    context = record_destabilizer_context(G, node, logger, record=False)
    destabilizer_found = context.get("destabilizer_operator") is not None

    checks["recent_destabilizer"] = {
        "passed": destabilizer_found,
        "distance": context.get("destabilizer_distance"),
        "operator": context.get("destabilizer_operator"),
    }
    if not destabilizer_found:
        recommendations.append(
            "Apply destabilizer (OZ/VAL) within last ~3 operations to elevate ΔNFR (U4b Part 2)."
        )

    # Check 5: the successful threshold sample is also the authority for
    # history selection. This prevents a valid timestamped physical history
    # from being contradicted by a separate legacy-only length check.
    min_history = 2
    history_source = sample.history_key if sample is not None else None
    history_length = 0
    if sample is not None:
        raw_history = G.nodes[node].get(sample.history_key)
        try:
            history_length = len(raw_history)
        except (OverflowError, TypeError):
            # A successful certificate proves that two effective samples
            # exist even if the supplied replayable input exposes no len().
            history_length = min_history
    history_passed = sample is not None
    checks["history_length"] = {
        "passed": history_passed,
        "length": history_length,
        "required": min_history,
        "source": history_source,
    }
    # Already covered by threshold check recommendations

    # Overall readiness
    all_passed = all(check.get("passed", False) for check in checks.values())

    return {
        "ready": all_passed,
        "checks": checks,
        "recommendations": recommendations,
    }
