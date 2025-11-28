"""Canonical precondition validators for ZHIR (Mutation) operator.

Implements comprehensive validation of mutation prerequisites including
threshold verification, grammar U4b compliance, and structural readiness.

This module provides strict, modular validation for the Mutation (ZHIR) operator,
aligning with the architectural pattern used by Coherence (IL) and Dissonance (OZ).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import NodeId, TNFRGraph
    import logging

from ...alias import get_attr
from ...config.operator_names import (
    BIFURCATION_WINDOWS,
    DESTABILIZERS_MODERATE,
    DESTABILIZERS_STRONG,
    DESTABILIZERS_WEAK,
)
from ...constants.aliases import ALIAS_VF
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

    For backward compatibility, threshold and U4b checks may be soft
    (warnings only) when strict validation disabled.

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
    if strict_validation:
        validate_grammar_u4b(G, node, logger)

    # 4. History length validation
    _validate_history_length(G, node)


def _validate_minimum_vf(G: TNFRGraph, node: NodeId) -> None:
    """Validate minimum structural frequency for phase transformation."""
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    min_vf = float(G.graph.get("ZHIR_MIN_VF", 0.05))

    if vf < min_vf:
        raise OperatorPreconditionError(
            "Mutation",
            f"Structural frequency too low for mutation (νf={vf:.3f} < {min_vf:.3f})",
        )


def _validate_history_length(G: TNFRGraph, node: NodeId) -> None:
    """Validate sufficient EPI history for velocity calculation."""
    epi_history = G.nodes[node].get("epi_history") or G.nodes[node].get("_epi_history", [])
    min_length = int(G.graph.get("ZHIR_MIN_HISTORY_LENGTH", 2))

    if len(epi_history) < min_length:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Node {node}: ZHIR applied without sufficient EPI history "
            f"(need ≥{min_length} points, have {len(epi_history)}). "
            f"Threshold verification may be inaccurate."
        )


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

    Notes
    -----
    - If ∂EPI/∂t < ξ: Logs warning (soft check for backward compatibility)
    - If ∂EPI/∂t ≥ ξ: Logs success, sets validation flag
    - If insufficient history: Logs warning, cannot verify

    The check is soft (warning only) unless ZHIR_STRICT_THRESHOLD_CHECK=True,
    maintaining backward compatibility with existing code.

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

    # Get EPI history - check both keys for compatibility
    epi_history = G.nodes[node].get("epi_history") or G.nodes[node].get("_epi_history", [])

    if len(epi_history) < 2:
        # Insufficient history - cannot verify threshold
        logger.warning(
            f"Node {node}: ZHIR applied without sufficient EPI history "
            f"(need ≥2 points, have {len(epi_history)}). Cannot verify threshold."
        )
        G.nodes[node]["_zhir_threshold_unknown"] = True
        return

    # Compute ∂EPI/∂t (discrete approximation using last two points)
    # For discrete operator applications with Δt=1: ∂EPI/∂t ≈ EPI_t - EPI_{t-1}
    depi_dt = abs(epi_history[-1] - epi_history[-2])

    # Get threshold from configuration
    xi_threshold = float(G.graph.get("ZHIR_THRESHOLD_XI", 0.1))

    # Verify threshold crossed
    if depi_dt < xi_threshold:
        # Allow mutation but log warning (soft check for backward compatibility)
        logger.warning(
            f"Node {node}: ZHIR applied with ∂EPI/∂t={depi_dt:.3f} < ξ={xi_threshold}. "
            f"Mutation may lack structural justification. "
            f"Consider increasing dissonance (OZ) first."
        )
        G.nodes[node]["_zhir_threshold_warning"] = True

        # Strict check if configured
        if bool(G.graph.get("ZHIR_STRICT_THRESHOLD_CHECK", False)):
            raise OperatorPreconditionError(
                "Mutation",
                f"Threshold not crossed: ∂EPI/∂t={depi_dt:.3f} < ξ={xi_threshold}. "
                f"Apply Dissonance (OZ) or Expansion (VAL) to increase structural velocity first.",
            )
    else:
        # Threshold met - log success
        logger.info(
            f"Node {node}: ZHIR threshold crossed (∂EPI/∂t={depi_dt:.3f} > ξ={xi_threshold})"
        )
        G.nodes[node]["_zhir_threshold_met"] = True


def validate_grammar_u4b(G: TNFRGraph, node: NodeId, logger: logging.Logger | None = None) -> None:
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
    if not glyph_history:
        # No history - cannot validate U4b
        logger.warning(f"Node {node}: No glyph history available. Cannot verify U4b compliance.")
        return

    # Import glyph_function_name to convert glyphs to operator names
    from ..grammar import glyph_function_name

    # Convert history to operator names
    history_names = [glyph_function_name(g) for g in glyph_history]

    # Part 1: Check for prior IL (Coherence)
    require_il = bool(G.graph.get("ZHIR_REQUIRE_IL_PRECEDENCE", False))
    il_found = "coherence" in history_names

    if require_il and not il_found:
        raise OperatorPreconditionError(
            "Mutation",
            "U4b violation: ZHIR requires prior IL (Coherence) for stable transformation base. "
            "Apply Coherence before mutation sequence. "
            f"Recent history: {history_names[-5:] if len(history_names) > 5 else history_names}",
        )

    if il_found:
        logger.debug(f"Node {node}: ZHIR IL precedence satisfied (prior Coherence found)")

    # Part 2: Check for recent destabilizer
    # This also records destabilizer context for telemetry
    context = record_destabilizer_context(G, node, logger)

    require_destabilizer = bool(G.graph.get("ZHIR_REQUIRE_DESTABILIZER", False))
    destabilizer_found = context.get("destabilizer_operator")

    if require_destabilizer and destabilizer_found is None:
        recent_history = context.get("recent_history", [])
        raise OperatorPreconditionError(
            "Mutation",
            "U4b violation: ZHIR requires recent destabilizer (OZ/VAL/etc) within ~3 ops. "
            f"Recent history: {recent_history}. "
            "Apply Dissonance or Expansion to elevate ΔNFR first.",
        )


def record_destabilizer_context(
    G: TNFRGraph, node: NodeId, logger: logging.Logger | None = None
) -> dict:
    """Detect and record which destabilizer enabled the current mutation.

    This implements R4 Extended telemetry by analyzing the glyph_history
    to determine which destabilizer type (strong/moderate/weak) is within
    its appropriate bifurcation window.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node being mutated
    logger : logging.Logger, optional
        Logger for telemetry output

    Returns
    -------
    dict
        Destabilizer context with keys:
        - destabilizer_type: "strong"/"moderate"/"weak"/None
        - destabilizer_operator: Name of destabilizer glyph
        - destabilizer_distance: Operations since destabilizer
        - recent_history: Last N operator names

    Notes
    -----
    The destabilizer context is stored in node['_mutation_context'] for
    structural tracing and post-hoc analysis. This enables understanding
    of bifurcation pathways without breaking TNFR structural invariants.

    **Bifurcation Windows (from BIFURCATION_WINDOWS)**:
    - Strong destabilizers (OZ, VAL): window = 4 operations
    - Moderate destabilizers: window = 2 operations
    - Weak destabilizers: window = 1 operation (immediate only)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators import Dissonance
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> Dissonance()(G, node)  # Apply OZ (strong destabilizer)
    >>> context = record_destabilizer_context(G, node)
    >>> context["destabilizer_type"]  # doctest: +SKIP
    'strong'
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
            "destabilizer_type": None,
            "destabilizer_operator": None,
            "destabilizer_distance": None,
            "recent_history": [],
        }
        G.nodes[node]["_mutation_context"] = context
        return context

    # Import glyph_function_name to convert glyphs to operator names
    from ..grammar import glyph_function_name

    # Get recent history (up to max window size)
    max_window = BIFURCATION_WINDOWS["strong"]
    recent = list(history)[-max_window:] if len(history) > max_window else list(history)
    recent_names = [glyph_function_name(g) for g in recent]

    # Search backwards for destabilizers, checking window constraints
    destabilizer_found = None
    destabilizer_type = None
    destabilizer_distance = None

    for i, op_name in enumerate(reversed(recent_names)):
        distance = i + 1  # Distance from mutation (1 = immediate predecessor)

        # Check strong destabilizers (window = 4)
        if op_name in DESTABILIZERS_STRONG and distance <= BIFURCATION_WINDOWS["strong"]:
            destabilizer_found = op_name
            destabilizer_type = "strong"
            destabilizer_distance = distance
            break

        # Check moderate destabilizers (window = 2)
        if op_name in DESTABILIZERS_MODERATE and distance <= BIFURCATION_WINDOWS["moderate"]:
            destabilizer_found = op_name
            destabilizer_type = "moderate"
            destabilizer_distance = distance
            break

        # Check weak destabilizers (window = 1, immediate only)
        if op_name in DESTABILIZERS_WEAK and distance == 1:
            destabilizer_found = op_name
            destabilizer_type = "weak"
            destabilizer_distance = distance
            break

    # Store context in node metadata for telemetry
    context = {
        "destabilizer_type": destabilizer_type,
        "destabilizer_operator": destabilizer_found,
        "destabilizer_distance": destabilizer_distance,
        "recent_history": recent_names,
    }
    G.nodes[node]["_mutation_context"] = context

    # Log telemetry for structural tracing
    if destabilizer_found:
        logger.info(
            f"Node {node}: ZHIR enabled by {destabilizer_type} destabilizer "
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
                "recent_destabilizer": {"passed": bool, "type": str|None, "distance": int|None},
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
    min_vf = float(G.graph.get("ZHIR_MIN_VF", 0.05))
    vf_passed = vf >= min_vf
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

    # Check 2: Threshold crossing
    epi_history = G.nodes[node].get("epi_history") or G.nodes[node].get("_epi_history", [])
    xi_threshold = float(G.graph.get("ZHIR_THRESHOLD_XI", 0.1))

    if len(epi_history) >= 2:
        depi_dt = abs(epi_history[-1] - epi_history[-2])
        threshold_passed = depi_dt >= xi_threshold
        checks["threshold_crossing"] = {
            "passed": threshold_passed,
            "depi_dt": depi_dt,
            "xi": xi_threshold,
        }
        if not threshold_passed:
            recommendations.append(
                f"Increase structural velocity: ∂EPI/∂t={depi_dt:.3f} < ξ={xi_threshold}. "
                f"Apply OZ (Dissonance) or VAL (Expansion) to elevate reorganization."
            )
    else:
        checks["threshold_crossing"] = {
            "passed": False,
            "depi_dt": None,
            "xi": xi_threshold,
            "reason": "Insufficient history",
        }
        recommendations.append(
            f"Build EPI history: only {len(epi_history)} points available (need ≥2). "
            f"Apply several operators to establish history."
        )

    # Check 3: IL precedence
    glyph_history = G.nodes[node].get("glyph_history", [])
    if glyph_history:
        from ..grammar import glyph_function_name

        history_names = [glyph_function_name(g) for g in glyph_history]
        il_found = "coherence" in history_names
    else:
        il_found = False
        history_names = []

    checks["il_precedence"] = {
        "passed": il_found,
        "found": il_found,
    }
    if not il_found:
        recommendations.append("Apply IL (Coherence) for stable transformation base (U4b Part 1).")

    # Check 4: Recent destabilizer
    logger = logging.getLogger(__name__)
    context = record_destabilizer_context(G, node, logger)
    destabilizer_found = context.get("destabilizer_operator") is not None

    checks["recent_destabilizer"] = {
        "passed": destabilizer_found,
        "type": context.get("destabilizer_type"),
        "distance": context.get("destabilizer_distance"),
        "operator": context.get("destabilizer_operator"),
    }
    if not destabilizer_found:
        recommendations.append(
            "Apply destabilizer (OZ/VAL) within last ~3 operations to elevate ΔNFR (U4b Part 2)."
        )

    # Check 5: History length
    min_history = int(G.graph.get("ZHIR_MIN_HISTORY_LENGTH", 2))
    history_passed = len(epi_history) >= min_history
    checks["history_length"] = {
        "passed": history_passed,
        "length": len(epi_history),
        "required": min_history,
    }
    # Already covered by threshold check recommendations

    # Overall readiness
    all_passed = all(check.get("passed", False) for check in checks.values())

    return {
        "ready": all_passed,
        "checks": checks,
        "recommendations": recommendations,
    }
