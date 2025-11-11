"""Unified validation interface consolidating grammar, graph and spectral checks.

RECOMMENDED: Use TNFRValidator for unified validation pipeline
==============================================================

The TNFRValidator class provides a single entry point for all TNFR validation
operations, consolidating input validation, graph validation, invariant checking,
operator preconditions, and runtime validation into one coherent API.

Example Usage::

    from tnfr.validation import TNFRValidator

    validator = TNFRValidator()

    # Comprehensive validation in one call
    result = validator.validate(
        graph=G,
        epi=0.5,
        vf=1.0,
        include_invariants=True,
    )

    if not result['passed']:
        print(f"Validation failed: {result['errors']}")

For detailed migration guide, see UNIFIED_VALIDATION_PIPELINE.md

Legacy API
==========

This package also re-exports individual validation functions for backward
compatibility, but these may be deprecated in future versions. New code should
use TNFRValidator instead.
"""

from __future__ import annotations

from typing import Any

from .base import SubjectT, ValidationOutcome, Validator  # noqa: F401
from ..operators import grammar as _grammar
from ..types import Glyph
from .config import ValidationConfig, configure_validation, validation_config  # noqa: F401
from .graph import GRAPH_VALIDATORS, run_validators  # noqa: F401
from .input_validation import (  # noqa: F401
    ValidationError,
    validate_dnfr_value,
    validate_epi_value,
    validate_glyph,
    validate_glyph_factors,
    validate_node_id,
    validate_operator_parameters,
    validate_theta_value,
    validate_tnfr_graph,
    validate_vf_value,
)
from .invariants import (  # noqa: F401
    Invariant10_DomainNeutrality,
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    Invariant3_DNFRSemantics,
    Invariant4_OperatorClosure,
    Invariant5_ExplicitPhaseChecks,
    Invariant6_NodeBirthCollapse,
    Invariant7_OperationalFractality,
    Invariant8_ControlledDeterminism,
    Invariant9_StructuralMetrics,
    InvariantSeverity,
    InvariantViolation,
    TNFRInvariant,
)
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr  # noqa: F401
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon  # noqa: F401
from .sequence_validator import SequenceSemanticValidator  # noqa: F401
from .soft_filters import (  # noqa: F401
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
from .validator import TNFRValidationError, TNFRValidator  # noqa: F401
from .window import validate_window  # noqa: F401


# NOTE: Compatibility module deprecated - grammar emerges from TNFR structural dynamics
# Legacy exports kept for backward compatibility but will be removed in future versions
try:
    from .compatibility import (
        CANON_COMPAT,
        CANON_FALLBACK,
        CompatibilityLevel,
        GRADUATED_COMPATIBILITY,
        get_compatibility_level,
    )

    _COMPAT_AVAILABLE = True
except ImportError:
    # Compatibility module removed - provide stubs for backward compatibility
    _COMPAT_AVAILABLE = False
    CANON_COMPAT = {}
    CANON_FALLBACK = {}

    class CompatibilityLevel:
        EXCELLENT = "excellent"
        GOOD = "good"
        CAUTION = "caution"
        AVOID = "avoid"

    GRADUATED_COMPATIBILITY = {}

    def get_compatibility_level(prev: str, next_op: str) -> str:
        """Deprecated: Use frequency transition validation instead."""
        import warnings

        warnings.warn(
            "get_compatibility_level is deprecated. "
            "Grammar rules now emerge naturally from TNFR structural dynamics. "
            "Use validate_frequency_transition from tnfr.operators.grammar instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "good"


_GRAMMAR_EXPORTS = tuple(getattr(_grammar, "__all__", ()))

globals().update({name: getattr(_grammar, name) for name in _GRAMMAR_EXPORTS})

_RUNTIME_EXPORTS = (
    "ValidationOutcome",
    "Validator",
    "GraphCanonicalValidator",
    "apply_canonical_clamps",
    "validate_canon",
    "GRAPH_VALIDATORS",
    "run_validators",
    "CANON_COMPAT",
    "CANON_FALLBACK",
    "validate_window",
    "coerce_glyph",
    "get_norm",
    "glyph_fallback",
    "normalized_dnfr",
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
    "NFRValidator",
    "ValidationError",
    "validate_epi_value",
    "validate_vf_value",
    "validate_theta_value",
    "validate_dnfr_value",
    "validate_node_id",
    "validate_glyph",
    "validate_tnfr_graph",
    "validate_glyph_factors",
    "validate_operator_parameters",
    "InvariantSeverity",
    "InvariantViolation",
    "TNFRInvariant",
    "Invariant1_EPIOnlyThroughOperators",
    "Invariant2_VfInHzStr",
    "Invariant3_DNFRSemantics",
    "Invariant4_OperatorClosure",
    "Invariant5_ExplicitPhaseChecks",
    "Invariant6_NodeBirthCollapse",
    "Invariant7_OperationalFractality",
    "Invariant8_ControlledDeterminism",
    "Invariant9_StructuralMetrics",
    "Invariant10_DomainNeutrality",
    "TNFRValidator",
    "TNFRValidationError",
    "SequenceSemanticValidator",
    "ValidationConfig",
    "validation_config",
    "configure_validation",
)

__all__ = _GRAMMAR_EXPORTS + _RUNTIME_EXPORTS

_ENFORCE_CANONICAL_GRAMMAR = _grammar.enforce_canonical_grammar


def enforce_canonical_grammar(
    G: Any,
    n: Any,
    cand: Any,
    ctx: Any | None = None,
) -> Any:
    """Proxy to the canonical grammar enforcement helper preserving Glyph outputs."""

    result = _ENFORCE_CANONICAL_GRAMMAR(G, n, cand, ctx)
    if isinstance(cand, Glyph) and not isinstance(result, Glyph):
        translated = _grammar.function_name_to_glyph(result)
        if translated is None and isinstance(result, str):
            try:
                translated = Glyph(result)
            except (TypeError, ValueError):
                translated = None
        if translated is not None:
            return translated
    return result


def __getattr__(name: str) -> Any:
    if name == "NFRValidator":
        from .spectral import NFRValidator as _NFRValidator

        return _NFRValidator
    raise AttributeError(name)
