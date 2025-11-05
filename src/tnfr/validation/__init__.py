"""Unified validation interface consolidating grammar, graph and spectral checks.

This package re-exports the canonical grammar helpers and runtime validators so
downstream code can rely on ``tnfr.validation`` as the single import path for
structural validation primitives.
"""

from __future__ import annotations

from ..compat.dataclass import dataclass
from typing import Any, Generic, Mapping, Protocol, TypeVar, runtime_checkable

SubjectT = TypeVar("SubjectT")

@dataclass(slots=True)
class ValidationOutcome(Generic[SubjectT]):
    """Result emitted by all canonical TNFR validators."""

    subject: SubjectT
    """The validated subject in canonical form."""

    passed: bool
    """Whether the validation succeeded without invariant violations."""

    summary: Mapping[str, Any]
    """Structured diagnostics describing the performed checks."""

    artifacts: Mapping[str, Any] | None = None
    """Optional artefacts (e.g. clamped nodes, normalised vectors)."""

@runtime_checkable
class Validator(Protocol[SubjectT]):
    """Contract implemented by runtime and spectral validators."""

    def validate(self, subject: SubjectT, /, **kwargs: Any) -> ValidationOutcome[SubjectT]:
        """Validate ``subject`` returning a :class:`ValidationOutcome`."""

    def report(self, outcome: "ValidationOutcome[SubjectT]") -> str:
        """Produce a concise textual explanation for ``outcome``."""

from .compatibility import CANON_COMPAT, CANON_FALLBACK  # noqa: F401
from ..operators import grammar as _grammar
from ..types import Glyph
from .graph import GRAPH_VALIDATORS, run_validators  # noqa: F401
from .window import validate_window  # noqa: F401
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon  # noqa: F401
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr  # noqa: F401
from .soft_filters import (  # noqa: F401
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
from .input_validation import (  # noqa: F401
    ValidationError,
    validate_epi_value,
    validate_vf_value,
    validate_theta_value,
    validate_dnfr_value,
    validate_node_id,
    validate_glyph,
    validate_tnfr_graph,
    validate_glyph_factors,
    validate_operator_parameters,
)
from .invariants import (  # noqa: F401
    InvariantSeverity,
    InvariantViolation,
    TNFRInvariant,
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    Invariant5_ExplicitPhaseChecks,
)
from .validator import (  # noqa: F401
    TNFRValidator,
    TNFRValidationError,
)
from .sequence_validator import (  # noqa: F401
    SequenceSemanticValidator,
)
from .config import (  # noqa: F401
    ValidationConfig,
    validation_config,
    configure_validation,
)
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
    "Invariant5_ExplicitPhaseChecks",
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
