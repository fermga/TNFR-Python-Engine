from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar

from ..operators.grammar import (
    GrammarContext,
    MutationPreconditionError,
    RepeatWindowError,
    SequenceSyntaxError,
    SequenceValidationResult,
    StructuralGrammarError,
    TholClosureError,
    TransitionCompatibilityError,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    record_grammar_violation,
    validate_sequence,
)
from ..types import Glyph, TNFRGraph
from .compatibility import CANON_COMPAT as CANON_COMPAT
from .compatibility import CANON_FALLBACK as CANON_FALLBACK
from .graph import GRAPH_VALIDATORS, run_validators
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon
from .soft_filters import (
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
from .spectral import NFRValidator
from .window import validate_window

SubjectT = TypeVar("SubjectT")

class ValidationOutcome(Generic[SubjectT]):
    subject: SubjectT
    passed: bool
    summary: Mapping[str, Any]
    artifacts: Mapping[str, Any] | None

class Validator(Protocol[SubjectT]):
    def validate(
        self, subject: SubjectT, /, **kwargs: Any
    ) -> ValidationOutcome[SubjectT]: ...
    def report(self, outcome: ValidationOutcome[SubjectT]) -> str: ...

__all__ = (
    "validate_sequence",
    "GrammarContext",
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "record_grammar_violation",
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
)
