from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar

from ..types import Glyph, TNFRGraph
from .compatibility import CANON_COMPAT as CANON_COMPAT, CANON_FALLBACK as CANON_FALLBACK
from .grammar import (
    GrammarContext,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
)
from .graph import GRAPH_VALIDATORS, run_validators, validate_window
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr
from .soft_filters import (acceleration_norm, check_repeats, maybe_force, soft_grammar_filters)
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon
from .spectral import NFRValidator
from .syntax import validate_sequence

SubjectT = TypeVar("SubjectT")


class ValidationOutcome(Generic[SubjectT]):
    subject: SubjectT
    passed: bool
    summary: Mapping[str, Any]
    artifacts: Mapping[str, Any] | None


class Validator(Protocol[SubjectT]):
    def validate(self, subject: SubjectT, /, **kwargs: Any) -> ValidationOutcome[SubjectT]: ...

    def report(self, outcome: ValidationOutcome[SubjectT]) -> str: ...


__all__ = (
    "ValidationOutcome",
    "Validator",
    "validate_sequence",
    "GrammarContext",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "validate_window",
    "run_validators",
    "GRAPH_VALIDATORS",
    "coerce_glyph",
    "glyph_fallback",
    "normalized_dnfr",
    "get_norm",
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
    "CANON_COMPAT",
    "CANON_FALLBACK",
    "GraphCanonicalValidator",
    "apply_canonical_clamps",
    "validate_canon",
    "NFRValidator",
)
