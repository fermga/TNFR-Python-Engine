"""Unified validation interface consolidating grammar, graph and spectral checks."""

from __future__ import annotations

from dataclasses import dataclass
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


from .compatibility import CANON_COMPAT, CANON_FALLBACK
from .grammar import (
    GrammarContext,
    MutationPreconditionError,
    RepeatWindowError,
    SequenceValidationResult,
    StructuralGrammarError,
    TholClosureError,
    TransitionCompatibilityError,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
)
from .graph import GRAPH_VALIDATORS, run_validators
from .window import validate_window
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr
from .soft_filters import (
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
from .syntax import validate_sequence

__all__ = (
    "ValidationOutcome",
    "Validator",
    "validate_sequence",
    "GrammarContext",
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceValidationResult",
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


def __getattr__(name: str) -> Any:
    if name == "NFRValidator":
        from .spectral import NFRValidator as _NFRValidator

        return _NFRValidator
    raise AttributeError(name)

