"""Compatibility layer exposing canonical grammar helpers from :mod:`tnfr.operators`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..operators.grammar import (
    GrammarContext,
    StructuralGrammarError,
    RepeatWindowError,
    MutationPreconditionError,
    TholClosureError,
    TransitionCompatibilityError,
    SequenceSyntaxError,
    SequenceValidationResult,
    _record_grammar_violation,
    _gram_state,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    parse_sequence,
    validate_sequence,
)

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "GrammarContext",
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "record_grammar_violation",
    "_gram_state",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "parse_sequence",
    "validate_sequence",
]


def record_grammar_violation(
    G: "TNFRGraph", node: "NodeId", error: StructuralGrammarError, *, stage: str
) -> None:
    """Public wrapper around :func:`_record_grammar_violation` preserving telemetry hooks."""

    _record_grammar_violation(G, node, error, stage=stage)
