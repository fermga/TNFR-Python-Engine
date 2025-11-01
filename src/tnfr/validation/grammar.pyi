from ..operators.grammar import (
    GrammarContext,
    StructuralGrammarError,
    RepeatWindowError,
    MutationPreconditionError,
    TholClosureError,
    TransitionCompatibilityError,
    SequenceSyntaxError,
    SequenceValidationResult,
    _gram_state,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    parse_sequence,
    validate_sequence,
)
from ..types import NodeId, TNFRGraph

__all__ = (
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
)


def record_grammar_violation(
    G: TNFRGraph, node: NodeId, error: StructuralGrammarError, *, stage: str
) -> None: ...
