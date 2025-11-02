"""Deprecated compatibility wrapper for :mod:`tnfr.operators.grammar`.

Import :mod:`tnfr.operators.grammar` directly to access the canonical grammar
interfaces. This module remains as a thin shim for historical callers and will
be removed once downstream packages adopt the canonical entry point.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ..operators.grammar import (
    GrammarContext,
    MutationPreconditionError,
    RepeatWindowError,
    record_grammar_violation as _canonical_record_violation,
    SequenceSyntaxError,
    SequenceValidationResult,
    StructuralGrammarError,
    TholClosureError,
    TransitionCompatibilityError,
    _gram_state,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    parse_sequence,
    validate_sequence,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
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

warnings.warn(
    "'tnfr.validation.grammar' is deprecated; import from 'tnfr.operators.grammar' "
    "for the canonical grammar interface.",
    DeprecationWarning,
    stacklevel=2,
)


def record_grammar_violation(
    G: "TNFRGraph", node: "NodeId", error: StructuralGrammarError, *, stage: str
) -> None:
    """Bridge to :func:`tnfr.operators.grammar.record_grammar_violation`."""

    warnings.warn(
        "'tnfr.validation.grammar.record_grammar_violation' is deprecated; "
        "use 'tnfr.operators.grammar.record_grammar_violation' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _canonical_record_violation(G, node, error, stage=stage)
