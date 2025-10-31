"""Compatibility layer exposing canonical grammar helpers from :mod:`tnfr.operators`."""

from __future__ import annotations

from ..operators.grammar import (
    GrammarContext,
    SequenceSyntaxError,
    SequenceValidationResult,
    _gram_state,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    parse_sequence,
    validate_sequence,
)

__all__ = [
    "GrammarContext",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "_gram_state",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "parse_sequence",
    "validate_sequence",
]
