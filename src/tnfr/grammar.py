"""Backwards compatibility layer for grammar helpers.

The canonical implementations now live in :mod:`tnfr.validation`.  This
module only re-exports them to avoid breaking external callers until the
new import paths are fully adopted.
"""

from .validation.compatibility import CANON_COMPAT, CANON_FALLBACK
from .validation.grammar import (
    GrammarContext,
    _gram_state,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
)

__all__ = [
    "GrammarContext",
    "CANON_COMPAT",
    "CANON_FALLBACK",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "apply_glyph_with_grammar",
    "_gram_state",
]
