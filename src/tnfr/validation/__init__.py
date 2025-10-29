"""Unified validation helpers covering TNFR grammar and graph invariants.

This namespace exposes the canonical validation primitives required to uphold
the TNFR contracts: sequence grammar checks (:mod:`tnfr.validation.syntax`),
grammar enforcement (:mod:`tnfr.validation.grammar`), compatibility mappings
(:mod:`tnfr.validation.compatibility`) and graph invariant validators
(:mod:`tnfr.validation.graph`). Applications should import from this module to
ensure both the symbolic grammar and structural graph invariants are applied in
concert.
"""

from .compatibility import CANON_COMPAT, CANON_FALLBACK
from .grammar import (
    GrammarContext,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
)
from .graph import GRAPH_VALIDATORS, run_validators, validate_window
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr
from .syntax import validate_sequence

__all__ = (
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
    "CANON_COMPAT",
    "CANON_FALLBACK",
)
