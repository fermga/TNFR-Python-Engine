"""Tests for :mod:`tnfr.validation.compatibility`."""

from tnfr.config.operator_names import CONTRACTION, MUTATION, RESONANCE, SILENCE
from tnfr.types import Glyph
from tnfr.validation.compatibility import CANON_COMPAT, CANON_FALLBACK
from tnfr.validation import glyph_function_name

def test_thol_maintains_closure_paths():
    """THOL blocks must always offer closure glyphs to preserve phase."""

    thol_transitions = {glyph_function_name(g) for g in CANON_COMPAT[Glyph.THOL]}
    assert SILENCE in thol_transitions
    assert CONTRACTION in thol_transitions

def test_fallbacks_reinforce_mutation_flow():
    """Fallbacks keep ΔNFR-driven mutation aligned with canon."""

    assert glyph_function_name(CANON_FALLBACK[Glyph.OZ]) == MUTATION
    assert glyph_function_name(CANON_FALLBACK[Glyph.NAV]) == RESONANCE
