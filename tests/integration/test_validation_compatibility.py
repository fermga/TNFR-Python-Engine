"""Tests for :mod:`tnfr.validation.compatibility`."""

from tnfr.types import Glyph
from tnfr.validation.compatibility import CANON_COMPAT, CANON_FALLBACK


def test_thol_maintains_closure_paths():
    """THOL blocks must always offer closure glyphs to preserve phase."""

    thol_transitions = CANON_COMPAT[Glyph.THOL]
    assert Glyph.SHA in thol_transitions
    assert Glyph.NUL in thol_transitions


def test_fallbacks_reinforce_mutation_flow():
    """Fallbacks keep ΔNFR-driven mutation aligned with canon."""

    assert CANON_FALLBACK[Glyph.OZ] == Glyph.ZHIR
    assert CANON_FALLBACK[Glyph.NAV] == Glyph.RA
