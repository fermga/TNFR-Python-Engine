"""Canonical TNFR compatibility tables.

This module centralises the canonical transitions that keep TNFR
coherence consistent across refactors.  It is intentionally lightweight
so :mod:`tnfr.validation.grammar` can depend on it without introducing
cycles.
"""

from __future__ import annotations

from ..types import Glyph

__all__ = ["CANON_COMPAT", "CANON_FALLBACK"]

# Canonical compatibilities (allowed next glyphs)
CANON_COMPAT: dict[Glyph, set[Glyph]] = {
    # Opening / initiation
    Glyph.AL: {Glyph.EN, Glyph.RA, Glyph.NAV, Glyph.VAL, Glyph.UM},
    Glyph.EN: {Glyph.IL, Glyph.UM, Glyph.RA, Glyph.NAV},
    # Stabilisation / diffusion / coupling
    Glyph.IL: {Glyph.RA, Glyph.VAL, Glyph.UM, Glyph.SHA},
    Glyph.UM: {Glyph.RA, Glyph.IL, Glyph.VAL, Glyph.NAV},
    Glyph.RA: {Glyph.IL, Glyph.VAL, Glyph.UM, Glyph.NAV},
    Glyph.VAL: {Glyph.UM, Glyph.RA, Glyph.IL, Glyph.NAV},
    # Dissonance → transition → mutation
    Glyph.OZ: {Glyph.ZHIR, Glyph.NAV},
    Glyph.ZHIR: {Glyph.IL, Glyph.NAV},
    Glyph.NAV: {Glyph.OZ, Glyph.ZHIR, Glyph.RA, Glyph.IL, Glyph.UM},
    # Closures / latent states
    Glyph.SHA: {Glyph.AL, Glyph.EN},
    Glyph.NUL: {Glyph.AL, Glyph.IL},
    # Self-organising blocks
    Glyph.THOL: {
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.NAV,
        Glyph.RA,
        Glyph.IL,
        Glyph.UM,
        Glyph.SHA,
        Glyph.NUL,
    },
}

# Canonical fallbacks when a transition is not allowed
CANON_FALLBACK: dict[Glyph, Glyph] = {
    Glyph.AL: Glyph.EN,
    Glyph.EN: Glyph.IL,
    Glyph.IL: Glyph.RA,
    Glyph.NAV: Glyph.RA,
    Glyph.NUL: Glyph.AL,
    Glyph.OZ: Glyph.ZHIR,
    Glyph.RA: Glyph.IL,
    Glyph.SHA: Glyph.AL,
    Glyph.THOL: Glyph.NAV,
    Glyph.UM: Glyph.RA,
    Glyph.VAL: Glyph.RA,
    Glyph.ZHIR: Glyph.IL,
}
