"""Canonical glyph constants tied to configuration presets."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from ..types import Glyph

# -------------------------
# Canonical order and functional classifications
# -------------------------

GLYPHS_CANONICAL: tuple[str, ...] = (
    Glyph.AL.value,  # 0
    Glyph.EN.value,  # 1
    Glyph.IL.value,  # 2
    Glyph.OZ.value,  # 3
    Glyph.UM.value,  # 4
    Glyph.RA.value,  # 5
    Glyph.SHA.value,  # 6
    Glyph.VAL.value,  # 7
    Glyph.NUL.value,  # 8
    Glyph.THOL.value,  # 9
    Glyph.ZHIR.value,  # 10
    Glyph.NAV.value,  # 11
    Glyph.REMESH.value,  # 12
)

GLYPHS_CANONICAL_SET: frozenset[str] = frozenset(GLYPHS_CANONICAL)

STABILIZERS: tuple[str, ...] = (
    Glyph.IL.value,
    Glyph.RA.value,
    Glyph.UM.value,
    Glyph.SHA.value,
)

DISRUPTORS: tuple[str, ...] = (
    Glyph.OZ.value,
    Glyph.ZHIR.value,
    Glyph.NAV.value,
    Glyph.THOL.value,
)

# General map of glyph groupings for cross-reference.
#
# Spanish keys (``estabilizadores`` / ``disruptivos``) were removed in TNFR 7.0
# to keep the public surface English-only. Code that still referenced those
# identifiers must switch to the canonical ``stabilizers`` / ``disruptors``
# entries or maintain a private compatibility layer.
GLYPH_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "stabilizers": STABILIZERS,
        "disruptors": DISRUPTORS,
        # Auxiliary groups for morphosyntactic metrics
        "ID": (Glyph.OZ.value,),
        "CM": (Glyph.ZHIR.value, Glyph.NAV.value),
        "NE": (Glyph.IL.value, Glyph.THOL.value),
        "PP_num": (Glyph.SHA.value,),
        "PP_den": (Glyph.REMESH.value,),
    }
)

# -------------------------
# Glyph angle map
# -------------------------

# Canonical angles for all recognised glyphs. They are computed from the
# canonical order and orientation rules for the "stabilizers" and
# "disruptors" categories.


def _build_angle_map() -> dict[str, float]:
    """Build the angle map in the σ-plane."""

    step = 2 * math.pi / len(GLYPHS_CANONICAL)
    canonical = {g: i * step for i, g in enumerate(GLYPHS_CANONICAL)}
    angles = dict(canonical)

    # Orientation rules
    for idx, g in enumerate(STABILIZERS):
        angles[g] = idx * math.pi / 4
    for idx, g in enumerate(DISRUPTORS):
        angles[g] = math.pi + idx * math.pi / 4

    # Manual exceptions
    angles[Glyph.VAL.value] = canonical[Glyph.RA.value]
    angles[Glyph.NUL.value] = canonical[Glyph.ZHIR.value]
    angles[Glyph.AL.value] = 0.0
    return angles


ANGLE_MAP: Mapping[str, float] = MappingProxyType(_build_angle_map())

__all__ = (
    "GLYPHS_CANONICAL",
    "GLYPHS_CANONICAL_SET",
    "STABILIZERS",
    "DISRUPTORS",
    "GLYPH_GROUPS",
    "ANGLE_MAP",
)
