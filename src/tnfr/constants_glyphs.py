"""Glyphs predeterminados."""
from __future__ import annotations

import math
from typing import Dict

from .types import Glyph

# -------------------------
# Orden canónico y clasificaciones funcionales
# -------------------------

GLYPHS_CANONICAL: tuple[str, ...] = (
    Glyph.AL.value,   # 0
    Glyph.EN.value,   # 1
    Glyph.IL.value,   # 2
    Glyph.OZ.value,   # 3
    Glyph.UM.value,   # 4
    Glyph.RA.value,   # 5
    Glyph.SHA.value,  # 6
    Glyph.VAL.value,  # 7
    Glyph.NUL.value,  # 8
    Glyph.THOL.value, # 9
    Glyph.ZHIR.value, # 10
    Glyph.NAV.value,  # 11
    Glyph.REMESH.value,  # 12
)

ESTABILIZADORES = (
    Glyph.IL.value,
    Glyph.RA.value,
    Glyph.UM.value,
    Glyph.SHA.value,
)

DISRUPTIVOS = (
    Glyph.OZ.value,
    Glyph.ZHIR.value,
    Glyph.NAV.value,
    Glyph.THOL.value,
)

# Mapa general de agrupaciones glíficas para referencia cruzada.
GLYPH_GROUPS = {
    "estabilizadores": ESTABILIZADORES,
    "disruptivos": DISRUPTIVOS,
    # Grupos auxiliares para métricas morfosintácticas
    "ID": (Glyph.OZ.value,),
    "CM": (Glyph.ZHIR.value, Glyph.NAV.value),
    "NE": (Glyph.IL.value, Glyph.THOL.value),
    "PP_num": (Glyph.SHA.value,),
    "PP_den": (Glyph.REMESH.value,),
}

# -------------------------
# Mapa de ángulos glíficos
# -------------------------

# Ángulos canónicos para todos los glyphs reconocidos. Las categorías
# anteriores se distribuyen uniformemente en el círculo y se ajustan a
# orientaciones semánticas específicas en el plano σ.
ANGLE_MAP: Dict[str, float] = {
    # AL no participa en el plano σ pero se incluye por completitud.
    # Comparte el ángulo base (0 rad) con IL de forma intencionada.
    Glyph.AL.value: 0.0,
    Glyph.EN.value: 2 * math.pi / 13,
    Glyph.IL.value: 0.0,
    Glyph.UM.value: math.pi / 2,
    Glyph.RA.value: math.pi / 4,
    Glyph.VAL.value: 10 * math.pi / 13,
    Glyph.OZ.value: math.pi,
    Glyph.ZHIR.value: 5 * math.pi / 4,
    Glyph.NAV.value: 3 * math.pi / 2,
    Glyph.THOL.value: 7 * math.pi / 4,
    Glyph.NUL.value: 20 * math.pi / 13,
    Glyph.SHA.value: 3 * math.pi / 4,
    Glyph.REMESH.value: 24 * math.pi / 13,
}

__all__ = [
    "GLYPHS_CANONICAL",
    "ESTABILIZADORES",
    "DISRUPTIVOS",
    "GLYPH_GROUPS",
    "ANGLE_MAP",
]
