"""
constants_glifos.py — categorización y ángulos de glifos.

Centraliza constantes relacionadas con las familias de glifos y el plano
angular utilizado por los observadores de sentido.
"""
from __future__ import annotations

import math
from typing import Dict, List

from .types import Glyph

# -------------------------
# Clasificaciones funcionales de glifos
# -------------------------

ESTABILIZADORES: List[str] = [
    Glyph.IL.value,
    Glyph.RA.value,
    Glyph.UM.value,
    Glyph.SHA.value,
]

DISRUPTIVOS: List[str] = [
    Glyph.OZ.value,
    Glyph.ZHIR.value,
    Glyph.NAV.value,
    Glyph.THOL.value,
]

# -------------------------
# Mapa de ángulos glíficos
# -------------------------

# Ángulos canónicos para todos los glifos reconocidos. Las categorías
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
