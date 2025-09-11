"""Default glyphs."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from .types import Glyph

# -------------------------
# Orden canónico y clasificaciones funcionales
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
GLYPH_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "estabilizadores": ESTABILIZADORES,
        "disruptivos": DISRUPTIVOS,
        # Grupos auxiliares para métricas morfosintácticas
        "ID": (Glyph.OZ.value,),
        "CM": (Glyph.ZHIR.value, Glyph.NAV.value),
        "NE": (Glyph.IL.value, Glyph.THOL.value),
        "PP_num": (Glyph.SHA.value,),
        "PP_den": (Glyph.REMESH.value,),
    }
)

# -------------------------
# Mapa de ángulos glíficos
# -------------------------

# Ángulos canónicos para todos los glyphs reconocidos. Se calculan a partir
# del orden canónico y reglas de orientación para las categorías
# "estabilizadores" y "disruptivos".


def _build_angle_map() -> dict[str, float]:
    """Construir el mapa de ángulos en el plano σ."""
    step = 2 * math.pi / len(GLYPHS_CANONICAL)
    canonical = {g: i * step for i, g in enumerate(GLYPHS_CANONICAL)}
    angles = dict(canonical)

    # Reglas específicas de orientación
    for idx, g in enumerate(ESTABILIZADORES):
        angles[g] = idx * math.pi / 4
    for idx, g in enumerate(DISRUPTIVOS):
        angles[g] = math.pi + idx * math.pi / 4

    # Excepciones manuales
    angles[Glyph.VAL.value] = canonical[Glyph.RA.value]
    angles[Glyph.NUL.value] = canonical[Glyph.ZHIR.value]
    angles[Glyph.AL.value] = 0.0
    return angles


ANGLE_MAP: Mapping[str, float] = MappingProxyType(_build_angle_map())

__all__ = (
    "GLYPHS_CANONICAL",
    "GLYPHS_CANONICAL_SET",
    "ESTABILIZADORES",
    "DISRUPTIVOS",
    "GLYPH_GROUPS",
    "ANGLE_MAP",
)
