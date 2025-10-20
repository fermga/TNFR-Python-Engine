"""Canonical glyph constants tied to configuration presets."""

from __future__ import annotations

import math
import warnings
from types import MappingProxyType
from typing import Mapping

from ..types import Glyph

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

# Mapa general de agrupaciones glíficas para referencia cruzada.
#
# Spanish keys (``estabilizadores`` / ``disruptivos``) were removed in TNFR 7.0
# to keep the public surface English-only. Code that still referenced those
# identifiers must switch to the canonical ``stabilizers`` / ``disruptors``
# entries or maintain a private compatibility layer.
GLYPH_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "stabilizers": STABILIZERS,
        "disruptors": DISRUPTORS,
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
    for idx, g in enumerate(STABILIZERS):
        angles[g] = idx * math.pi / 4
    for idx, g in enumerate(DISRUPTORS):
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
    "STABILIZERS",
    "DISRUPTORS",
    "GLYPH_GROUPS",
    "ANGLE_MAP",
)


def __getattr__(name: str) -> object:
    """Provide guidance for removed Spanish glyph aliases."""

    legacy_aliases = {
        "ESTABILIZADORES": "STABILIZERS",
        "DISRUPTIVOS": "DISRUPTORS",
    }
    try:
        replacement = legacy_aliases[name]
    except KeyError as exc:  # pragma: no cover - mirrors default behaviour
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    warnings.warn(
        (
            "Spanish glyph alias '%s' was removed from tnfr.config.constants; "
            "import '%s' instead. This compatibility warning will be removed in TNFR 8.0."
        )
        % (name.lower(), replacement),
        FutureWarning,
        stacklevel=2,
    )
    raise AttributeError(
        f"Spanish glyph alias '{name}' was removed; use '{replacement}' instead."
    )
