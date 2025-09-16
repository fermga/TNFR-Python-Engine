"""Predefined configurations."""

from __future__ import annotations
from .execution import seq, block, wait, basic_canonical_example
from .types import Glyph

# Secuencias comunes
ARRANQUE_BASICO = seq(Glyph.AL, Glyph.EN)
CIERRE_BASICO = seq(Glyph.RA, Glyph.SHA)
NUCLEO_VAL_UM = seq(Glyph.VAL, Glyph.UM)

__all__ = (
    "ARRANQUE_BASICO",
    "CIERRE_BASICO",
    "NUCLEO_VAL_UM",
    "get_preset",
)


_PRESETS = {
    "arranque_resonante": ARRANQUE_BASICO
    + seq(Glyph.IL, Glyph.RA)
    + NUCLEO_VAL_UM
    + seq(wait(3), Glyph.SHA),
    "mutacion_contenida": ARRANQUE_BASICO
    + seq(block(Glyph.OZ, Glyph.ZHIR, Glyph.IL, repeat=2))
    + CIERRE_BASICO,
    "exploracion_acople": ARRANQUE_BASICO
    + seq(Glyph.IL)
    + NUCLEO_VAL_UM
    + seq(block(Glyph.OZ, Glyph.NAV, Glyph.IL, repeat=1))
    + CIERRE_BASICO,
    "ejemplo_canonico": basic_canonical_example(),
    # Topologías fractales: expansión/contracción modular
    "fractal_expand": seq(
        block(Glyph.THOL, Glyph.VAL, Glyph.UM, repeat=2, close=Glyph.NUL),
        Glyph.RA,
    ),
    "fractal_contract": seq(
        block(Glyph.THOL, Glyph.NUL, Glyph.UM, repeat=2, close=Glyph.SHA),
        Glyph.RA,
    ),
}


def get_preset(name: str):
    if name not in _PRESETS:
        raise KeyError(f"Preset no encontrado: {name}")
    return _PRESETS[name]
