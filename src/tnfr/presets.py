"""Configuraciones predefinidas."""
from __future__ import annotations
from .program import seq, block, wait, ejemplo_canonico_basico
from .types import Glyph


_PRESETS = {
    "arranque_resonante": seq(Glyph.AL, Glyph.EN, Glyph.IL, Glyph.RA, Glyph.VAL, Glyph.UM, wait(3), Glyph.SHA),
    "mutacion_contenida": seq(Glyph.AL, Glyph.EN, block(Glyph.OZ, Glyph.ZHIR, Glyph.IL, repeat=2), Glyph.RA, Glyph.SHA),
    "exploracion_acople": seq(
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.VAL,
        Glyph.UM,
        block(Glyph.OZ, Glyph.NAV, Glyph.IL, repeat=1),
        Glyph.RA,
        Glyph.SHA,
    ),
    "ejemplo_canonico": ejemplo_canonico_basico(),
    # Topologías fractales: expansión/contracción modular
    "fractal_expand": seq(block(Glyph.THOL, Glyph.VAL, Glyph.UM, repeat=2, close=Glyph.NUL), Glyph.RA),
    "fractal_contract": seq(block(Glyph.THOL, Glyph.NUL, Glyph.UM, repeat=2, close=Glyph.SHA), Glyph.RA),
}


def get_preset(name: str):
    if name not in _PRESETS:
        raise KeyError(f"Preset no encontrado: {name}")
    return _PRESETS[name]
