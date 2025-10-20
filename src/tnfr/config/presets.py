"""Predefined TNFR configuration sequences."""

from __future__ import annotations

from ..execution import (
    CANONICAL_PRESET_NAME,
    CANONICAL_PROGRAM_TOKENS,
    block,
    seq,
    wait,
)
from ..types import Glyph, PresetTokens

__all__ = (
    "get_preset",
    "PREFERRED_PRESET_NAMES",
    "LEGACY_PRESET_NAMES",
    "PRESET_NAME_ALIASES",
)


_PRIMARY_PRESETS: dict[str, PresetTokens] = {
    "resonant_bootstrap": seq(
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.RA,
        Glyph.VAL,
        Glyph.UM,
        wait(3),
        Glyph.SHA,
    ),
    "contained_mutation": seq(
        Glyph.AL,
        Glyph.EN,
        block(Glyph.OZ, Glyph.ZHIR, Glyph.IL, repeat=2),
        Glyph.RA,
        Glyph.SHA,
    ),
    "coupling_exploration": seq(
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.VAL,
        Glyph.UM,
        block(Glyph.OZ, Glyph.NAV, Glyph.IL, repeat=1),
        Glyph.RA,
        Glyph.SHA,
    ),
    "fractal_expand": seq(
        block(Glyph.THOL, Glyph.VAL, Glyph.UM, repeat=2, close=Glyph.NUL),
        Glyph.RA,
    ),
    "fractal_contract": seq(
        block(Glyph.THOL, Glyph.NUL, Glyph.UM, repeat=2, close=Glyph.SHA),
        Glyph.RA,
    ),
    "canonical_example": list(CANONICAL_PROGRAM_TOKENS),
}

_LEGACY_PRESET_ALIASES: dict[str, str] = {
    "arranque_resonante": "resonant_bootstrap",
    "mutacion_contenida": "contained_mutation",
    "exploracion_acople": "coupling_exploration",
    CANONICAL_PRESET_NAME: "canonical_example",
}

PREFERRED_PRESET_NAMES: tuple[str, ...] = tuple(_PRIMARY_PRESETS.keys())
LEGACY_PRESET_NAMES: tuple[str, ...] = tuple(_LEGACY_PRESET_ALIASES.keys())
PRESET_NAME_ALIASES: dict[str, str] = dict(_LEGACY_PRESET_ALIASES)

_PRESETS: dict[str, PresetTokens] = {**_PRIMARY_PRESETS}
for alias, target in _LEGACY_PRESET_ALIASES.items():
    _PRESETS[alias] = _PRIMARY_PRESETS[target]


def get_preset(name: str) -> PresetTokens:
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(f"Preset no encontrado: {name}") from None
