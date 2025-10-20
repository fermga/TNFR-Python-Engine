"""Predefined TNFR configuration sequences.

Legacy preset identifiers are no longer accepted. Requests for Spanish or
otherwise retired names raise :class:`KeyError` with guidance pointing to the
canonical English identifier.
"""

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
    "REMOVED_PRESET_NAMES",
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
    CANONICAL_PRESET_NAME: list(CANONICAL_PROGRAM_TOKENS),
}

_REMOVED_PRESETS: dict[str, tuple[str, str]] = {
    "arranque_resonante": ("resonant_bootstrap", "TNFR 7.0"),
    "mutacion_contenida": ("contained_mutation", "TNFR 7.0"),
    "exploracion_acople": ("coupling_exploration", "TNFR 7.0"),
    "ejemplo_canonico": (CANONICAL_PRESET_NAME, "TNFR 9.0"),
}

PREFERRED_PRESET_NAMES: tuple[str, ...] = tuple(_PRIMARY_PRESETS.keys())
REMOVED_PRESET_NAMES: tuple[str, ...] = tuple(_REMOVED_PRESETS.keys())

_PRESETS: dict[str, PresetTokens] = {**_PRIMARY_PRESETS}


def get_preset(name: str) -> PresetTokens:
    removed = _REMOVED_PRESETS.get(name)
    if removed is not None:
        replacement, version = removed
        raise KeyError(
            (
                "Legacy preset identifier '%s' was removed in %s. "
                "Use '%s' instead."
            )
            % (name, version, replacement)
        )

    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(f"Preset not found: {name}") from None
