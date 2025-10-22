"""Predefined TNFR configuration sequences.

Only the canonical English preset identifiers are recognised.
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
    "legacy_preset_guidance",
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

PREFERRED_PRESET_NAMES: tuple[str, ...] = tuple(_PRIMARY_PRESETS.keys())

_PRESETS: dict[str, PresetTokens] = {**_PRIMARY_PRESETS}


_LEGACY_PRESET_ALIASES: dict[str, str] = {
    "arranque_resonante": "resonant_bootstrap",
    "mutacion_contenida": "contained_mutation",
    "exploracion_acople": "coupling_exploration",
    "ejemplo_canonico": CANONICAL_PRESET_NAME,
}


def legacy_preset_guidance(name: str) -> str | None:
    """Return CLI guidance for historical preset aliases.

    Parameters
    ----------
    name:
        Identifier received from the CLI.

    Returns
    -------
    str | None
        A human readable guidance string when ``name`` matches a removed
        alias. ``None`` when no dedicated guidance is available.
    """

    canonical = _LEGACY_PRESET_ALIASES.get(name)
    if canonical is None:
        return None
    return (
        f"Legacy preset identifier '{name}' was removed in TNFR 9.0. "
        f"Use '{canonical}' instead."
    )


def get_preset(name: str) -> PresetTokens:
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(f"Preset not found: {name}") from None
