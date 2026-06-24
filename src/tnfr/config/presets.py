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


def legacy_preset_guidance(name: str) -> str | None:
    """Return CLI guidance for preset lookups.

    Legacy aliases were removed; the function now always returns ``None``.
    ``name`` is accepted to preserve the public helper signature.
    """

    return None


def get_preset(name: str) -> PresetTokens:
    """Return the preset token sequence identified by ``name``."""

    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(f"Preset not found: {name}") from None
