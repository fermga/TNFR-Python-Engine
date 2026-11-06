"""Predefined contextual TNFR programs.

Only the canonical English preset identifiers are recognised.
Entries are supplied programs, not unconditional word or live-state admission.
In particular, nested THOL and Mutation require their actual inherited context.
Execution rejects blocked requests instead of silently substituting a glyph.
"""

from __future__ import annotations

from copy import deepcopy

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
        block(Glyph.OZ, Glyph.ZHIR, Glyph.IL, repeat=2, close=Glyph.NUL),
        Glyph.RA,
        Glyph.SHA,
    ),
    "coupling_exploration": seq(
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.VAL,
        Glyph.UM,
        block(Glyph.OZ, Glyph.NAV, Glyph.IL, repeat=1, close=Glyph.NUL),
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


def get_preset(name: str) -> PresetTokens:
    """Return an owned copy of the preset and its mutable nested blocks."""

    try:
        return deepcopy(_PRESETS[name])
    except KeyError:
        raise KeyError(f"Preset not found: {name}") from None
