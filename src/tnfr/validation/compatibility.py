"""Canonical TNFR compatibility tables expressed via structural operators."""

from __future__ import annotations

from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from ..operators import grammar as _grammar
from ..types import Glyph

__all__ = ["CANON_COMPAT", "CANON_FALLBACK"]

# Canonical compatibilities (allowed next operators) expressed via structural names
_STRUCTURAL_COMPAT: dict[str, set[str]] = {
    # Opening / initiation
    EMISSION: {RECEPTION, RESONANCE, TRANSITION, EXPANSION, COUPLING},
    RECEPTION: {COHERENCE, COUPLING, RESONANCE, TRANSITION},
    # Stabilisation / diffusion / coupling
    COHERENCE: {RESONANCE, EXPANSION, COUPLING, SILENCE},
    COUPLING: {RESONANCE, COHERENCE, EXPANSION, TRANSITION},
    RESONANCE: {COHERENCE, EXPANSION, COUPLING, TRANSITION},
    EXPANSION: {COUPLING, RESONANCE, COHERENCE, TRANSITION},
    # Dissonance → transition → mutation
    DISSONANCE: {MUTATION, TRANSITION},
    MUTATION: {COHERENCE, TRANSITION},
    TRANSITION: {DISSONANCE, MUTATION, RESONANCE, COHERENCE, COUPLING},
    # Closures / latent states
    SILENCE: {EMISSION, RECEPTION},
    CONTRACTION: {EMISSION, COHERENCE},
    # Self-organising blocks
    SELF_ORGANIZATION: {
        DISSONANCE,
        MUTATION,
        TRANSITION,
        RESONANCE,
        COHERENCE,
        COUPLING,
        SILENCE,
        CONTRACTION,
    },
}


def _name_to_glyph(name: str) -> Glyph:
    glyph = _grammar.function_name_to_glyph(name)
    if glyph is None:
        raise KeyError(f"No glyph mapped to structural operator '{name}'")
    return glyph


def _translate_structural() -> tuple[dict[Glyph, set[Glyph]], dict[Glyph, Glyph]]:
    compat: dict[Glyph, set[Glyph]] = {}
    for src, targets in _STRUCTURAL_COMPAT.items():
        src_glyph = _name_to_glyph(src)
        compat[src_glyph] = {_name_to_glyph(target) for target in targets}
    fallback: dict[Glyph, Glyph] = {}
    for src, target in _STRUCTURAL_FALLBACK.items():
        fallback[_name_to_glyph(src)] = _name_to_glyph(target)
    return compat, fallback


# Canonical fallbacks when a transition is not allowed (structural names)
_STRUCTURAL_FALLBACK: dict[str, str] = {
    EMISSION: RECEPTION,
    RECEPTION: COHERENCE,
    COHERENCE: RESONANCE,
    TRANSITION: RESONANCE,
    CONTRACTION: EMISSION,
    DISSONANCE: MUTATION,
    RESONANCE: COHERENCE,
    SILENCE: EMISSION,
    SELF_ORGANIZATION: TRANSITION,
    COUPLING: RESONANCE,
    EXPANSION: RESONANCE,
    MUTATION: COHERENCE,
}


CANON_COMPAT, CANON_FALLBACK = _translate_structural()

# Re-export structural tables for internal consumers that operate on functional
# identifiers without exposing them as part of the public API.
_STRUCTURAL_COMPAT_TABLE = _STRUCTURAL_COMPAT
_STRUCTURAL_FALLBACK_TABLE = _STRUCTURAL_FALLBACK
