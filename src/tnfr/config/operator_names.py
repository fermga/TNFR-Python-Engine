"""Canonical operator name constants and reusable sets."""

from __future__ import annotations


# Canonical operator identifiers (English tokens)
EMISSION = "emission"
RECEPTION = "reception"
COHERENCE = "coherence"
DISSONANCE = "dissonance"
COUPLING = "coupling"
RESONANCE = "resonance"
SILENCE = "silence"
EXPANSION = "expansion"
CONTRACTION = "contraction"
SELF_ORGANIZATION = "self_organization"
MUTATION = "mutation"
TRANSITION = "transition"
RECURSIVITY = "recursivity"


# Canonical collections -------------------------------------------------------

CANONICAL_OPERATOR_NAMES = frozenset(
    {
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        COUPLING,
        RESONANCE,
        SILENCE,
        EXPANSION,
        CONTRACTION,
        SELF_ORGANIZATION,
        MUTATION,
        TRANSITION,
        RECURSIVITY,
    }
)

ALL_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES
ENGLISH_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES

INICIO_VALIDOS = frozenset({EMISSION, RECURSIVITY})
TRAMO_INTERMEDIO = frozenset({DISSONANCE, COUPLING, RESONANCE})
CIERRE_VALIDO = frozenset({SILENCE, TRANSITION, RECURSIVITY})
AUTOORGANIZACION_CIERRES = frozenset({SILENCE, CONTRACTION})


def canonical_operator_name(name: str) -> str:
    """Return the canonical operator token for ``name``."""

    return name


def operator_display_name(name: str) -> str:
    """Return the display label for ``name`` (currently the canonical token)."""

    return canonical_operator_name(name)


__all__ = [
    "EMISSION",
    "RECEPTION",
    "COHERENCE",
    "DISSONANCE",
    "COUPLING",
    "RESONANCE",
    "SILENCE",
    "EXPANSION",
    "CONTRACTION",
    "SELF_ORGANIZATION",
    "MUTATION",
    "TRANSITION",
    "RECURSIVITY",
    "CANONICAL_OPERATOR_NAMES",
    "ENGLISH_OPERATOR_NAMES",
    "ALL_OPERATOR_NAMES",
    "INICIO_VALIDOS",
    "TRAMO_INTERMEDIO",
    "CIERRE_VALIDO",
    "AUTOORGANIZACION_CIERRES",
    "canonical_operator_name",
    "operator_display_name",
]
