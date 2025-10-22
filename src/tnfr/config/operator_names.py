"""Canonical operator name constants and reusable sets."""

from __future__ import annotations

from typing import Any


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

VALID_START_OPERATORS = frozenset({EMISSION, RECURSIVITY})
INTERMEDIATE_OPERATORS = frozenset({DISSONANCE, COUPLING, RESONANCE})
VALID_END_OPERATORS = frozenset({SILENCE, TRANSITION, RECURSIVITY})
SELF_ORGANIZATION_CLOSURES = frozenset({SILENCE, CONTRACTION})


_LEGACY_COLLECTION_ALIASES: dict[str, str] = {
    "INICIO_VALIDOS": "VALID_START_OPERATORS",
    "TRAMO_INTERMEDIO": "INTERMEDIATE_OPERATORS",
    "CIERRE_VALIDO": "VALID_END_OPERATORS",
    "AUTOORGANIZACION_CIERRES": "SELF_ORGANIZATION_CLOSURES",
}

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
    "VALID_START_OPERATORS",
    "INTERMEDIATE_OPERATORS",
    "VALID_END_OPERATORS",
    "SELF_ORGANIZATION_CLOSURES",
    "canonical_operator_name",
    "operator_display_name",
]


def __getattr__(name: str) -> Any:
    """Provide guidance for legacy operator collection aliases."""

    canonical = _LEGACY_COLLECTION_ALIASES.get(name)
    if canonical is not None:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'; use '{canonical}' instead."
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
