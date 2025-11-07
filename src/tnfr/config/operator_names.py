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
VALID_END_OPERATORS = frozenset({SILENCE, TRANSITION, RECURSIVITY, DISSONANCE})
SELF_ORGANIZATION_CLOSURES = frozenset({SILENCE, CONTRACTION})

# R4 Bifurcation control: operators that enable structural transformations
# Legacy single-level destabilizers (for backward compatibility)
DESTABILIZERS = frozenset({DISSONANCE, TRANSITION, EXPANSION})  # OZ, NAV, VAL
TRANSFORMERS = frozenset({MUTATION, SELF_ORGANIZATION})  # ZHIR, THOL
BIFURCATION_WINDOW = 3  # Legacy: Search window for destabilizer precedent

# R4 Extended: Graduated destabilizer classification by intensity
DESTABILIZERS_STRONG = frozenset({DISSONANCE})  # OZ: explicit dissonance
DESTABILIZERS_MODERATE = frozenset({TRANSITION, EXPANSION})  # NAV, VAL: indirect
DESTABILIZERS_WEAK = frozenset({RECEPTION})  # EN: latent potential

# All destabilizers (union of all levels)
DESTABILIZERS_ALL = DESTABILIZERS_STRONG | DESTABILIZERS_MODERATE | DESTABILIZERS_WEAK

# R4 Extended: Bifurcation windows by destabilizer intensity
# These define how many operators can separate a destabilizer from a transformer
BIFURCATION_WINDOWS = {
    'strong': 4,    # OZ permits ZHIR/THOL within 4 operators
    'moderate': 2,  # NAV/VAL permit ZHIR/THOL within 2 operators
    'weak': 1,      # EN requires ZHIR/THOL as immediate successor
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
    "DESTABILIZERS",
    "TRANSFORMERS",
    "BIFURCATION_WINDOW",
    "DESTABILIZERS_STRONG",
    "DESTABILIZERS_MODERATE",
    "DESTABILIZERS_WEAK",
    "DESTABILIZERS_ALL",
    "BIFURCATION_WINDOWS",
    "canonical_operator_name",
    "operator_display_name",
]


def __getattr__(name: str) -> Any:
    """Provide a consistent ``AttributeError`` when names are missing."""

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
