"""Canonical operator name constants and reusable sets.

Starting with TNFR 0.12 the engine uses **English** identifiers as the
canonical operator tokens. Spanish identifiers remain available as
compatibility aliases and will be removed in a future release.
"""

from __future__ import annotations

from itertools import chain
import warnings

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


# Legacy Spanish aliases (scheduled for removal) -----------------------------

EMISION = "emision"
RECEPCION = "recepcion"
COHERENCIA = "coherencia"
DISONANCIA = "disonancia"
ACOPLAMIENTO = "acoplamiento"
RESONANCIA = "resonancia"
SILENCIO = "silencio"
EXPANSION_ES = "expansion"
CONTRACCION = "contraccion"
AUTOORGANIZACION = "autoorganizacion"
MUTACION = "mutacion"
TRANSICION = "transicion"
RECURSIVIDAD = "recursividad"

SPANISH_TO_ENGLISH = {
    EMISION: EMISSION,
    RECEPCION: RECEPTION,
    COHERENCIA: COHERENCE,
    DISONANCIA: DISSONANCE,
    ACOPLAMIENTO: COUPLING,
    RESONANCIA: RESONANCE,
    SILENCIO: SILENCE,
    EXPANSION_ES: EXPANSION,
    CONTRACCION: CONTRACTION,
    AUTOORGANIZACION: SELF_ORGANIZATION,
    MUTACION: MUTATION,
    TRANSICION: TRANSITION,
    RECURSIVIDAD: RECURSIVITY,
}


# Bidirectional alias tables -------------------------------------------------

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
ENGLISH_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES
SPANISH_OPERATOR_NAMES = frozenset(SPANISH_TO_ENGLISH.keys())

ALIASES_BY_CANONICAL = {
    canonical: frozenset(
        chain([canonical], (alias for alias, mapped in SPANISH_TO_ENGLISH.items() if mapped == canonical))
    )
    for canonical in CANONICAL_OPERATOR_NAMES
}

CANONICAL_NAME_BY_ALIAS = {
    **{name: name for name in CANONICAL_OPERATOR_NAMES},
    **SPANISH_TO_ENGLISH,
}


def canonical_operator_name(name: str) -> str:
    """Return the canonical (English) operator token for ``name``.

    Using a legacy Spanish token triggers a :class:`DeprecationWarning` and
    returns the corresponding English identifier.
    """

    canonical = CANONICAL_NAME_BY_ALIAS.get(name, name)
    if name in SPANISH_TO_ENGLISH and canonical != name:
        warnings.warn(
            (
                "Spanish operator token '%s' is deprecated; use the English "
                "identifier '%s' instead"
            )
            % (name, canonical),
            DeprecationWarning,
            stacklevel=2,
        )
    return canonical


def operator_display_name(name: str) -> str:
    """Return a slash-joined label listing aliases for ``name``."""

    canonical = canonical_operator_name(name)
    aliases = ALIASES_BY_CANONICAL.get(canonical)
    if not aliases:
        return canonical
    return "/".join(sorted(aliases))


# Canonical collections used by validation and orchestration logic
ALL_OPERATOR_NAMES = frozenset(chain.from_iterable(ALIASES_BY_CANONICAL.values()))

INICIO_VALIDOS = frozenset(
    chain.from_iterable(ALIASES_BY_CANONICAL[name] for name in (EMISSION, RECURSIVITY))
)
TRAMO_INTERMEDIO = frozenset(
    chain.from_iterable(
        ALIASES_BY_CANONICAL[name]
        for name in (
            DISSONANCE,
            COUPLING,
            RESONANCE,
        )
    )
)
CIERRE_VALIDO = frozenset(
    chain.from_iterable(
        ALIASES_BY_CANONICAL[name]
        for name in (
            SILENCE,
            TRANSITION,
            RECURSIVITY,
        )
    )
)
AUTOORGANIZACION_CIERRES = frozenset(
    chain.from_iterable(
        ALIASES_BY_CANONICAL[name] for name in (SILENCE, CONTRACTION)
    )
)

__all__ = [
    # Canonical English tokens
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
    # Legacy Spanish tokens
    "EMISION",
    "RECEPCION",
    "COHERENCIA",
    "DISONANCIA",
    "ACOPLAMIENTO",
    "RESONANCIA",
    "SILENCIO",
    "EXPANSION_ES",
    "CONTRACCION",
    "AUTOORGANIZACION",
    "MUTACION",
    "TRANSICION",
    "RECURSIVIDAD",
    # Collections and helpers
    "SPANISH_TO_ENGLISH",
    "CANONICAL_OPERATOR_NAMES",
    "SPANISH_OPERATOR_NAMES",
    "ENGLISH_OPERATOR_NAMES",
    "ALL_OPERATOR_NAMES",
    "INICIO_VALIDOS",
    "TRAMO_INTERMEDIO",
    "CIERRE_VALIDO",
    "AUTOORGANIZACION_CIERRES",
    "ALIASES_BY_CANONICAL",
    "CANONICAL_NAME_BY_ALIAS",
    "canonical_operator_name",
    "operator_display_name",
]
