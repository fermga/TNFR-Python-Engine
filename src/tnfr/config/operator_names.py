"""Canonical operator name constants and reusable sets."""

from __future__ import annotations

from itertools import chain

# Individual operator identifiers (Spanish canonical tokens)
EMISION = "emision"
RECEPCION = "recepcion"
COHERENCIA = "coherencia"
DISONANCIA = "disonancia"
ACOPLAMIENTO = "acoplamiento"
RESONANCIA = "resonancia"
SILENCIO = "silencio"
EXPANSION = "expansion"
CONTRACCION = "contraccion"
AUTOORGANIZACION = "autoorganizacion"
MUTACION = "mutacion"
TRANSICION = "transicion"
RECURSIVIDAD = "recursividad"

# English equivalents used for multilingual dispatch
EMISSION = "emission"
RECEPTION = "reception"
COHERENCE = "coherence"
DISSONANCE = "dissonance"
COUPLING = "coupling"
RESONANCE = "resonance"
SILENCE = "silence"
EXPANSION_EN = "expansion"
CONTRACTION = "contraction"
SELF_ORGANIZATION = "self_organization"
MUTATION = "mutation"
TRANSITION = "transition"
RECURSIVITY = "recursivity"

# Bidirectional alias tables -------------------------------------------------

ENGLISH_NAME_BY_CANONICAL = {
    EMISION: EMISSION,
    RECEPCION: RECEPTION,
    COHERENCIA: COHERENCE,
    DISONANCIA: DISSONANCE,
    ACOPLAMIENTO: COUPLING,
    RESONANCIA: RESONANCE,
    SILENCIO: SILENCE,
    EXPANSION: EXPANSION_EN,
    CONTRACCION: CONTRACTION,
    AUTOORGANIZACION: SELF_ORGANIZATION,
    MUTACION: MUTATION,
    TRANSICION: TRANSITION,
    RECURSIVIDAD: RECURSIVITY,
}

ALIASES_BY_CANONICAL = {
    canonical: frozenset({canonical, english})
    for canonical, english in ENGLISH_NAME_BY_CANONICAL.items()
}

CANONICAL_OPERATOR_NAMES = frozenset(ALIASES_BY_CANONICAL.keys())
SPANISH_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES
ENGLISH_OPERATOR_NAMES = frozenset(ENGLISH_NAME_BY_CANONICAL.values())
CANONICAL_NAME_BY_ALIAS = {
    alias: canonical
    for canonical, aliases in ALIASES_BY_CANONICAL.items()
    for alias in aliases
}


def canonical_operator_name(name: str) -> str:
    """Return the canonical (Spanish) operator token for ``name``."""

    return CANONICAL_NAME_BY_ALIAS.get(name, name)


def operator_display_name(canonical: str) -> str:
    """Return a slash-joined label listing aliases for ``canonical``."""

    aliases = ALIASES_BY_CANONICAL.get(canonical)
    if not aliases:
        return canonical
    return "/".join(sorted(aliases))


# Canonical collections used by validation and orchestration logic
ALL_OPERATOR_NAMES = frozenset(chain.from_iterable(ALIASES_BY_CANONICAL.values()))

INICIO_VALIDOS = frozenset(
    chain.from_iterable(ALIASES_BY_CANONICAL[name] for name in (EMISION, RECURSIVIDAD))
)
TRAMO_INTERMEDIO = frozenset(
    chain.from_iterable(
        ALIASES_BY_CANONICAL[name]
        for name in (
            DISONANCIA,
            ACOPLAMIENTO,
            RESONANCIA,
        )
    )
)
CIERRE_VALIDO = frozenset(
    chain.from_iterable(
        ALIASES_BY_CANONICAL[name]
        for name in (
            SILENCIO,
            TRANSICION,
            RECURSIVIDAD,
        )
    )
)
AUTOORGANIZACION_CIERRES = frozenset(
    chain.from_iterable(
        ALIASES_BY_CANONICAL[name] for name in (SILENCIO, CONTRACCION)
    )
)

__all__ = [
    # Canonical Spanish tokens
    "EMISION",
    "RECEPCION",
    "COHERENCIA",
    "DISONANCIA",
    "ACOPLAMIENTO",
    "RESONANCIA",
    "SILENCIO",
    "EXPANSION",
    "CONTRACCION",
    "AUTOORGANIZACION",
    "MUTACION",
    "TRANSICION",
    "RECURSIVIDAD",
    # English aliases
    "EMISSION",
    "RECEPTION",
    "COHERENCE",
    "DISSONANCE",
    "COUPLING",
    "RESONANCE",
    "SILENCE",
    "EXPANSION_EN",
    "CONTRACTION",
    "SELF_ORGANIZATION",
    "MUTATION",
    "TRANSITION",
    "RECURSIVITY",
    # Collections and helpers
    "CANONICAL_OPERATOR_NAMES",
    "SPANISH_OPERATOR_NAMES",
    "ENGLISH_OPERATOR_NAMES",
    "ENGLISH_NAME_BY_CANONICAL",
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
