"""Canonical operator name constants and reusable sets."""

from __future__ import annotations

# Individual operator identifiers
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

# Canonical collections used by validation and orchestration logic
ALL_OPERATOR_NAMES = frozenset(
    {
        EMISION,
        RECEPCION,
        COHERENCIA,
        DISONANCIA,
        ACOPLAMIENTO,
        RESONANCIA,
        SILENCIO,
        EXPANSION,
        CONTRACCION,
        AUTOORGANIZACION,
        MUTACION,
        TRANSICION,
        RECURSIVIDAD,
    }
)

INICIO_VALIDOS = frozenset({EMISION, RECURSIVIDAD})
TRAMO_INTERMEDIO = frozenset({DISONANCIA, ACOPLAMIENTO, RESONANCIA})
CIERRE_VALIDO = frozenset({SILENCIO, TRANSICION, RECURSIVIDAD})
AUTOORGANIZACION_CIERRES = frozenset({SILENCIO, CONTRACCION})

__all__ = [
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
    "ALL_OPERATOR_NAMES",
    "INICIO_VALIDOS",
    "TRAMO_INTERMEDIO",
    "CIERRE_VALIDO",
    "AUTOORGANIZACION_CIERRES",
]
