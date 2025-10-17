"""Registry mapping operator names to their classes."""

from __future__ import annotations

from .definitions import (
    Operador,
    Emision,
    Recepcion,
    Coherencia,
    Disonancia,
    Acoplamiento,
    Resonancia,
    Silencio,
    Expansion,
    Contraccion,
    Autoorganizacion,
    Mutacion,
    Transicion,
    Recursividad,
)

OPERADORES: dict[str, type[Operador]] = {
    Emision.name: Emision,
    Recepcion.name: Recepcion,
    Coherencia.name: Coherencia,
    Disonancia.name: Disonancia,
    Acoplamiento.name: Acoplamiento,
    Resonancia.name: Resonancia,
    Silencio.name: Silencio,
    Expansion.name: Expansion,
    Contraccion.name: Contraccion,
    Autoorganizacion.name: Autoorganizacion,
    Mutacion.name: Mutacion,
    Transicion.name: Transicion,
    Recursividad.name: Recursividad,
}

__all__ = ("OPERADORES",)
