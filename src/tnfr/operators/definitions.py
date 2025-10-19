"""Definitions for canonical TNFR structural operators."""

from __future__ import annotations

from typing import Any

import networkx as nx

from ..types import Glyph
from ..config.operator_names import (
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
)
from .registry import register_operator

__all__ = (
    "Operador",
    "Emision",
    "Recepcion",
    "Coherencia",
    "Disonancia",
    "Acoplamiento",
    "Resonancia",
    "Silencio",
    "Expansion",
    "Contraccion",
    "Autoorganizacion",
    "Mutacion",
    "Transicion",
    "Recursividad",
)


class Operador:
    """Base class for TNFR operators.

    Each operator defines ``name`` (ASCII identifier) and ``glyph``
    (símbolo TNFR canónico). Calling an instance applies the corresponding
    symbol to the node.
    """

    name = "operador"
    glyph = None  # tipo: str

    def __call__(self, G: nx.Graph, node: Any, **kw: Any) -> None:
        if self.glyph is None:
            raise NotImplementedError("Operador sin glyph asignado")
        from ..validation.grammar import (
            apply_glyph_with_grammar,
        )  # local import to avoid cycles

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


@register_operator
class Emision(Operador):
    """Aplicación del operador de emisión (símbolo ``AL``)."""

    __slots__ = ()
    name = EMISION
    glyph = Glyph.AL.value


@register_operator
class Recepcion(Operador):
    """Operador de recepción (símbolo ``EN``)."""

    __slots__ = ()
    name = RECEPCION
    glyph = Glyph.EN.value


@register_operator
class Coherencia(Operador):
    """Operador de coherencia (símbolo ``IL``)."""

    __slots__ = ()
    name = COHERENCIA
    glyph = Glyph.IL.value


@register_operator
class Disonancia(Operador):
    """Operador de disonancia (símbolo ``OZ``)."""

    __slots__ = ()
    name = DISONANCIA
    glyph = Glyph.OZ.value


@register_operator
class Acoplamiento(Operador):
    """Operador de acoplamiento (símbolo ``UM``)."""

    __slots__ = ()
    name = ACOPLAMIENTO
    glyph = Glyph.UM.value


@register_operator
class Resonancia(Operador):
    """Operador de resonancia (símbolo ``RA``)."""

    __slots__ = ()
    name = RESONANCIA
    glyph = Glyph.RA.value


@register_operator
class Silencio(Operador):
    """Operador de silencio (símbolo ``SHA``)."""

    __slots__ = ()
    name = SILENCIO
    glyph = Glyph.SHA.value


@register_operator
class Expansion(Operador):
    """Operador de expansión (símbolo ``VAL``)."""

    __slots__ = ()
    name = EXPANSION
    glyph = Glyph.VAL.value


@register_operator
class Contraccion(Operador):
    """Operador de contracción (símbolo ``NUL``)."""

    __slots__ = ()
    name = CONTRACCION
    glyph = Glyph.NUL.value


@register_operator
class Autoorganizacion(Operador):
    """Operador de autoorganización (símbolo ``THOL``)."""

    __slots__ = ()
    name = AUTOORGANIZACION
    glyph = Glyph.THOL.value


@register_operator
class Mutacion(Operador):
    """Operador de mutación (símbolo ``ZHIR``)."""

    __slots__ = ()
    name = MUTACION
    glyph = Glyph.ZHIR.value


@register_operator
class Transicion(Operador):
    """Operador de transición (símbolo ``NAV``)."""

    __slots__ = ()
    name = TRANSICION
    glyph = Glyph.NAV.value


@register_operator
class Recursividad(Operador):
    """Operador de recursividad (símbolo ``REMESH``)."""

    __slots__ = ()
    name = RECURSIVIDAD
    glyph = Glyph.REMESH.value
