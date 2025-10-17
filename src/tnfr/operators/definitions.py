"""Definitions for canonical TNFR structural operators."""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from ..types import Glyph

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
        from ..grammar import apply_glyph_with_grammar  # local import to avoid cycles

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


class Emision(Operador):
    """Aplicación del operador de emisión (símbolo ``AL``)."""

    __slots__ = ()
    name = "emision"
    glyph = Glyph.AL.value


class Recepcion(Operador):
    """Operador de recepción (símbolo ``EN``)."""

    __slots__ = ()
    name = "recepcion"
    glyph = Glyph.EN.value


class Coherencia(Operador):
    """Operador de coherencia (símbolo ``IL``)."""

    __slots__ = ()
    name = "coherencia"
    glyph = Glyph.IL.value


class Disonancia(Operador):
    """Operador de disonancia (símbolo ``OZ``)."""

    __slots__ = ()
    name = "disonancia"
    glyph = Glyph.OZ.value


class Acoplamiento(Operador):
    """Operador de acoplamiento (símbolo ``UM``)."""

    __slots__ = ()
    name = "acoplamiento"
    glyph = Glyph.UM.value


class Resonancia(Operador):
    """Operador de resonancia (símbolo ``RA``)."""

    __slots__ = ()
    name = "resonancia"
    glyph = Glyph.RA.value


class Silencio(Operador):
    """Operador de silencio (símbolo ``SHA``)."""

    __slots__ = ()
    name = "silencio"
    glyph = Glyph.SHA.value


class Expansion(Operador):
    """Operador de expansión (símbolo ``VAL``)."""

    __slots__ = ()
    name = "expansion"
    glyph = Glyph.VAL.value


class Contraccion(Operador):
    """Operador de contracción (símbolo ``NUL``)."""

    __slots__ = ()
    name = "contraccion"
    glyph = Glyph.NUL.value


class Autoorganizacion(Operador):
    """Operador de autoorganización (símbolo ``THOL``)."""

    __slots__ = ()
    name = "autoorganizacion"
    glyph = Glyph.THOL.value


class Mutacion(Operador):
    """Operador de mutación (símbolo ``ZHIR``)."""

    __slots__ = ()
    name = "mutacion"
    glyph = Glyph.ZHIR.value


class Transicion(Operador):
    """Operador de transición (símbolo ``NAV``)."""

    __slots__ = ()
    name = "transicion"
    glyph = Glyph.NAV.value


class Recursividad(Operador):
    """Operador de recursividad (símbolo ``REMESH``)."""

    __slots__ = ()
    name = "recursividad"
    glyph = Glyph.REMESH.value
