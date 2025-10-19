from __future__ import annotations

from typing import Any, ClassVar, Tuple

from ..types import Glyph, TNFRGraph

__all__: Tuple[str, ...]


class Operador:
    name: ClassVar[str]
    glyph: ClassVar[Glyph | None]

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None: ...


class Emision(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Recepcion(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Coherencia(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Disonancia(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Acoplamiento(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Resonancia(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Silencio(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Expansion(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Contraccion(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Autoorganizacion(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Mutacion(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Transicion(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]


class Recursividad(Operador):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]
