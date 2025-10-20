"""Definitions for canonical TNFR structural operators."""

from __future__ import annotations

from typing import Any, ClassVar

from ..types import Glyph, TNFRGraph
from ..config.operator_names import (
    ACOPLAMIENTO,
    AUTOORGANIZACION,
    COHERENCIA,
    CONTRACCION,
    DISONANCIA,
    EMISION,
    EXPANSION,
    MUTACION,
    RECEPCION,
    RECURSIVIDAD,
    RESONANCIA,
    SILENCIO,
    TRANSICION,
    COUPLING,
    COHERENCE,
    CONTRACTION,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
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
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
)


class Operador:
    """Base class for TNFR operators.

    Each operator defines ``name`` (ASCII identifier) and ``glyph``
    (símbolo TNFR canónico). Calling an instance applies the corresponding
    symbol to the node.
    """

    name: ClassVar[str] = "operador"
    glyph: ClassVar[Glyph | None] = None

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
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
    name: ClassVar[str] = EMISION
    glyph: ClassVar[Glyph] = Glyph.AL


@register_operator
class Recepcion(Operador):
    """Operador de recepción (símbolo ``EN``)."""

    __slots__ = ()
    name: ClassVar[str] = RECEPCION
    glyph: ClassVar[Glyph] = Glyph.EN


@register_operator
class Coherencia(Operador):
    """Operador de coherencia (símbolo ``IL``)."""

    __slots__ = ()
    name: ClassVar[str] = COHERENCIA
    glyph: ClassVar[Glyph] = Glyph.IL


@register_operator
class Disonancia(Operador):
    """Operador de disonancia (símbolo ``OZ``)."""

    __slots__ = ()
    name: ClassVar[str] = DISONANCIA
    glyph: ClassVar[Glyph] = Glyph.OZ


@register_operator
class Acoplamiento(Operador):
    """Operador de acoplamiento (símbolo ``UM``)."""

    __slots__ = ()
    name: ClassVar[str] = ACOPLAMIENTO
    glyph: ClassVar[Glyph] = Glyph.UM


@register_operator
class Resonancia(Operador):
    """Operador de resonancia (símbolo ``RA``)."""

    __slots__ = ()
    name: ClassVar[str] = RESONANCIA
    glyph: ClassVar[Glyph] = Glyph.RA


@register_operator
class Silencio(Operador):
    """Operador de silencio (símbolo ``SHA``)."""

    __slots__ = ()
    name: ClassVar[str] = SILENCIO
    glyph: ClassVar[Glyph] = Glyph.SHA


@register_operator
class Expansion(Operador):
    """Operador de expansión (símbolo ``VAL``)."""

    __slots__ = ()
    name: ClassVar[str] = EXPANSION
    glyph: ClassVar[Glyph] = Glyph.VAL


@register_operator
class Contraccion(Operador):
    """Operador de contracción (símbolo ``NUL``)."""

    __slots__ = ()
    name: ClassVar[str] = CONTRACCION
    glyph: ClassVar[Glyph] = Glyph.NUL


@register_operator
class Autoorganizacion(Operador):
    """Operador de autoorganización (símbolo ``THOL``)."""

    __slots__ = ()
    name: ClassVar[str] = AUTOORGANIZACION
    glyph: ClassVar[Glyph] = Glyph.THOL


@register_operator
class Mutacion(Operador):
    """Operador de mutación (símbolo ``ZHIR``)."""

    __slots__ = ()
    name: ClassVar[str] = MUTACION
    glyph: ClassVar[Glyph] = Glyph.ZHIR


@register_operator
class Transicion(Operador):
    """Operador de transición (símbolo ``NAV``)."""

    __slots__ = ()
    name: ClassVar[str] = TRANSICION
    glyph: ClassVar[Glyph] = Glyph.NAV


@register_operator
class Recursividad(Operador):
    """Operador de recursividad (símbolo ``REMESH``)."""

    __slots__ = ()
    name: ClassVar[str] = RECURSIVIDAD
    glyph: ClassVar[Glyph] = Glyph.REMESH


@register_operator
class Emission(Emision):
    """English alias for :class:`Emision` (glyph ``AL``)."""

    __slots__ = ()
    name: ClassVar[str] = EMISSION


@register_operator
class Reception(Recepcion):
    """English alias for :class:`Recepcion` (glyph ``EN``)."""

    __slots__ = ()
    name: ClassVar[str] = RECEPTION


@register_operator
class Coherence(Coherencia):
    """English alias for :class:`Coherencia` (glyph ``IL``)."""

    __slots__ = ()
    name: ClassVar[str] = COHERENCE


@register_operator
class Dissonance(Disonancia):
    """English alias for :class:`Disonancia` (glyph ``OZ``)."""

    __slots__ = ()
    name: ClassVar[str] = DISSONANCE


@register_operator
class Coupling(Acoplamiento):
    """English alias for :class:`Acoplamiento` (glyph ``UM``)."""

    __slots__ = ()
    name: ClassVar[str] = COUPLING


@register_operator
class Resonance(Resonancia):
    """English alias for :class:`Resonancia` (glyph ``RA``)."""

    __slots__ = ()
    name: ClassVar[str] = RESONANCE


@register_operator
class Silence(Silencio):
    """English alias for :class:`Silencio` (glyph ``SHA``)."""

    __slots__ = ()
    name: ClassVar[str] = SILENCE


@register_operator
class Contraction(Contraccion):
    """English alias for :class:`Contraccion` (glyph ``NUL``)."""

    __slots__ = ()
    name: ClassVar[str] = CONTRACTION


@register_operator
class SelfOrganization(Autoorganizacion):
    """English alias for :class:`Autoorganizacion` (glyph ``THOL``)."""

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION


@register_operator
class Mutation(Mutacion):
    """English alias for :class:`Mutacion` (glyph ``ZHIR``)."""

    __slots__ = ()
    name: ClassVar[str] = MUTATION


@register_operator
class Transition(Transicion):
    """English alias for :class:`Transicion` (glyph ``NAV``)."""

    __slots__ = ()
    name: ClassVar[str] = TRANSITION


@register_operator
class Recursivity(Recursividad):
    """English alias for :class:`Recursividad` (glyph ``REMESH``)."""

    __slots__ = ()
    name: ClassVar[str] = RECURSIVITY
