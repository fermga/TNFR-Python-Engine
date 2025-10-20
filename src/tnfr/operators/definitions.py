"""Definitions for canonical TNFR structural operators.

English identifiers are the public API. Spanish wrappers were removed in
TNFR 2.0, so downstream code must import these classes directly.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import (
    COHERENCE,
    COUPLING,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
    CONTRACTION,
    EXPANSION,
)
from ..types import Glyph, TNFRGraph
from .registry import register_operator

__all__ = [
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
]


class Operator:
    """Base class for TNFR operators.

    Each operator defines ``name`` (ASCII identifier) and ``glyph``. Calling an
    instance applies the corresponding glyph to the node.
    """

    name: ClassVar[str] = "operator"
    glyph: ClassVar[Glyph | None] = None

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        if self.glyph is None:
            raise NotImplementedError("Operator without assigned glyph")
        from ..validation.grammar import (  # local import to avoid cycles
            apply_glyph_with_grammar,
        )

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


@register_operator
class Emission(Operator):
    """Emission operator (glyph ``AL``)."""

    __slots__ = ()
    name: ClassVar[str] = EMISSION
    glyph: ClassVar[Glyph] = Glyph.AL


@register_operator
class Reception(Operator):
    """Reception operator (glyph ``EN``)."""

    __slots__ = ()
    name: ClassVar[str] = RECEPTION
    glyph: ClassVar[Glyph] = Glyph.EN


@register_operator
class Coherence(Operator):
    """Coherence operator (glyph ``IL``)."""

    __slots__ = ()
    name: ClassVar[str] = COHERENCE
    glyph: ClassVar[Glyph] = Glyph.IL


@register_operator
class Dissonance(Operator):
    """Dissonance operator (glyph ``OZ``)."""

    __slots__ = ()
    name: ClassVar[str] = DISSONANCE
    glyph: ClassVar[Glyph] = Glyph.OZ


@register_operator
class Coupling(Operator):
    """Coupling operator (glyph ``UM``)."""

    __slots__ = ()
    name: ClassVar[str] = COUPLING
    glyph: ClassVar[Glyph] = Glyph.UM


@register_operator
class Resonance(Operator):
    """Resonance operator (glyph ``RA``)."""

    __slots__ = ()
    name: ClassVar[str] = RESONANCE
    glyph: ClassVar[Glyph] = Glyph.RA


@register_operator
class Silence(Operator):
    """Silence operator (glyph ``SHA``)."""

    __slots__ = ()
    name: ClassVar[str] = SILENCE
    glyph: ClassVar[Glyph] = Glyph.SHA


@register_operator
class Expansion(Operator):
    """Expansion operator (glyph ``VAL``)."""

    __slots__ = ()
    name: ClassVar[str] = EXPANSION
    glyph: ClassVar[Glyph] = Glyph.VAL


@register_operator
class Contraction(Operator):
    """Contraction operator (glyph ``NUL``)."""

    __slots__ = ()
    name: ClassVar[str] = CONTRACTION
    glyph: ClassVar[Glyph] = Glyph.NUL


@register_operator
class SelfOrganization(Operator):
    """Self-organization operator (glyph ``THOL``)."""

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION
    glyph: ClassVar[Glyph] = Glyph.THOL


@register_operator
class Mutation(Operator):
    """Mutation operator (glyph ``ZHIR``)."""

    __slots__ = ()
    name: ClassVar[str] = MUTATION
    glyph: ClassVar[Glyph] = Glyph.ZHIR


@register_operator
class Transition(Operator):
    """Transition operator (glyph ``NAV``)."""

    __slots__ = ()
    name: ClassVar[str] = TRANSITION
    glyph: ClassVar[Glyph] = Glyph.NAV


@register_operator
class Recursivity(Operator):
    """Recursivity operator (glyph ``REMESH``)."""

    __slots__ = ()
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH
