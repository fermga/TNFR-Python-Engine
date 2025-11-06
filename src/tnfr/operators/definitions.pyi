from __future__ import annotations

from ..types import Glyph, NodeId, TNFRGraph
from typing import Any, ClassVar

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
    name: ClassVar[str]
    glyph: ClassVar[Glyph | None]
    def __call__(self, G: TNFRGraph, node: NodeId, **kw: Any) -> None: ...

class Emission(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Reception(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Coherence(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Dissonance(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Coupling(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Resonance(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Silence(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Expansion(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Contraction(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class SelfOrganization(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Mutation(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Transition(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]

class Recursivity(Operator):
    name: ClassVar[str]
    glyph: ClassVar[Glyph]
