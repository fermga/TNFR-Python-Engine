"""Type definitions and protocols shared across the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol

__all__ = ("GraphLike", "NodeState", "Glyph")


class GraphLike(Protocol):
    """Protocol for graph objects used throughout TNFR metrics.

    The metrics helpers assume a single coherent graph interface so that
    coherence, resonance and derived indicators read/write data through the
    same structural access points.
    """

    graph: dict[str, Any]

    def nodes(self, data: bool = ...) -> Iterable[Any]: ...

    def number_of_nodes(self) -> int: ...

    def neighbors(self, n: Any) -> Iterable[Any]: ...

    def __iter__(self) -> Iterable[Any]: ...


@dataclass(slots=True)
class NodeState:
    EPI: float = 0.0
    vf: float = 0.0  # νf
    theta: float = 0.0  # θ
    Si: float = 0.0
    epi_kind: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Glyph(str, Enum):
    """Canonical TNFR glyphs."""

    AL = "AL"
    EN = "EN"
    IL = "IL"
    OZ = "OZ"
    UM = "UM"
    RA = "RA"
    SHA = "SHA"
    VAL = "VAL"
    NUL = "NUL"
    THOL = "THOL"
    ZHIR = "ZHIR"
    NAV = "NAV"
    REMESH = "REMESH"
