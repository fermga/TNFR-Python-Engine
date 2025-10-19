"""Type definitions and protocols shared across the engine."""

from __future__ import annotations

from collections.abc import Hashable
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Protocol, TypeAlias

__all__ = ("Graph", "Node", "GraphLike", "Glyph")


if TYPE_CHECKING:  # pragma: no cover - import-time typing hook
    import networkx as nx  # type: ignore[import-untyped]

    Graph: TypeAlias = nx.Graph
else:  # pragma: no cover - runtime fallback without networkx
    Graph: TypeAlias = Any

Node: TypeAlias = Hashable


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
