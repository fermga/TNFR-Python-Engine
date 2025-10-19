"""Type definitions and protocols shared across the engine."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Protocol, TypeAlias

__all__ = (
    "TNFRGraph",
    "Graph",
    "NodeId",
    "Node",
    "GammaSpec",
    "EPIValue",
    "DeltaNFR",
    "SecondDerivativeEPI",
    "Phase",
    "StructuralFrequency",
    "SenseIndex",
    "CouplingWeight",
    "CoherenceMetric",
    "DeltaNFRHook",
    "GraphLike",
    "Glyph",
    "GlyphLoadDistribution",
)


if TYPE_CHECKING:  # pragma: no cover - import-time typing hook
    import networkx as nx

    TNFRGraph: TypeAlias = nx.Graph
else:  # pragma: no cover - runtime fallback without networkx
    TNFRGraph: TypeAlias = Any
#: Graph container storing TNFR nodes, edges and their coherence telemetry.

Graph: TypeAlias = TNFRGraph
#: Backwards-compatible alias for :data:`TNFRGraph`.

NodeId: TypeAlias = Hashable
#: Hashable identifier for a coherent TNFR node.

Node: TypeAlias = NodeId
#: Backwards-compatible alias for :data:`NodeId`.

GammaSpec: TypeAlias = Mapping[str, Any]
#: Mapping describing Γ evaluation parameters for a node or graph.

EPIValue: TypeAlias = float
#: Scalar Primary Information Structure value carried by a node.

DeltaNFR: TypeAlias = float
#: Scalar internal reorganisation driver ΔNFR applied to a node.

SecondDerivativeEPI: TypeAlias = float
#: Second derivative ∂²EPI/∂t² tracking bifurcation pressure.

Phase: TypeAlias = float
#: Phase (φ) describing a node's synchrony relative to its neighbors.

StructuralFrequency: TypeAlias = float
#: Structural frequency νf expressed in Hz_str.

SenseIndex: TypeAlias = float
#: Sense index Si capturing a node's reorganising capacity.

CouplingWeight: TypeAlias = float
#: Weight attached to edges describing coupling coherence strength.

CoherenceMetric: TypeAlias = float
#: Aggregated measure of coherence such as C(t) or Si.


class _DeltaNFRHookProtocol(Protocol):
    """Callable signature expected for ΔNFR update hooks.

    Hooks receive the graph instance and may expose optional keyword
    arguments such as ``n_jobs`` or cache controls. Additional positional
    arguments are reserved for future extensions and ignored by the core
    engine, keeping compatibility with user-provided hooks that only need the
    graph reference.
    """

    def __call__(
        self,
        graph: TNFRGraph,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        ...


DeltaNFRHook: TypeAlias = _DeltaNFRHookProtocol
#: Callable hook invoked to compute ΔNFR for a :data:`TNFRGraph`.


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


GlyphLoadDistribution: TypeAlias = dict[Glyph | str, float]
#: Normalised glyph load proportions keyed by :class:`Glyph` or aggregate labels.
