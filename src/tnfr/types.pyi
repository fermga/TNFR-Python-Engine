from typing import Any, Callable, ContextManager, Iterable, Protocol, cast
from collections.abc import Hashable, Mapping, Sequence
from enum import Enum
from typing import TypedDict

from ._compat import TypeAlias

try:
    import networkx as nx  # type: ignore[import-not-found]
except Exception:
    class _FallbackGraph: ...

    class _FallbackNetworkX:
        Graph = _FallbackGraph

    nx = cast(Any, _FallbackNetworkX())

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception:
    class _FallbackNdArray: ...

    class _FallbackNumpy:
        ndarray = _FallbackNdArray

    np = cast(Any, _FallbackNumpy())

from .tokens import Token
from .trace import TraceMetadata
from .glyph_history import HistoryDict as _HistoryDict

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...

TNFRGraph: TypeAlias = nx.Graph
Graph: TypeAlias = TNFRGraph
NodeId: TypeAlias = Hashable
Node: TypeAlias = NodeId
GammaSpec: TypeAlias = Mapping[str, Any]
EPIValue: TypeAlias = float
DeltaNFR: TypeAlias = float
SecondDerivativeEPI: TypeAlias = float
Phase: TypeAlias = float
StructuralFrequency: TypeAlias = float
SenseIndex: TypeAlias = float
CouplingWeight: TypeAlias = float
CoherenceMetric: TypeAlias = float
TimingContext: TypeAlias = ContextManager[None]
PresetTokens: TypeAlias = Sequence[Token]


class SelectorThresholds(TypedDict):
    si_hi: float
    si_lo: float
    dnfr_hi: float
    dnfr_lo: float
    accel_hi: float
    accel_lo: float


class SelectorWeights(TypedDict):
    w_si: float
    w_dnfr: float
    w_accel: float


SelectorMetrics: TypeAlias = tuple[float, float, float]
SelectorNorms: TypeAlias = Mapping[str, float]

class _DeltaNFRHookProtocol(Protocol):
    def __call__(self, graph: TNFRGraph, /, *args: Any, **kwargs: Any) -> None: ...

DeltaNFRHook: TypeAlias = _DeltaNFRHookProtocol

class GraphLike(Protocol):
    graph: dict[str, Any]

    def nodes(self, data: bool = ...) -> Iterable[Any]: ...
    def number_of_nodes(self) -> int: ...
    def neighbors(self, n: Any) -> Iterable[Any]: ...
    def __iter__(self) -> Iterable[Any]: ...

class IntegratorProtocol(Protocol):
    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None = ...,
        t: float | None = ...,
        method: str | None = ...,
        n_jobs: int | None = ...,
    ) -> None: ...

class Glyph(str, Enum):
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
GlyphSelector: TypeAlias = Callable[[TNFRGraph, NodeId], Glyph | str]
SelectorPreselectionMetrics: TypeAlias = Mapping[Any, SelectorMetrics]
SelectorPreselectionChoices: TypeAlias = Mapping[Any, Glyph | str]
SelectorPreselectionPayload: TypeAlias = tuple[
    SelectorPreselectionMetrics,
    SelectorPreselectionChoices,
]
TraceFieldFn: TypeAlias = Callable[[TNFRGraph], TraceMetadata]
TraceFieldMap: TypeAlias = Mapping[str, TraceFieldFn]
TraceFieldRegistry: TypeAlias = dict[str, dict[str, TraceFieldFn]]
HistoryState: TypeAlias = _HistoryDict | dict[str, Any]
TraceCallback: TypeAlias = Callable[[TNFRGraph, dict[str, Any]], None]
DiagnosisNodeData: TypeAlias = Mapping[str, Any]
DiagnosisSharedState: TypeAlias = Mapping[str, Any]
DiagnosisPayload: TypeAlias = dict[str, Any]
DiagnosisResult: TypeAlias = tuple[NodeId, DiagnosisPayload]
DiagnosisPayloadChunk: TypeAlias = list[DiagnosisNodeData]
DiagnosisResultList: TypeAlias = list[DiagnosisResult]
DnfrCacheVectors: TypeAlias = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]
DnfrVectorMap: TypeAlias = dict[str, np.ndarray | None]
NeighborStats: TypeAlias = tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float] | None,
    Sequence[float] | None,
    Sequence[float] | None,
]
