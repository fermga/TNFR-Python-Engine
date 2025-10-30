from collections.abc import Hashable, Mapping, MutableMapping, MutableSequence, Sequence
from enum import Enum
from typing import Any, Callable, ContextManager, Iterable, Protocol, TypedDict, cast

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

from .glyph_history import HistoryDict as _HistoryDict
from .tokens import Token
from .trace import TraceMetadata

__all__: tuple[str, ...] = (
    "TNFRGraph",
    "Graph",
    "ValidatorFunc",
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
    "IntegratorProtocol",
    "Glyph",
    "GlyphCode",
    "GlyphLoadDistribution",
    "GlyphSelector",
    "SelectorPreselectionMetrics",
    "SelectorPreselectionChoices",
    "SelectorPreselectionPayload",
    "SelectorMetrics",
    "SelectorNorms",
    "SelectorThresholds",
    "SelectorWeights",
    "TraceCallback",
    "CallbackError",
    "TraceFieldFn",
    "TraceFieldMap",
    "TraceFieldRegistry",
    "HistoryState",
    "DiagnosisNodeData",
    "DiagnosisSharedState",
    "DiagnosisPayload",
    "DiagnosisResult",
    "DiagnosisPayloadChunk",
    "DiagnosisResultList",
    "DnfrCacheVectors",
    "DnfrVectorMap",
    "NeighborStats",
    "TimingContext",
    "PresetTokens",
    "ProgramTokens",
    "ArgSpec",
    "TNFRConfigValue",
    "SigmaVector",
    "SigmaTrace",
    "FloatArray",
    "FloatMatrix",
    "NodeInitAttrMap",
    "NodeAttrMap",
    "GlyphogramRow",
    "GlyphTimingTotals",
    "GlyphTimingByNode",
    "GlyphCounts",
    "GlyphMetricsHistoryValue",
    "GlyphMetricsHistory",
    "MetricsListHistory",
)

def __getattr__(name: str) -> Any: ...

TNFRGraph: TypeAlias = nx.Graph
Graph: TypeAlias = TNFRGraph
ValidatorFunc: TypeAlias = Callable[[TNFRGraph], None]
NodeId: TypeAlias = Hashable
Node: TypeAlias = NodeId
NodeInitAttrMap: TypeAlias = MutableMapping[str, float]
NodeAttrMap: TypeAlias = Mapping[str, Any]
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
ProgramTokens: TypeAlias = Sequence[Token]
ArgSpec: TypeAlias = tuple[str, Mapping[str, Any]]

TNFRConfigScalar: TypeAlias = bool | int | float | str | None
TNFRConfigSequence: TypeAlias = Sequence[TNFRConfigScalar]
TNFRConfigValue: TypeAlias = (
    TNFRConfigScalar | TNFRConfigSequence | Mapping[str, "TNFRConfigValue"]
)

class _SigmaVectorRequired(TypedDict):
    x: float
    y: float
    mag: float
    angle: float
    n: int


class _SigmaVectorOptional(TypedDict, total=False):
    glyph: str
    w: float
    t: float


class SigmaVector(_SigmaVectorRequired, _SigmaVectorOptional): ...


class SigmaTrace(TypedDict):
    t: list[float]
    sigma_x: list[float]
    sigma_y: list[float]
    mag: list[float]
    angle: list[float]


FloatArray: TypeAlias = np.ndarray
FloatMatrix: TypeAlias = np.ndarray

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

GlyphCode: TypeAlias = Glyph | str
GlyphLoadDistribution: TypeAlias = dict[Glyph | str, float]


class _SelectorLifecycle(Protocol):
    def __call__(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...
    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None: ...
    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...


GlyphSelector: TypeAlias = Callable[[TNFRGraph, NodeId], GlyphCode] | _SelectorLifecycle
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


class CallbackError(TypedDict):
    event: str
    step: int | None
    error: str
    traceback: str
    fn: str
    name: str | None


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

GlyphogramRow: TypeAlias = MutableMapping[str, float]
GlyphTimingTotals: TypeAlias = MutableMapping[str, float]
GlyphTimingByNode: TypeAlias = MutableMapping[
    Any, MutableMapping[str, MutableSequence[float]]
]
GlyphCounts: TypeAlias = Mapping[str, int]
GlyphMetricsHistoryValue: TypeAlias = MutableMapping[Any, Any] | MutableSequence[Any]
GlyphMetricsHistory: TypeAlias = MutableMapping[str, GlyphMetricsHistoryValue]
MetricsListHistory: TypeAlias = MutableMapping[str, list[Any]]
