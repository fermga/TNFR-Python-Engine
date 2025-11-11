from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from enum import Enum
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Protocol,
    TypedDict,
    Union,
    cast,
)

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

__all__: tuple[str, ...] = (
    "TNFRGraph",
    "Graph",
    "ValidatorFunc",
    "NodeId",
    "Node",
    "GammaSpec",
    "EPIValue",
    "BEPIProtocol",
    "ensure_bepi",
    "serialize_bepi",
    "ZERO_BEPI_STORAGE",
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
    "TraceMetadata",
    "TraceSnapshot",
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
    "ParallelWijPayload",
    "RemeshMeta",
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

class BEPIProtocol(Protocol): ...

EPIValue: TypeAlias = BEPIProtocol
ZERO_BEPI_STORAGE: dict[str, tuple[complex, ...] | tuple[float, ...]]

def ensure_bepi(value: Any) -> "BEPIElement": ...
def serialize_bepi(
    value: Any,
) -> dict[str, tuple[complex, ...] | tuple[float, ...]]: ...

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

TNFRConfigScalar: TypeAlias = Union[bool, int, float, str, None]
TNFRConfigSequence: TypeAlias = Sequence[TNFRConfigScalar]
TNFRConfigValue: TypeAlias = Union[
    TNFRConfigScalar, TNFRConfigSequence, MutableMapping[str, "TNFRConfigValue"]
]

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

class _NodeViewLike(Protocol):
    def __iter__(self) -> Iterable[Any]: ...
    def __call__(self, data: bool = ...) -> Iterable[Any]: ...
    def __getitem__(self, node: Any) -> Mapping[str, Any]: ...

class _EdgeViewLike(Protocol):
    def __iter__(self) -> Iterable[Any]: ...
    def __call__(self, data: bool = ...) -> Iterable[Any]: ...

class GraphLike(Protocol):
    graph: MutableMapping[str, Any]
    nodes: _NodeViewLike
    edges: _EdgeViewLike

    def number_of_nodes(self) -> int: ...
    def neighbors(self, n: Any) -> Iterable[Any]: ...
    def __getitem__(self, node: Any) -> MutableMapping[Any, Any]: ...
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

GlyphCode: TypeAlias = Union[Glyph, str]
GlyphLoadDistribution: TypeAlias = dict[Union[Glyph, str], float]

class _SelectorLifecycle(Protocol):
    def __call__(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...
    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None: ...
    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...

GlyphSelector: TypeAlias = Union[Callable[[TNFRGraph, NodeId], GlyphCode], _SelectorLifecycle]
SelectorPreselectionMetrics: TypeAlias = Mapping[Any, SelectorMetrics]
SelectorPreselectionChoices: TypeAlias = Mapping[Any, Union[Glyph, str]]
SelectorPreselectionPayload: TypeAlias = tuple[
    SelectorPreselectionMetrics,
    SelectorPreselectionChoices,
]
TraceFieldFn: TypeAlias = Callable[[TNFRGraph], "TraceMetadata"]
TraceFieldMap: TypeAlias = Mapping[str, "TraceFieldFn"]
TraceFieldRegistry: TypeAlias = dict[str, dict[str, "TraceFieldFn"]]

class TraceMetadata(TypedDict, total=False):
    gamma: Mapping[str, Any]
    grammar: Mapping[str, Any]
    selector: str | None
    dnfr_weights: Mapping[str, Any]
    si_weights: Mapping[str, Any]
    si_sensitivity: Mapping[str, Any]
    callbacks: Mapping[str, list[str] | None]
    thol_open_nodes: int
    kuramoto: Mapping[str, float]
    sigma: Mapping[str, float]
    glyphs: Mapping[str, int]

class TraceSnapshot(TraceMetadata, total=False):
    t: float
    phase: str

HistoryState: TypeAlias = Union[_HistoryDict, dict[str, Any]]
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
DnfrVectorMap: TypeAlias = dict[str, Union[np.ndarray, None]]
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
GlyphTimingByNode: TypeAlias = MutableMapping[Any, MutableMapping[str, MutableSequence[float]]]
GlyphCounts: TypeAlias = Mapping[str, int]
GlyphMetricsHistoryValue: TypeAlias = Union[MutableMapping[Any, Any], MutableSequence[Any]]
GlyphMetricsHistory: TypeAlias = MutableMapping[str, GlyphMetricsHistoryValue]
MetricsListHistory: TypeAlias = MutableMapping[str, list[Any]]

class RemeshMeta(TypedDict, total=False):
    alpha: float
    alpha_source: str
    tau_global: int
    tau_local: int
    step: int | None
    topo_hash: str | None
    epi_mean_before: float
    epi_mean_after: float
    epi_checksum_before: str
    epi_checksum_after: str
    stable_frac_last: float
    phase_sync_last: float
    glyph_disr_last: float

class ParallelWijPayload(TypedDict):
    epi_vals: Sequence[float]
    vf_vals: Sequence[float]
    si_vals: Sequence[float]
    cos_vals: Sequence[float]
    sin_vals: Sequence[float]
    weights: tuple[float, float, float, float]
    epi_range: float
    vf_range: float
