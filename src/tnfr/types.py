"""Type definitions and protocols shared across the engine."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, MutableMapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, ContextManager, Iterable, Protocol, TypedDict
from types import SimpleNamespace

from ._compat import TypeAlias

try:  # pragma: no cover - optional dependency for typing only
    import numpy as np
except Exception:  # pragma: no cover - graceful fallback when NumPy is missing
    np = SimpleNamespace(ndarray=Any, float_=float)  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - import-time typing hook
    try:
        import numpy.typing as npt
    except Exception:  # pragma: no cover - fallback when NumPy typing is missing
        npt = SimpleNamespace(NDArray=Any)  # type: ignore[assignment]
else:  # pragma: no cover - runtime fallback without numpy.typing
    npt = SimpleNamespace(NDArray=Any)  # type: ignore[assignment]

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
    "IntegratorProtocol",
    "Glyph",
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
    "FloatArray",
    "FloatMatrix",
    "NodeInitAttrMap",
)


if TYPE_CHECKING:  # pragma: no cover - import-time typing hook
    import networkx as nx
    from .trace import TraceMetadata
    from .glyph_history import HistoryDict as _HistoryDict
    from .tokens import Token as _Token

    TNFRGraph: TypeAlias = nx.Graph
else:  # pragma: no cover - runtime fallback without networkx
    TNFRGraph: TypeAlias = Any
    _HistoryDict = Any  # type: ignore[assignment]
    _Token = Any  # type: ignore[assignment]
#: Graph container storing TNFR nodes, edges and their coherence telemetry.

if TYPE_CHECKING:
    FloatArray: TypeAlias = npt.NDArray[np.float_]
    FloatMatrix: TypeAlias = npt.NDArray[np.float_]
else:  # pragma: no cover - runtime fallback without NumPy
    FloatArray: TypeAlias = Any
    FloatMatrix: TypeAlias = Any

Graph: TypeAlias = TNFRGraph
#: Backwards-compatible alias for :data:`TNFRGraph`.

NodeId: TypeAlias = Hashable
#: Hashable identifier for a coherent TNFR node.

Node: TypeAlias = NodeId
#: Backwards-compatible alias for :data:`NodeId`.

NodeInitAttrMap: TypeAlias = MutableMapping[str, float]
#: Mutable mapping storing scalar node attributes during initialization.

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

TimingContext: TypeAlias = ContextManager[None]
#: Context manager used to measure execution time for cache operations.

ProgramTokens: TypeAlias = Sequence[_Token]
#: Sequence of execution tokens composing a TNFR program.

PresetTokens: TypeAlias = Sequence[_Token]
#: Sequence of execution tokens composing a preset program.

ArgSpec: TypeAlias = tuple[str, Mapping[str, Any]]
#: CLI argument specification pairing an option flag with keyword arguments.


TNFRConfigScalar: TypeAlias = bool | int | float | str | None
"""Primitive value allowed within TNFR configuration stores."""

TNFRConfigSequence: TypeAlias = Sequence[TNFRConfigScalar]
"""Homogeneous sequence of scalar TNFR configuration values."""

TNFRConfigValue: TypeAlias = (
    TNFRConfigScalar | TNFRConfigSequence | Mapping[str, "TNFRConfigValue"]
)
"""Permissible configuration entry for TNFR coherence defaults.

The alias captures the recursive structure used by TNFR defaults: scalars
express structural thresholds, booleans toggle operators, and nested mappings
or sequences describe coherent parameter bundles such as γ grammars,
selector advice or trace capture lists.
"""


class _SigmaVectorRequired(TypedDict):
    """Mandatory components for a σ-vector in the sense plane."""

    x: float
    y: float
    mag: float
    angle: float
    n: int


class _SigmaVectorOptional(TypedDict, total=False):
    """Optional metadata captured when tracking σ-vectors."""

    glyph: str
    w: float
    t: float


class SigmaVector(_SigmaVectorRequired, _SigmaVectorOptional):
    """Typed dictionary describing σ-vector telemetry."""


class SelectorThresholds(TypedDict):
    """Normalised thresholds applied by the glyph selector."""

    si_hi: float
    si_lo: float
    dnfr_hi: float
    dnfr_lo: float
    accel_hi: float
    accel_lo: float


class SelectorWeights(TypedDict):
    """Normalised weights controlling selector scoring."""

    w_si: float
    w_dnfr: float
    w_accel: float


SelectorMetrics: TypeAlias = tuple[float, float, float]
"""Tuple grouping normalised Si, |ΔNFR| and acceleration values."""

SelectorNorms: TypeAlias = Mapping[str, float]
"""Mapping storing maxima used to normalise selector metrics."""


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

    def nodes(self, data: bool = ...) -> Iterable[Any]:
        ...

    def number_of_nodes(self) -> int:
        ...

    def neighbors(self, n: Any) -> Iterable[Any]:
        ...

    def __iter__(self) -> Iterable[Any]:
        ...


class IntegratorProtocol(Protocol):
    """Interface describing configurable nodal equation integrators."""

    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None,
        t: float | None,
        method: str | None,
        n_jobs: int | None,
    ) -> None:
        ...


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

class _SelectorLifecycle(Protocol):
    """Protocol describing the selector lifecycle supported by the runtime."""

    def __call__(self, graph: TNFRGraph, node: NodeId) -> Glyph | str:
        ...

    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None:
        ...

    def select(self, graph: TNFRGraph, node: NodeId) -> Glyph | str:
        ...


GlyphSelector: TypeAlias = (
    Callable[[TNFRGraph, NodeId], Glyph | str] | _SelectorLifecycle
)
#: Selector callable or object returning the glyph to apply for ``NodeId``.

SelectorPreselectionMetrics: TypeAlias = Mapping[Any, SelectorMetrics]
#: Mapping of nodes to their normalised selector metrics.

SelectorPreselectionChoices: TypeAlias = Mapping[Any, Glyph | str]
#: Mapping of nodes to their preferred glyph choices prior to grammar filters.

SelectorPreselectionPayload: TypeAlias = tuple[
    SelectorPreselectionMetrics,
    SelectorPreselectionChoices,
]
#: Tuple grouping selector metrics and base decisions for preselection steps.

TraceFieldFn: TypeAlias = Callable[[TNFRGraph], "TraceMetadata"]
#: Callable producing :class:`tnfr.trace.TraceMetadata` from a :data:`TNFRGraph`.

TraceFieldMap: TypeAlias = Mapping[str, "TraceFieldFn"]
#: Mapping of trace field names to their producers for a given phase.

TraceFieldRegistry: TypeAlias = dict[str, dict[str, "TraceFieldFn"]]
#: Registry grouping trace field producers by capture phase.

HistoryState: TypeAlias = _HistoryDict | dict[str, Any]
#: History container used to accumulate glyph metrics and logs for the graph.

TraceCallback: TypeAlias = Callable[[TNFRGraph, dict[str, Any]], None]
#: Callback signature used by :func:`tnfr.trace.register_trace`.

DiagnosisNodeData: TypeAlias = Mapping[str, Any]
#: Raw nodal measurement payload used prior to computing diagnostics.

DiagnosisSharedState: TypeAlias = Mapping[str, Any]
#: Shared read-only state propagated to diagnosis workers.

DiagnosisPayload: TypeAlias = dict[str, Any]
#: Structured diagnostics exported for a single node.

DiagnosisResult: TypeAlias = tuple[NodeId, DiagnosisPayload]
#: Node identifier paired with its :data:`DiagnosisPayload`.

DiagnosisPayloadChunk: TypeAlias = list[DiagnosisNodeData]
#: Chunk of nodal payloads processed together by diagnosis workers.

DiagnosisResultList: TypeAlias = list[DiagnosisResult]
#: Collection of diagnosis results matching worker output shape.

DnfrCacheVectors: TypeAlias = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]
"""Tuple grouping cached NumPy vectors for θ, EPI, νf and trigonometric projections."""

DnfrVectorMap: TypeAlias = dict[str, np.ndarray | None]
"""Mapping of TNFR state aliases to their NumPy buffers synchronized from lists."""

NeighborStats: TypeAlias = tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float] | None,
    Sequence[float] | None,
    Sequence[float] | None,
]
"""Bundle of neighbour accumulators for cosine, sine, EPI, νf and topology totals."""
