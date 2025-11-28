"""Type definitions and protocols shared across the engine."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from enum import Enum
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from numbers import Real

from ._compat import TypeAlias

if TYPE_CHECKING:
    from .mathematics import BEPIElement

try:  # pragma: no cover - optional dependency for typing only
    import numpy as np
except Exception:  # pragma: no cover - graceful fallback when NumPy is missing
    np = SimpleNamespace(ndarray=Any, float64=float)  # type: ignore[assignment]

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
    "ValidatorFunc",
    "NodeId",
    "Node",
    "GammaSpec",
    "EPIValue",
    "BEPIProtocol",
    "ensure_bepi",
    "serialize_bepi",
    "serialize_bepi_json",
    "deserialize_bepi_json",
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
    "RemeshMeta",
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
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing hook
    import networkx as nx

    from .glyph_history import HistoryDict as _HistoryDict
    from .tokens import Token as _Token

    TNFRGraph: TypeAlias = nx.Graph
else:  # pragma: no cover - runtime fallback without networkx
    TNFRGraph: TypeAlias = Any
    _HistoryDict = Any  # type: ignore[assignment]
    _Token = Any  # type: ignore[assignment]
#: Graph container storing TNFR nodes, edges and their coherence telemetry.

if TYPE_CHECKING:
    FloatArray: TypeAlias = npt.NDArray[np.float64]
    FloatMatrix: TypeAlias = npt.NDArray[np.float64]
else:  # pragma: no cover - runtime fallback without NumPy
    FloatArray: TypeAlias = Any
    FloatMatrix: TypeAlias = Any

Graph: TypeAlias = TNFRGraph
#: Backwards-compatible alias for :data:`TNFRGraph`.

ValidatorFunc: TypeAlias = Callable[[TNFRGraph], None]
"""Callable signature enforced by graph validation hooks."""

NodeId: TypeAlias = Hashable
#: Hashable identifier for a coherent TNFR node.

Node: TypeAlias = NodeId
#: Backwards-compatible alias for :data:`NodeId`.

NodeInitAttrMap: TypeAlias = MutableMapping[str, float]
#: Mutable mapping storing scalar node attributes during initialization.

NodeAttrMap: TypeAlias = Mapping[str, Any]
#: Read-only mapping exposing resolved node attributes during execution.

GammaSpec: TypeAlias = Mapping[str, Any]
#: Mapping describing Γ evaluation parameters for a node or graph.


@runtime_checkable
class BEPIProtocol(Protocol):
    """Structural contract describing BEPI-compatible values."""

    f_continuous: Any
    a_discrete: Any
    x_grid: Any

    def direct_sum(self, other: Any) -> Any: ...

    def tensor(self, vector: Sequence[complex] | np.ndarray) -> np.ndarray: ...

    def adjoint(self) -> Any: ...

    def compose(
        self,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        spectral_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Any: ...


EPIValue: TypeAlias = BEPIProtocol
#: BEPI Primary Information Structure carried by a node.

ZERO_BEPI_STORAGE: dict[str, tuple[complex, ...] | tuple[float, ...]] = {
    "continuous": (0j, 0j),
    "discrete": (0j, 0j),
    "grid": (0.0, 1.0),
}
"""Canonical zero element used as fallback when EPI data is missing."""


def _is_scalar(value: Any) -> bool:
    scalar_types: tuple[type[Any], ...]
    np_scalar = getattr(np, "generic", None)
    if np_scalar is None:
        scalar_types = (int, float, complex, Real)
    else:
        scalar_types = (int, float, complex, Real, np_scalar)
    return isinstance(value, scalar_types)


def ensure_bepi(value: Any) -> "BEPIElement":
    """Normalise arbitrary inputs into a :class:`~tnfr.mathematics.BEPIElement`."""

    from .mathematics import BEPIElement as _BEPIElement

    if isinstance(value, _BEPIElement):
        return value
    if _is_scalar(value):
        scalar = complex(value)
        return _BEPIElement((scalar, scalar), (scalar, scalar), (0.0, 1.0))
    if isinstance(value, Mapping):
        try:
            continuous = value["continuous"]
            discrete = value["discrete"]
            grid = value["grid"]
        except KeyError as exc:  # pragma: no cover - defensive
            missing = exc.args[0]
            raise ValueError(f"Missing '{missing}' key for BEPI serialization.") from exc
        return _BEPIElement(continuous, discrete, grid)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 3:
            raise ValueError("Sequential BEPI representations must contain 3 elements.")
        continuous, discrete, grid = value
        return _BEPIElement(continuous, discrete, grid)
    raise TypeError(f"Unsupported BEPI value type: {type(value)!r}")


def serialize_bepi(value: Any) -> dict[str, tuple[complex, ...] | tuple[float, ...]]:
    """Serialise a BEPI element into canonical ``continuous/discrete/grid`` tuples."""

    element = ensure_bepi(value)
    continuous = tuple(complex(v) for v in element.f_continuous.tolist())
    discrete = tuple(complex(v) for v in element.a_discrete.tolist())
    grid = tuple(float(v) for v in element.x_grid.tolist())
    return {"continuous": continuous, "discrete": discrete, "grid": grid}


def serialize_bepi_json(value: Any) -> dict[str, list[dict[str, float]] | list[float]]:
    """Serialize a BEPI element into JSON-compatible format.

    Complex numbers are represented as dicts with 'real' and 'imag' keys.
    This enables full JSON/YAML serialization while preserving structural coherence.

    Parameters
    ----------
    value : Any
        A BEPIElement instance or value convertible to one.

    Returns
    -------
    dict
        Dictionary with 'continuous', 'discrete', and 'grid' keys, where
        complex values are represented as ``{"real": float, "imag": float}``
        and grid values remain as floats.

    Examples
    --------
    >>> from tnfr.mathematics import BEPIElement
    >>> bepi = BEPIElement((1+2j, 3+0j), (4+5j,), (0.0, 1.0))
    >>> serialize_bepi_json(bepi)  # doctest: +SKIP
    {
        'continuous': [{'real': 1.0, 'imag': 2.0}, {'real': 3.0, 'imag': 0.0}],
        'discrete': [{'real': 4.0, 'imag': 5.0}],
        'grid': [0.0, 1.0]
    }
    """
    element = ensure_bepi(value)

    def _complex_to_dict(c: complex) -> dict[str, float]:
        return {"real": float(c.real), "imag": float(c.imag)}

    continuous = [_complex_to_dict(v) for v in element.f_continuous.tolist()]
    discrete = [_complex_to_dict(v) for v in element.a_discrete.tolist()]
    grid = [float(v) for v in element.x_grid.tolist()]

    return {"continuous": continuous, "discrete": discrete, "grid": grid}


def deserialize_bepi_json(
    data: dict[str, list[dict[str, float]] | list[float]],
) -> "BEPIElement":
    """Deserialize a BEPI element from JSON-compatible format.

    Reconstructs complex numbers from dicts with 'real' and 'imag' keys.

    Parameters
    ----------
    data : dict
        Dictionary with 'continuous', 'discrete', and 'grid' keys in JSON format.
        The 'continuous' and 'discrete' values should be lists of dicts with
        'real' and 'imag' keys, while 'grid' should be a list of floats.

    Returns
    -------
    BEPIElement
        Reconstructed BEPI element with validated structural integrity.

    Examples
    --------
    >>> data = {
    ...     'continuous': [{'real': 1.0, 'imag': 2.0}, {'real': 3.0, 'imag': 0.0}],
    ...     'discrete': [{'real': 4.0, 'imag': 5.0}],
    ...     'grid': [0.0, 1.0]
    ... }
    >>> bepi = deserialize_bepi_json(data)  # doctest: +SKIP
    """
    from .mathematics import BEPIElement as _BEPIElement

    def _dict_to_complex(d: dict[str, float] | float | complex) -> complex:
        if isinstance(d, dict):
            return complex(d["real"], d["imag"])
        return complex(d)

    continuous = [_dict_to_complex(v) for v in data["continuous"]]
    discrete = [_dict_to_complex(v) for v in data["discrete"]]
    grid = data["grid"]

    return _BEPIElement(continuous, discrete, grid)


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
    TNFRConfigScalar
    | TNFRConfigSequence
    | MutableMapping[str, "TNFRConfigValue"]
)
"""Permissible configuration entry for TNFR coherence defaults.

The alias captures the recursive structure used by TNFR defaults: scalars
express structural thresholds, booleans toggle operators, and nested
mappings
or sequences describe coherent parameter bundles such as γ grammars,
selector advice or trace capture lists.

Configuration dictionaries support the full
:class:`~collections.abc.MutableMapping` protocol, enabling dict-like
operations such as ``.get()``, ``__setitem__``,
and ``.update()`` for runtime configuration adjustments.
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


class SigmaTrace(TypedDict):
    """Time-aligned σ(t) trace exported alongside glyphograms."""

    t: list[float]
    sigma_x: list[float]
    sigma_y: list[float]
    mag: list[float]
    angle: list[float]


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


@runtime_checkable
class _DeltaNFRHookProtocol(Protocol):
    """Callable signature expected for ΔNFR update hooks.

    Hooks receive the graph instance and may expose optional keyword
    arguments such as ``n_jobs`` or cache controls. Additional positional
    arguments are reserved for future extensions and ignored by the core
    engine, keeping compatibility with user-provided hooks that only need the
    graph reference.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    hook implementations conform to the expected callable signature.
    """

    def __call__(
        self,
        graph: TNFRGraph,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...


DeltaNFRHook: TypeAlias = _DeltaNFRHookProtocol
#: Callable hook invoked to compute ΔNFR for a :data:`TNFRGraph`.


@runtime_checkable
class _NodeViewLike(Protocol):
    """Subset of :class:`networkx.NodeView` behaviour relied on by TNFR.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    node view implementations conform to the expected interface.
    """

    def __iter__(self) -> Iterable[Any]: ...

    def __call__(self, data: bool = ...) -> Iterable[Any]: ...

    def __getitem__(self, node: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class _EdgeViewLike(Protocol):
    """Subset of :class:`networkx.EdgeView` behaviour relied on by TNFR.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    edge view implementations conform to the expected interface.
    """

    def __iter__(self) -> Iterable[Any]: ...

    def __call__(self, data: bool = ...) -> Iterable[Any]: ...


@runtime_checkable
class GraphLike(Protocol):
    """Protocol describing graph objects consumed by TNFR subsystems.

    Graph-like containers must expose cached-property style ``nodes`` and
    ``edges`` views compatible with :mod:`networkx`, a ``neighbors`` iterator,
    ``number_of_nodes`` introspection and a metadata mapping via ``.graph``.
    Metrics, cache utilities and CLI diagnostics assume this interface when
    traversing structural coherence data.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    graph implementations conform to the expected TNFR graph interface.
    """

    graph: MutableMapping[str, Any]
    nodes: _NodeViewLike
    edges: _EdgeViewLike

    def number_of_nodes(self) -> int:
        """Return the total number of coherent nodes in the graph."""

        ...

    def neighbors(self, n: Any) -> Iterable[Any]:
        """Yield adjacent nodes coupled to ``n`` within the structure."""

        ...

    def __getitem__(self, node: Any) -> MutableMapping[Any, Any]:
        """Expose adjacency metadata for ``node`` using ``G[node]``
        semantics."""

        ...

    def __iter__(self) -> Iterable[Any]:
        """Iterate over nodes to allow direct structural traversals."""

        ...


@runtime_checkable
class IntegratorProtocol(Protocol):
    """Interface describing configurable nodal equation integrators.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    integrator implementations conform to the expected interface.
    """

    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None,
        t: float | None,
        method: str | None,
        n_jobs: int | None,
    ) -> None:
        """Advance the nodal equation for ``graph`` using integrator
        configuration."""

        ...


class Glyph(str, Enum):
    """Canonical TNFR structural symbols (glyphs).

    Glyphs are the structural symbols (AL, EN, IL, etc.) that represent the
    application of structural operators. Each structural operator (Emission,
    Reception, Coherence, etc.) is associated with a specific glyph symbol.

    For public-facing documentation and APIs, refer to these by their
    structural operator names rather than the internal glyph codes.
    """

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
"""Structural operator symbol (glyph) identifier accepted by selector
pipelines and grammars."""

GlyphLoadDistribution: TypeAlias = dict[Glyph | str, float]
"""Normalised load proportions keyed by structural operator symbol
(glyph) or aggregate labels."""


@runtime_checkable
class _SelectorLifecycle(Protocol):
    """Protocol describing the selector lifecycle supported by the runtime.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    selector implementations conform to the expected lifecycle interface.
    """

    def __call__(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...

    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None: ...

    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...


GlyphSelector: TypeAlias = (
    Callable[[TNFRGraph, NodeId], GlyphCode] | _SelectorLifecycle
)
"""Selector callable or object returning the structural operator symbol
(glyph) to apply for a node."""

SelectorPreselectionMetrics: TypeAlias = Mapping[Any, SelectorMetrics]
"""Mapping of nodes to their normalised selector metrics."""

SelectorPreselectionChoices: TypeAlias = Mapping[Any, Glyph | str]
"""Mapping of nodes to their preferred structural operator symbol
(glyph) prior to grammar filters."""

SelectorPreselectionPayload: TypeAlias = tuple[
    SelectorPreselectionMetrics,
    SelectorPreselectionChoices,
]
#: Tuple grouping selector metrics and base decisions for preselection steps.

TraceFieldFn: TypeAlias = Callable[[TNFRGraph], "TraceMetadata"]
#: Callable producing :class:`tnfr.trace.TraceMetadata` from a
#: :data:`TNFRGraph`.

TraceFieldMap: TypeAlias = Mapping[str, "TraceFieldFn"]
#: Mapping of trace field names to their producers for a given phase.

TraceFieldRegistry: TypeAlias = dict[str, dict[str, "TraceFieldFn"]]
#: Registry grouping trace field producers by capture phase.


class TraceMetadata(TypedDict, total=False):
    """Metadata captured by trace field producers across phases."""

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
    """Trace metadata snapshot recorded in TNFR history."""

    t: float
    phase: str


HistoryState: TypeAlias = _HistoryDict | dict[str, Any]
#: History container used to accumulate glyph metrics and logs for the graph.


class CallbackError(TypedDict):
    """Metadata captured for a failed callback invocation."""

    event: str
    step: int | None
    error: str
    traceback: str
    fn: str
    name: str | None


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
"""Tuple grouping cached NumPy vectors for θ, EPI, νf and trigonometric
projections."""

DnfrVectorMap: TypeAlias = dict[str, np.ndarray | None]
"""Mapping of TNFR state aliases to their NumPy buffers synchronized
from lists."""

NeighborStats: TypeAlias = tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float] | None,
    Sequence[float] | None,
    Sequence[float] | None,
]
"""Bundle of neighbour accumulators for cosine, sine, EPI, νf and
topology totals."""

GlyphogramRow: TypeAlias = MutableMapping[str, float]
"""Row exported by glyph timing summaries."""

GlyphTimingTotals: TypeAlias = MutableMapping[str, float]
"""Aggregate glyph timing totals keyed by glyph code."""

GlyphTimingByNode: TypeAlias = MutableMapping[
    Any, MutableMapping[str, MutableSequence[float]]
]
"""Glyph timing segments stored per node during audits."""

GlyphCounts: TypeAlias = Mapping[str, int]
"""Glyph occurrence counters keyed by glyph code."""

GlyphMetricsHistoryValue: TypeAlias = (
    MutableMapping[Any, Any] | MutableSequence[Any]
)
"""Flexible container used by glyph history accumulators."""

GlyphMetricsHistory: TypeAlias = MutableMapping[str, GlyphMetricsHistoryValue]
"""History map storing glyph metrics by identifier."""

MetricsListHistory: TypeAlias = MutableMapping[str, list[Any]]
"""Mapping associating glyph metric identifiers with time series."""


class RemeshMeta(TypedDict, total=False):
    """Event metadata persisted after applying REMESH coherence operators."""

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
    """Container for broadcasting Wij coherence components to worker pools."""

    epi_vals: Sequence[float]
    vf_vals: Sequence[float]
    si_vals: Sequence[float]
    cos_vals: Sequence[float]
    sin_vals: Sequence[float]
    weights: tuple[float, float, float, float]
    epi_range: float
    vf_range: float
