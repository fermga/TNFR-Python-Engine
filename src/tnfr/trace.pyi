from typing import Any, Callable, NamedTuple, TypedDict
from collections.abc import Iterable, Mapping

from .types import TNFRGraph, TraceFieldFn, TraceFieldMap, TraceFieldRegistry

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...


class CallbackSpec(NamedTuple):
    name: str | None
    func: Callable[..., Any]


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


kuramoto_R_psi: Callable[[TNFRGraph], tuple[float, float]]
TRACE_FIELDS: TraceFieldRegistry

def _callback_names(
    callbacks: Mapping[str, CallbackSpec] | Iterable[CallbackSpec],
) -> list[str]: ...

def mapping_field(G: TNFRGraph, graph_key: str, out_key: str) -> TraceMetadata: ...

def _trace_capture(G: TNFRGraph, phase: str, fields: TraceFieldMap) -> None: ...

def register_trace_field(phase: str, name: str, func: TraceFieldFn) -> None: ...

def gamma_field(G: TNFRGraph) -> TraceMetadata: ...

def grammar_field(G: TNFRGraph) -> TraceMetadata: ...

def dnfr_weights_field(G: TNFRGraph) -> TraceMetadata: ...

def selector_field(G: TNFRGraph) -> TraceMetadata: ...

def si_weights_field(G: TNFRGraph) -> TraceMetadata: ...

def callbacks_field(G: TNFRGraph) -> TraceMetadata: ...

def thol_state_field(G: TNFRGraph) -> TraceMetadata: ...

def kuramoto_field(G: TNFRGraph) -> TraceMetadata: ...

def sigma_field(G: TNFRGraph) -> TraceMetadata: ...

def glyph_counts_field(G: TNFRGraph) -> TraceMetadata: ...

def register_trace(G: TNFRGraph) -> None: ...
