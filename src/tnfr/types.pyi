from typing import Any, Callable, Iterable, Protocol, TypeAlias
from collections.abc import Hashable, Mapping

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...

TNFRGraph: TypeAlias = Any
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

class _DeltaNFRHookProtocol(Protocol):
    def __call__(self, graph: TNFRGraph, /, *args: Any, **kwargs: Any) -> None: ...

DeltaNFRHook: TypeAlias = _DeltaNFRHookProtocol

class GraphLike(Protocol):
    graph: dict[str, Any]

    def nodes(self, data: bool = ...) -> Iterable[Any]: ...
    def number_of_nodes(self) -> int: ...
    def neighbors(self, n: Any) -> Iterable[Any]: ...
    def __iter__(self) -> Iterable[Any]: ...

class Glyph(str): ...

GlyphLoadDistribution: TypeAlias = dict[Glyph | str, float]
TraceFieldFn: TypeAlias = Callable[[TNFRGraph], Any]
TraceFieldMap: TypeAlias = Mapping[str, TraceFieldFn]
TraceFieldRegistry: TypeAlias = dict[str, dict[str, TraceFieldFn]]
TraceCallback: TypeAlias = Callable[[TNFRGraph, dict[str, Any]], None]
