from __future__ import annotations

import numpy as np
from .mathematics import (
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    NFRValidator,
    StateProjector,
)
from .types import (
    CouplingWeight,
    DeltaNFR,
    EPIValue,
    NodeId,
    Phase,
    SecondDerivativeEPI,
    SenseIndex,
    StructuralFrequency,
    TNFRGraph,
)
from collections.abc import Hashable
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    MutableMapping,
    Protocol,
    Sequence,
    SupportsFloat,
    TypeVar,
)

__all__ = ["NodeNX", "NodeProtocol", "add_edge"]

T = TypeVar("T")

@dataclass(frozen=True)
class AttrSpec:
    aliases: tuple[str, ...]
    default: Any = ...
    getter: Callable[[MutableMapping[str, Any], tuple[str, ...], Any], Any] = ...
    setter: Callable[..., None] = ...
    to_python: Callable[[Any], Any] = ...
    to_storage: Callable[[Any], Any] = ...
    use_graph_setter: bool = ...
    def build_property(self) -> property: ...

def add_edge(
    graph: TNFRGraph,
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
    overwrite: bool = False,
) -> None: ...

class NodeProtocol(Protocol):
    EPI: EPIValue
    vf: StructuralFrequency
    theta: Phase
    Si: SenseIndex
    epi_kind: str
    dnfr: DeltaNFR
    d2EPI: SecondDerivativeEPI
    graph: MutableMapping[str, Any]
    def neighbors(self) -> Iterable[NodeProtocol | Hashable]: ...
    def has_edge(self, other: NodeProtocol) -> bool: ...
    def add_edge(
        self, other: NodeProtocol, weight: CouplingWeight, *, overwrite: bool = False
    ) -> None: ...
    def offset(self) -> int: ...
    def all_nodes(self) -> Iterable[NodeProtocol]: ...

class NodeNX(NodeProtocol):
    EPI: EPIValue
    vf: StructuralFrequency
    theta: Phase
    Si: SenseIndex
    epi_kind: str
    dnfr: DeltaNFR
    d2EPI: SecondDerivativeEPI
    G: TNFRGraph
    n: NodeId
    graph: MutableMapping[str, Any]
    state_projector: StateProjector
    enable_math_validation: bool
    hilbert_space: HilbertSpace
    coherence_operator: CoherenceOperator | None
    frequency_operator: FrequencyOperator | None
    coherence_threshold: float | None
    validator: NFRValidator | None
    rng: np.random.Generator | None
    def __init__(
        self,
        G: TNFRGraph,
        n: NodeId,
        *,
        state_projector: StateProjector | None = None,
        enable_math_validation: bool | None = None,
        hilbert_space: HilbertSpace | None = None,
        coherence_operator: CoherenceOperator | None = None,
        coherence_dim: int | None = None,
        coherence_spectrum: Sequence[float] | np.ndarray | None = None,
        coherence_c_min: float | None = None,
        frequency_operator: FrequencyOperator | None = None,
        frequency_matrix: Sequence[Sequence[complex]] | np.ndarray | None = None,
        coherence_threshold: float | None = None,
        validator: NFRValidator | None = None,
        rng: np.random.Generator | None = None,
    ) -> None: ...
    @classmethod
    def from_graph(cls, G: TNFRGraph, n: NodeId) -> NodeNX: ...
    def neighbors(self) -> Iterable[NodeId]: ...
    def has_edge(self, other: NodeProtocol) -> bool: ...
    def add_edge(
        self, other: NodeProtocol, weight: CouplingWeight, *, overwrite: bool = False
    ) -> None: ...
    def offset(self) -> int: ...
    def all_nodes(self) -> Iterable[NodeProtocol]: ...
    def run_sequence_with_validation(
        self,
        ops: Iterable[Callable[[TNFRGraph, NodeId], None]],
        *,
        projector: StateProjector | None = None,
        hilbert_space: HilbertSpace | None = None,
        coherence_operator: CoherenceOperator | None = None,
        coherence_dim: int | None = None,
        coherence_spectrum: Sequence[float] | np.ndarray | None = None,
        coherence_c_min: float | None = None,
        coherence_threshold: float | None = None,
        frequency_operator: FrequencyOperator | None = None,
        frequency_matrix: Sequence[Sequence[complex]] | np.ndarray | None = None,
        validator: NFRValidator | None = None,
        enforce_frequency_positivity: bool | None = None,
        enable_validation: bool | None = None,
        rng: np.random.Generator | None = None,
        log_metrics: bool = False,
    ) -> dict[str, Any]: ...
