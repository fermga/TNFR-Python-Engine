from __future__ import annotations

from collections.abc import Hashable, Iterable, MutableMapping
from typing import Any, Callable, Optional, Protocol, SupportsFloat, TypeVar

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

T = TypeVar("T")

__all__ = ("NodeNX", "NodeProtocol", "add_edge")


class AttrSpec:
    aliases: tuple[str, ...]
    default: Any
    getter: Callable[[MutableMapping[str, Any], tuple[str, ...], Any], Any]
    setter: Callable[..., None]
    to_python: Callable[[Any], Any]
    to_storage: Callable[[Any], Any]
    use_graph_setter: bool

    def build_property(self) -> property: ...


ALIAS_EPI: tuple[str, ...]
ALIAS_VF: tuple[str, ...]
ALIAS_THETA: tuple[str, ...]
ALIAS_SI: tuple[str, ...]
ALIAS_EPI_KIND: tuple[str, ...]
ALIAS_DNFR: tuple[str, ...]
ALIAS_D2EPI: tuple[str, ...]

ATTR_SPECS: dict[str, AttrSpec]


def _add_edge_common(
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
) -> Optional[CouplingWeight]: ...


def add_edge(
    graph: TNFRGraph,
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
    overwrite: bool = ...,
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

    def _glyph_storage(self) -> MutableMapping[str, object]: ...

    def has_edge(self, other: NodeProtocol) -> bool: ...

    def add_edge(
        self,
        other: NodeProtocol,
        weight: CouplingWeight,
        *,
        overwrite: bool = ...,
    ) -> None: ...

    def offset(self) -> int: ...

    def all_nodes(self) -> Iterable[NodeProtocol]: ...


class NodeNX(NodeProtocol):
    G: TNFRGraph
    n: NodeId
    graph: MutableMapping[str, Any]

    def __init__(self, G: TNFRGraph, n: NodeId) -> None: ...

    @classmethod
    def from_graph(cls, G: TNFRGraph, n: NodeId) -> "NodeNX": ...

    def _glyph_storage(self) -> MutableMapping[str, Any]: ...

    @property
    def EPI(self) -> EPIValue: ...

    @EPI.setter
    def EPI(self, value: EPIValue) -> None: ...

    @property
    def vf(self) -> StructuralFrequency: ...

    @vf.setter
    def vf(self, value: StructuralFrequency) -> None: ...

    @property
    def theta(self) -> Phase: ...

    @theta.setter
    def theta(self, value: Phase) -> None: ...

    @property
    def Si(self) -> SenseIndex: ...

    @Si.setter
    def Si(self, value: SenseIndex) -> None: ...

    @property
    def epi_kind(self) -> str: ...

    @epi_kind.setter
    def epi_kind(self, value: str) -> None: ...

    @property
    def dnfr(self) -> DeltaNFR: ...

    @dnfr.setter
    def dnfr(self, value: DeltaNFR) -> None: ...

    @property
    def d2EPI(self) -> SecondDerivativeEPI: ...

    @d2EPI.setter
    def d2EPI(self, value: SecondDerivativeEPI) -> None: ...

    def neighbors(self) -> Iterable[NodeId]: ...

    def has_edge(self, other: NodeProtocol) -> bool: ...

    def add_edge(
        self,
        other: NodeProtocol,
        weight: CouplingWeight,
        *,
        overwrite: bool = ...,
    ) -> None: ...

    def offset(self) -> int: ...

    def all_nodes(self) -> Iterable[NodeProtocol]: ...


