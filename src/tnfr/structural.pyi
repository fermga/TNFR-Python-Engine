from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable

from .operators.definitions import (
    Operator,
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Coupling,
    Resonance,
    Silence,
    Expansion,
    Contraction,
    SelfOrganization,
    Mutation,
    Transition,
    Recursivity,
)

if TYPE_CHECKING:
    import networkx as nx

__all__: tuple[str, ...]


def __getattr__(name: str) -> Any: ...


def create_nfr(
    name: str,
    *,
    epi: float = ...,
    vf: float = ...,
    theta: float = ...,
    graph: "nx.Graph" | None = ...,
    dnfr_hook: Callable[..., None] = ...,
) -> tuple["nx.Graph", str]: ...


OPERATORS: dict[str, Operator]


def validate_sequence(names: Iterable[str]) -> tuple[bool, str]: ...


def run_sequence(G: "nx.Graph", node: Hashable, ops: Iterable[Operator]) -> None: ...
