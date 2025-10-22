from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable

from .operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
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
