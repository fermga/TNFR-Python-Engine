from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Hashable

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
from .operators.compat import (
    Operador,
    Emision,
    Recepcion,
    Coherencia,
    Disonancia,
    Acoplamiento,
    Resonancia,
    Silencio,
    Contraccion,
    Autoorganizacion,
    Mutacion,
    Transicion,
    Recursividad as RecursividadLegacy,
)

Recursividad = RecursividadLegacy

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


OPERADORES: dict[str, Operator]


def validate_sequence(names: Iterable[str]) -> tuple[bool, str]: ...


def run_sequence(G: "nx.Graph", node: Hashable, ops: Iterable[Operator]) -> None: ...
