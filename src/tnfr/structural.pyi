from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Hashable

from .operators.definitions import (
    Acoplamiento,
    Autoorganizacion,
    Coherencia,
    Contraccion,
    Disonancia,
    Emision,
    Expansion,
    Mutacion,
    Operador,
    Recepcion,
    Recursividad,
    Resonancia,
    Silencio,
    Transicion,
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


OPERADORES: dict[str, Operador]


def validate_sequence(names: Iterable[str]) -> tuple[bool, str]: ...


def run_sequence(G: "nx.Graph", node: Hashable, ops: Iterable[Operador]) -> None: ...
