from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional

from ._compat import TypeAlias
from .tokens import TARGET, THOL, WAIT, OpTag, Token
from .types import Glyph, NodeId, TNFRGraph

__all__: list[str]

def __getattr__(name: str) -> Any: ...

AdvanceFn = Callable[[TNFRGraph], None]
TraceEntry = dict[str, Any]
ProgramTrace: TypeAlias = deque[TraceEntry]
HandlerFn = Callable[
    [TNFRGraph, Any, Sequence[NodeId] | None, ProgramTrace, AdvanceFn],
    Sequence[NodeId] | None,
]

CANONICAL_PRESET_NAME: str
CANONICAL_PROGRAM_TOKENS: tuple[Token, ...]
HANDLERS: dict[OpTag, HandlerFn]

def _apply_glyph_to_targets(
    G: TNFRGraph, g: Glyph | str, nodes: Iterable[NodeId] | None = ...
) -> None: ...
def _record_trace(trace: ProgramTrace, G: TNFRGraph, op: OpTag, **data: Any) -> None: ...
def compile_sequence(
    sequence: Iterable[Token] | Sequence[Token] | Any,
    *,
    max_materialize: int | None = ...,
) -> list[tuple[OpTag, Any]]: ...
def play(G: TNFRGraph, sequence: Sequence[Token], step_fn: Optional[AdvanceFn] = ...) -> None: ...
def seq(*tokens: Token) -> list[Token]: ...
def block(*tokens: Token, repeat: int = ..., close: Glyph | None = ...) -> THOL: ...
def target(nodes: Iterable[NodeId] | None = ...) -> TARGET: ...
def wait(steps: int = ...) -> WAIT: ...
def basic_canonical_example() -> list[Token]: ...
