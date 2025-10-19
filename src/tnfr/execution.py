"""Execution helpers for canonical TNFR programs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, TypeAlias, cast

import networkx as nx  # networkx is used at runtime

from .utils import MAX_MATERIALIZE_DEFAULT, ensure_collection, is_non_string_sequence
from .constants import get_param
from .dynamics import step
from .flatten import _flatten
from .glyph_history import ensure_history
from .validation.grammar import apply_glyph_with_grammar
from .tokens import Node as TokenNode
from .tokens import OpTag, TARGET, THOL, WAIT, Token
from .types import Glyph

Graph: TypeAlias = nx.Graph
Node: TypeAlias = TokenNode
AdvanceFn = Callable[[Graph], None]
TraceEntry = dict[str, Any]
ProgramTrace: TypeAlias = deque[TraceEntry]
HandlerFn = Callable[
    [Graph, Any, Optional[Sequence[Node]], ProgramTrace, AdvanceFn],
    Optional[Sequence[Node]],
]

__all__ = [
    "AdvanceFn",
    "CANONICAL_PRESET_NAME",
    "CANONICAL_PROGRAM_TOKENS",
    "HANDLERS",
    "_apply_glyph_to_targets",
    "_record_trace",
    "compile_sequence",
    "basic_canonical_example",
    "block",
    "play",
    "seq",
    "target",
    "wait",
]


CANONICAL_PRESET_NAME = "ejemplo_canonico"
CANONICAL_PROGRAM_TOKENS: tuple[Token, ...] = (
    Glyph.SHA,
    Glyph.AL,
    Glyph.RA,
    Glyph.ZHIR,
    Glyph.NUL,
    Glyph.THOL,
)


def _window(G: Graph) -> int:
    return int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))


def _apply_glyph_to_targets(
    G: Graph, g: Glyph | str, nodes: Optional[Iterable[Node]] = None
) -> None:
    """Apply ``g`` to ``nodes`` (or all nodes) respecting the grammar."""

    nodes_iter = G.nodes() if nodes is None else nodes
    w = _window(G)
    apply_glyph_with_grammar(G, nodes_iter, g, w)


def _advance(G: Graph, step_fn: AdvanceFn) -> None:
    step_fn(G)


def _record_trace(trace: ProgramTrace, G: Graph, op: OpTag, **data: Any) -> None:
    trace.append({"t": float(G.graph.get("_t", 0.0)), "op": op.name, **data})


def _advance_and_record(
    G: Graph,
    trace: ProgramTrace,
    label: OpTag,
    step_fn: AdvanceFn,
    *,
    times: int = 1,
    **data: Any,
) -> None:
    for _ in range(times):
        _advance(G, step_fn)
    _record_trace(trace, G, label, **data)


def _handle_target(
    G: Graph,
    payload: TARGET,
    _curr_target: Optional[Sequence[Node]],
    trace: ProgramTrace,
    _step_fn: AdvanceFn,
) -> Sequence[Node]:
    """Handle a ``TARGET`` token and return the active node set."""

    nodes_src = G.nodes() if payload.nodes is None else payload.nodes
    nodes = ensure_collection(nodes_src, max_materialize=None)
    if is_non_string_sequence(nodes):
        curr_target = cast(Sequence[Node], nodes)
    else:
        curr_target = tuple(nodes)
    _record_trace(trace, G, OpTag.TARGET, n=len(curr_target))
    return curr_target


def _handle_wait(
    G: Graph,
    steps: int,
    curr_target: Optional[Sequence[Node]],
    trace: ProgramTrace,
    step_fn: AdvanceFn,
) -> Optional[Sequence[Node]]:
    _advance_and_record(G, trace, OpTag.WAIT, step_fn, times=steps, k=steps)
    return curr_target


def _handle_glyph(
    G: Graph,
    g: Glyph | str,
    curr_target: Optional[Sequence[Node]],
    trace: ProgramTrace,
    step_fn: AdvanceFn,
    label: OpTag = OpTag.GLYPH,
) -> Optional[Sequence[Node]]:
    _apply_glyph_to_targets(G, g, curr_target)
    _advance_and_record(G, trace, label, step_fn, g=g)
    return curr_target


def _handle_thol(
    G: Graph,
    g: Glyph | str | None,
    curr_target: Optional[Sequence[Node]],
    trace: ProgramTrace,
    step_fn: AdvanceFn,
) -> Optional[Sequence[Node]]:
    return _handle_glyph(
        G, g or Glyph.THOL.value, curr_target, trace, step_fn, label=OpTag.THOL
    )


HANDLERS: dict[OpTag, HandlerFn] = {
    OpTag.TARGET: _handle_target,
    OpTag.WAIT: _handle_wait,
    OpTag.GLYPH: _handle_glyph,
    OpTag.THOL: _handle_thol,
}


def play(
    G: Graph, sequence: Sequence[Token], step_fn: Optional[AdvanceFn] = None
) -> None:
    """Execute a canonical sequence on graph ``G``."""

    step_fn = step_fn or step

    curr_target: Optional[Sequence[Node]] = None

    history = ensure_history(G)
    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    trace_obj = history.get("program_trace")
    trace: ProgramTrace
    if not isinstance(trace_obj, deque) or trace_obj.maxlen != maxlen:
        trace = cast(ProgramTrace, deque(trace_obj or [], maxlen=maxlen))
        history["program_trace"] = trace
    else:
        trace = cast(ProgramTrace, trace_obj)

    for op, payload in _flatten(sequence):
        handler: HandlerFn | None = HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown operation: {op}")
        curr_target = handler(G, payload, curr_target, trace, step_fn)


def compile_sequence(
    sequence: Iterable[Token] | Sequence[Token] | Any,
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
) -> list[tuple[OpTag, Any]]:
    """Return the operations executed by :func:`play` for ``sequence``."""

    return _flatten(sequence, max_materialize=max_materialize)


def seq(*tokens: Token) -> list[Token]:
    return list(tokens)


def block(
    *tokens: Token, repeat: int = 1, close: Optional[Glyph] = None
) -> THOL:
    return THOL(body=list(tokens), repeat=repeat, force_close=close)


def target(nodes: Optional[Iterable[Node]] = None) -> TARGET:
    return TARGET(nodes=nodes)


def wait(steps: int = 1) -> WAIT:
    return WAIT(steps=max(1, int(steps)))


def basic_canonical_example() -> list[Token]:
    """Reference canonical sequence.

    Returns a copy of the canonical preset tokens to keep CLI defaults aligned
    with :func:`tnfr.config.presets.get_preset`.
    """

    return list(CANONICAL_PROGRAM_TOKENS)
