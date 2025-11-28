"""Execution helpers for canonical TNFR programs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, cast

from ._compat import TypeAlias
from .constants import get_param
from .dynamics import step
from .flatten import _flatten
from .glyph_history import ensure_history
from .tokens import TARGET, THOL, WAIT, OpTag, Token
from .types import Glyph, NodeId, TNFRGraph
from .utils import MAX_MATERIALIZE_DEFAULT, ensure_collection, is_non_string_sequence
from .validation import apply_glyph_with_grammar

AdvanceFn = Callable[[TNFRGraph], None]
TraceEntry = dict[str, Any]
ProgramTrace: TypeAlias = deque[TraceEntry]
HandlerFn = Callable[
    [TNFRGraph, Any, Optional[Sequence[NodeId]], ProgramTrace, AdvanceFn],
    Optional[Sequence[NodeId]],
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

CANONICAL_PRESET_NAME = "canonical_example"
CANONICAL_PROGRAM_TOKENS: tuple[Token, ...] = (
    Glyph.SHA,  # silence - initial stabilization
    Glyph.AL,  # emission - initiate pattern
    Glyph.RA,  # reception - capture information
    Glyph.OZ,  # dissonance - required before mutation (grammar rule)
    Glyph.ZHIR,  # mutation - phase change
    Glyph.NUL,  # contraction - compress structure
    Glyph.THOL,  # self_organization - recursive reorganization
)


def _window(G: TNFRGraph) -> int:
    return int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))


def _apply_glyph_to_targets(
    G: TNFRGraph, g: Glyph | str, nodes: Optional[Iterable[NodeId]] = None
) -> None:
    """Apply ``g`` to ``nodes`` (or all nodes) respecting the grammar."""

    nodes_iter = G.nodes() if nodes is None else nodes
    w = _window(G)
    apply_glyph_with_grammar(G, nodes_iter, g, w)


def _advance(G: TNFRGraph, step_fn: AdvanceFn) -> None:
    step_fn(G)


def _record_trace(trace: ProgramTrace, G: TNFRGraph, op: OpTag, **data: Any) -> None:
    """Append an operation snapshot to ``trace`` using graph time metadata."""

    trace.append({"t": float(G.graph.get("_t", 0.0)), "op": op.name, **data})


def _advance_and_record(
    G: TNFRGraph,
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
    G: TNFRGraph,
    payload: TARGET,
    _curr_target: Optional[Sequence[NodeId]],
    trace: ProgramTrace,
    _step_fn: AdvanceFn,
) -> Sequence[NodeId]:
    """Handle a ``TARGET`` token and return the active node set."""

    nodes_src = G.nodes() if payload.nodes is None else payload.nodes
    nodes = ensure_collection(nodes_src, max_materialize=None)
    if is_non_string_sequence(nodes):
        curr_target = cast(Sequence[NodeId], nodes)
    else:
        curr_target = tuple(nodes)
    _record_trace(trace, G, OpTag.TARGET, n=len(curr_target))
    return curr_target


def _handle_wait(
    G: TNFRGraph,
    steps: int,
    curr_target: Optional[Sequence[NodeId]],
    trace: ProgramTrace,
    step_fn: AdvanceFn,
) -> Optional[Sequence[NodeId]]:
    _advance_and_record(G, trace, OpTag.WAIT, step_fn, times=steps, k=steps)
    return curr_target


def _handle_glyph(
    G: TNFRGraph,
    g: Glyph | str,
    curr_target: Optional[Sequence[NodeId]],
    trace: ProgramTrace,
    step_fn: AdvanceFn,
    label: OpTag = OpTag.GLYPH,
) -> Optional[Sequence[NodeId]]:
    _apply_glyph_to_targets(G, g, curr_target)
    _advance_and_record(G, trace, label, step_fn, g=g)
    return curr_target


def _handle_thol(
    G: TNFRGraph,
    g: Glyph | str | None,
    curr_target: Optional[Sequence[NodeId]],
    trace: ProgramTrace,
    step_fn: AdvanceFn,
) -> Optional[Sequence[NodeId]]:
    return _handle_glyph(G, g or Glyph.THOL.value, curr_target, trace, step_fn, label=OpTag.THOL)


HANDLERS: dict[OpTag, HandlerFn] = {
    OpTag.TARGET: _handle_target,
    OpTag.WAIT: _handle_wait,
    OpTag.GLYPH: _handle_glyph,
    OpTag.THOL: _handle_thol,
}


def play(G: TNFRGraph, sequence: Sequence[Token], step_fn: Optional[AdvanceFn] = None) -> None:
    """Execute a canonical sequence on graph ``G``."""

    step_fn = step_fn or step

    curr_target: Optional[Sequence[NodeId]] = None

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
    """Return a mutable list of ``tokens`` for explicit sequence editing."""

    return list(tokens)


def block(*tokens: Token, repeat: int = 1, close: Optional[Glyph] = None) -> THOL:
    """Build a THOL block with optional repetition and forced closure."""

    return THOL(body=list(tokens), repeat=repeat, force_close=close)


def target(nodes: Optional[Iterable[NodeId]] = None) -> TARGET:
    """Return a TARGET token selecting ``nodes`` (defaults to all nodes)."""

    return TARGET(nodes=nodes)


def wait(steps: int = 1) -> WAIT:
    """Return a WAIT token forcing ``steps`` structural updates before resuming."""

    return WAIT(steps=max(1, int(steps)))


def basic_canonical_example() -> list[Token]:
    """Return the canonical preset sequence.

    Returns a copy of the canonical preset tokens to keep CLI defaults aligned
    with :func:`tnfr.config.presets.get_preset`.
    """

    return list(CANONICAL_PROGRAM_TOKENS)
