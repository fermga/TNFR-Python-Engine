"""Execution helpers for canonical TNFR programs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias, cast

from ._exact_time import finite_represented_real
from .constants import get_param
from .dynamics import step
from .flatten import _flatten
from .glyph_history import ensure_history
from .tokens import TARGET, THOL, WAIT, BlockBoundary, OpTag, Token, _program_count
from .types import Glyph, NodeId, TNFRGraph
from .utils import MAX_MATERIALIZE_DEFAULT, ensure_collection, is_non_string_sequence
from .validation import apply_glyph_with_grammar

AdvanceFn = Callable[[TNFRGraph], None]
TraceEntry = dict[str, Any]
ProgramTrace: TypeAlias = deque[TraceEntry]
HandlerFn = Callable[
    [TNFRGraph, Any, Sequence[NodeId] | None, ProgramTrace, AdvanceFn],
    Sequence[NodeId] | None,
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
    Glyph.AL,
    Glyph.IL,
    Glyph.SHA,
)


def _window(G: TNFRGraph) -> int:
    from .validation.window import validate_window

    return validate_window(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))


def _rejection_mode(G: TNFRGraph) -> str:
    mode = G.graph.get("GRAMMAR_REJECTION_MODE", "fallback")
    if type(mode) is not str:
        raise TypeError("GRAMMAR_REJECTION_MODE must be 'fallback' or 'raise'")
    if mode not in {"fallback", "raise"}:
        raise ValueError("GRAMMAR_REJECTION_MODE must be 'fallback' or 'raise'")
    return mode


def _apply_glyph_to_targets(
    G: TNFRGraph, g: Glyph | str, nodes: Iterable[NodeId] | None = None
) -> None:
    """Apply the requested program glyph or reject; never substitute a glyph.

    This is the native glyph route, not an atomic network stage. No future
    handler context is inferred across targets or nested program scopes.
    """

    nodes_iter = G.nodes() if nodes is None else nodes
    w = _window(G)
    mode = _rejection_mode(G)
    had_mode = "GRAMMAR_REJECTION_MODE" in G.graph
    G.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    try:
        apply_glyph_with_grammar(G, nodes_iter, g, w)
    finally:
        if had_mode:
            G.graph["GRAMMAR_REJECTION_MODE"] = mode
        else:
            G.graph.pop("GRAMMAR_REJECTION_MODE", None)


def _advance(G: TNFRGraph, step_fn: AdvanceFn) -> None:
    step_fn(G)


def _record_trace(
    trace: ProgramTrace, G: TNFRGraph, op: OpTag, **data: Any
) -> TraceEntry:
    """Append an operation snapshot to ``trace`` using graph time metadata."""

    entry = {"t": _trace_time(G), "op": op.name, **data}
    trace.append(entry)
    return entry


def _trace_time(G: TNFRGraph) -> float:
    """Admit the retained structural clock; absence uses the zero convention."""

    return finite_represented_real(G.graph.get("_t", 0.0), "program trace clock")[0]


def _advance_and_record(
    G: TNFRGraph,
    trace: ProgramTrace,
    label: OpTag,
    step_fn: AdvanceFn,
    *,
    times: int = 1,
    **data: Any,
) -> None:
    entry = _record_trace(
        trace,
        G,
        label,
        selection_origin="supplied_program",
        step_origin="native_runtime" if step_fn is step else "supplied_step_function",
        record_boundary="before_step",
        step_status="pending",
        step_calls_requested=times,
        step_calls_started=0,
        step_calls_completed=0,
        clock_sample_available=True,
        **data,
    )
    try:
        for _ in range(times):
            _trace_time(G)
            entry["step_calls_started"] += 1
            _advance(G, step_fn)
            entry["step_calls_completed"] += 1
            entry["t"] = _trace_time(G)
        entry["step_status"] = "completed"
        entry["record_boundary"] = "after_step"
    except BaseException:
        # Preserve the supplied event/request even when the following step
        # fails. Callback effects remain; no invalid clock enters the trace.
        entry["step_status"] = "failed"
        entry["record_boundary"] = "step_failed"
        try:
            entry["t"] = _trace_time(G)
        except Exception:
            entry["clock_sample_available"] = False
        raise


def _handle_target(
    G: TNFRGraph,
    payload: TARGET,
    _curr_target: Sequence[NodeId] | None,
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
    _record_trace(
        trace,
        G,
        OpTag.TARGET,
        n=len(curr_target),
        targets=tuple(curr_target),
        selection_origin="supplied_program",
        record_boundary="target_selection",
    )
    return curr_target


def _handle_wait(
    G: TNFRGraph,
    steps: int,
    curr_target: Sequence[NodeId] | None,
    trace: ProgramTrace,
    step_fn: AdvanceFn,
) -> Sequence[NodeId] | None:
    _advance_and_record(G, trace, OpTag.WAIT, step_fn, times=steps, k=steps)
    return curr_target


def _handle_glyph(
    G: TNFRGraph,
    g: Glyph | str,
    curr_target: Sequence[NodeId] | None,
    trace: ProgramTrace,
    step_fn: AdvanceFn,
    label: OpTag = OpTag.GLYPH,
) -> Sequence[NodeId] | None:
    # Resolve the actual ordered population before dispatch: a child created
    # during an operation cannot retroactively become one of its targets.
    targets = tuple(G.nodes()) if curr_target is None else tuple(curr_target)
    # A tuple may itself be a valid node ID; dispatch an unambiguous collection.
    _apply_glyph_to_targets(G, g, list(targets))
    _advance_and_record(
        G,
        trace,
        label,
        step_fn,
        g=g,
        targets=targets,
        event_time=_trace_time(G),
        event_boundary="after_requested_glyph",
    )
    return curr_target


def _handle_thol(
    G: TNFRGraph,
    g: Glyph | str | None,
    curr_target: Sequence[NodeId] | None,
    trace: ProgramTrace,
    step_fn: AdvanceFn,
) -> Sequence[NodeId] | None:
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
    G: TNFRGraph, sequence: Sequence[Token], step_fn: AdvanceFn | None = None
) -> None:
    """Execute a scoped program through native glyphs and the declared step.

    Syntax is admitted in full before graph mutation. Block target changes
    are local to each body repetition; no child target is inferred. Explicit
    glyphs reject grammar violations instead of falling back. Live failures
    do not roll back already accepted operations or step-function effects.

    The step function runs after every glyph, including the THOL opening and
    explicit closure. WAIT requests additional calls. The default runtime
    step can itself select operators and update phase/capacity; none of these
    counts declares an event-free dwell, physical duration or complete law.
    """

    operations = _flatten(sequence)
    step_fn = step if step_fn is None else step_fn
    if not callable(step_fn):
        raise TypeError("step_fn must be callable")
    _window(G)
    _rejection_mode(G)
    _trace_time(G)
    maxlen = _program_count(
        get_param(G, "PROGRAM_TRACE_MAXLEN"), "PROGRAM_TRACE_MAXLEN", minimum=0
    )

    curr_target: Sequence[NodeId] | None = None

    history = ensure_history(G)
    trace_obj = history.get("program_trace")
    trace: ProgramTrace
    if not isinstance(trace_obj, deque) or trace_obj.maxlen != maxlen:
        trace = cast(ProgramTrace, deque(trace_obj or [], maxlen=maxlen))
        history["program_trace"] = trace
    else:
        trace = cast(ProgramTrace, trace_obj)

    scopes: list[tuple[BlockBoundary, Sequence[NodeId] | None]] = []
    for op, payload in operations:
        if op is OpTag.BLOCK_ENTER:
            scopes.append((payload, curr_target))
        elif op is OpTag.BLOCK_EXIT:
            boundary, curr_target = scopes.pop()
            if boundary != payload:
                raise ValueError("Compiled block boundaries do not match")
        else:
            handler: HandlerFn | None = HANDLERS.get(op)
            if handler is None:
                raise ValueError(f"Unknown operation: {op}")
            curr_target = handler(G, payload, curr_target, trace, step_fn)
            continue
        _record_trace(
            trace,
            G,
            op,
            path=payload.path,
            iteration=payload.iteration,
            repeat=payload.repeat,
        )


def compile_sequence(
    sequence: Iterable[Token] | Sequence[Token] | Any,
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
) -> list[tuple[OpTag, Any]]:
    """Return admitted operations, including non-operator block boundaries.

    This is compilation, not complete-word or live-state admission.
    """

    return _flatten(sequence, max_materialize=max_materialize)


def seq(*tokens: Token) -> list[Token]:
    """Return a mutable list of ``tokens`` for explicit sequence editing."""

    return list(tokens)


def block(*tokens: Token, repeat: int = 1, close: Glyph | None = None) -> THOL:
    """Build a lexically scoped THOL block; ``close=None`` adds no glyph."""

    return THOL(body=list(tokens), repeat=repeat, force_close=close)


def target(nodes: Iterable[NodeId] | None = None) -> TARGET:
    """Return a TARGET token selecting ``nodes`` (defaults to all nodes)."""

    return TARGET(nodes=nodes)


def wait(steps: int = 1) -> WAIT:
    """Return a WAIT token forcing ``steps`` structural updates before resuming."""

    return WAIT(steps=steps)


def basic_canonical_example() -> list[Token]:
    """Return the canonical preset sequence.

    Returns a copy of the canonical preset tokens to keep CLI defaults aligned
    with :func:`tnfr.config.presets.get_preset`.
    """

    return list(CANONICAL_PROGRAM_TOKENS)
