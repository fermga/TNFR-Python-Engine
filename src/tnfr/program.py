"""TNFR programming language.

Public exports are declared in ``__all__`` for explicit star imports.
"""

from __future__ import annotations
from typing import Any, Optional, Union
from dataclasses import dataclass
from collections import deque
from collections.abc import Callable, Iterable, Sequence

from .token_parser import _flatten_tokens, validate_token, _parse_tokens

from .constants import get_param
from .grammar import apply_glyph_with_grammar
from .constants_glyphs import GLYPHS_CANONICAL_SET
from .types import Glyph
from .collections_utils import ensure_collection
from .glyph_history import ensure_history

# Basic types
Node = Any
AdvanceFn = Callable[[Any], None]  # normalmente dynamics.step
_STEP_FN: Optional[AdvanceFn] = None

__all__ = [
    "WAIT",
    "TARGET",
    "THOL",
    "Token",
    "THOL_SENTINEL",
    "_flatten_thol",
    "_flatten",
    "_handle_target",
    "_handle_wait",
    "_handle_glyph",
    "_flatten_tokens",
    "validate_token",
    "_parse_tokens",
    "seq",
    "block",
    "target",
    "wait",
    "play",
    "basic_canonical_example",
]

# ---------------------
# DSL constructs
# ---------------------


@dataclass(slots=True)
class WAIT:
    """Wait a number of steps without applying glyphs.

    Attributes:
        steps: number of steps the system advances.
    """

    steps: int = 1


@dataclass(slots=True)
class TARGET:
    """Select the subset of nodes for subsequent glyphs.

    Attributes:
        nodes: iterable of nodes; ``None`` for all nodes.
    """

    nodes: Optional[Iterable[Node]] = None  # None = all nodes


@dataclass(slots=True)
class THOL:
    """THOL block that opens self-organisation.

    Attributes:
        body: sequence of tokens to execute within the block.
        repeat: how many times to repeat ``body``.
        force_close: ``Glyph.SHA`` or ``Glyph.NUL`` to force closure.
    """

    body: Sequence[Any]
    repeat: int = 1  # number of times to repeat the body
    force_close: Optional[Glyph] = (
        None  # None → automatic closure; SHA or NUL to force
    )


Token = Union[Glyph, WAIT, TARGET, THOL]

# Sentinel used internally by ``_flatten`` to mark the end of a ``THOL`` block
THOL_SENTINEL = object()

# ---------------------
# Internal utilities
# ---------------------


def _window(G) -> int:
    return int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))


def _all_nodes(G):
    """Return a view or generator over all nodes in ``G``.

    Using ``G.nodes()`` avoids creating an intermediate list and allows
    callers to materialise the collection only when strictly necessary.
    """
    return G.nodes()


# ---------------------
# Execution core
# ---------------------


def _apply_glyph_to_targets(
    G, g: Glyph | str, nodes: Optional[Iterable[Node]] = None
):
    """Apply ``g`` to ``nodes`` (or all nodes) respecting the grammar.

    ``nodes`` may be any iterable of nodes, including a ``NodeView`` or other
    iterables. To avoid unnecessary materialisation, iteration happens over the
    iterable as-is; callers must materialise it if indexing is needed.
    """
    nodes_iter = _all_nodes(G) if nodes is None else nodes
    w = _window(G)
    apply_glyph_with_grammar(G, nodes_iter, g, w)


def _advance(G, step_fn: Optional[AdvanceFn] = None):
    global _STEP_FN
    if step_fn is None:
        if _STEP_FN is None:
            from .dynamics import step as _STEP_FN
        step_fn = _STEP_FN
    step_fn(G)


# ---------------------
# Sequence compilation → list of atomic operations
# ---------------------


def _flatten_thol(item: THOL, stack: deque[Any]) -> None:
    """Expand a ``THOL`` block onto ``stack`` for processing."""
    repeats = int(item.repeat)
    if repeats < 1:
        raise ValueError("repeat must be ≥1")
    if item.force_close is not None and not isinstance(item.force_close, Glyph):
        raise ValueError("force_close must be a Glyph")
    closing = (
        item.force_close
        if isinstance(item.force_close, Glyph)
        and item.force_close in {Glyph.SHA, Glyph.NUL}
        else None
    )
    seq = ensure_collection(item.body)

    def _iter_reversed_body():
        for _ in range(repeats):
            if closing is not None:
                yield closing
            yield from reversed(seq)

    stack.extend(_iter_reversed_body())
    stack.append(THOL_SENTINEL)


def _flatten(seq: Sequence[Token]) -> list[tuple[str, Any]]:
    """Return list of operations ``(op, payload)``.
    ``op`` ∈ { 'GLYPH', 'WAIT', 'TARGET', 'THOL' }.

    Implemented iteratively using an explicit stack to avoid deep recursion
    when ``THOL`` blocks are nested.
    """
    ops: list[tuple[str, Any]] = []
    stack: deque[Any] = deque(reversed(ensure_collection(seq)))

    while stack:
        item = stack.pop()
        if item is THOL_SENTINEL:
            ops.append(("THOL", Glyph.THOL.value))
        elif isinstance(item, TARGET):
            ops.append(("TARGET", item))
        elif isinstance(item, WAIT):
            steps = max(1, int(getattr(item, "steps", 1)))
            ops.append(("WAIT", steps))
        elif isinstance(item, THOL):
            _flatten_thol(item, stack)
        elif isinstance(item, (Glyph, str)):
            g = item.value if isinstance(item, Glyph) else str(item)
            if g not in GLYPHS_CANONICAL_SET:
                raise ValueError(f"Non-canonical glyph: {g}")
            ops.append(("GLYPH", g))
        else:
            raise TypeError(f"Unsupported token: {item!r}")
    return ops


# ---------------------
# Handlers for atomic tokens
# ---------------------


def _record_trace(trace: deque, G, op: str, **data) -> None:
    trace.append({"t": float(G.graph.get("_t", 0.0)), "op": op, **data})


def _advance_and_record(
    G, trace: deque, label: str, step_fn: Optional[AdvanceFn], *, times: int = 1, **data
) -> None:
    for _ in range(times):
        _advance(G, step_fn)
    _record_trace(trace, G, label, **data)


def _handle_target(G, payload: TARGET, _curr_target, trace: deque, _step_fn):
    """Handle a ``TARGET`` token and return the active node set.

    Returns
    -------
    Collection[Node]
        Collection of nodes to be used for subsequent operations.
    """
    nodes_src = _all_nodes(G) if payload.nodes is None else payload.nodes
    nodes = ensure_collection(nodes_src)
    curr_target = nodes if isinstance(nodes, Sequence) else tuple(nodes)
    _record_trace(trace, G, "TARGET", n=len(curr_target))
    return curr_target


def _handle_wait(
    G, steps: int, curr_target, trace: deque, step_fn: Optional[AdvanceFn]
):
    _advance_and_record(G, trace, "WAIT", step_fn, times=steps, k=steps)
    return curr_target


def _handle_glyph(
    G,
    g: str,
    curr_target,
    trace: deque,
    step_fn: Optional[AdvanceFn],
    label: str = "GLYPH",
):
    _apply_glyph_to_targets(G, g, curr_target)
    _advance_and_record(G, trace, label, step_fn, g=g)
    return curr_target


HANDLERS = {
    "TARGET": _handle_target,
    "WAIT": _handle_wait,
    "GLYPH": _handle_glyph,
    "THOL": lambda G, g, curr_target, trace, step_fn: _handle_glyph(
        G, g or Glyph.THOL.value, curr_target, trace, step_fn, label="THOL"
    ),
}


# ---------------------
# Public API
# ---------------------


def play(
    G, sequence: Sequence[Token], step_fn: Optional[AdvanceFn] = None
) -> None:
    """Execute a canonical sequence on graph ``G``.

    Rules:
      - Use ``TARGET(nodes=...)`` to change the subset of application.
      - ``WAIT(k)`` advances ``k`` steps with the current selector
        (no forced glyph).
      - ``THOL([...], repeat=r, force_close=…)`` opens a self-organising block,
        repeats the body and optionally forces closure with SHA/NUL.
      - Glyphs are applied via ``enforce_canonical_grammar``.
    """
    if step_fn is None:
        from . import dynamics

        step_fn = dynamics.step

    ops = _flatten(sequence)
    curr_target: Optional[list[Node]] = None

    # Traza de programa en history
    history = ensure_history(G)
    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    trace = history.get("program_trace")
    if not isinstance(trace, deque) or trace.maxlen != maxlen:
        trace = deque(trace or [], maxlen=maxlen)
        history["program_trace"] = trace

    for op, payload in ops:
        handler = HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown operation: {op}")
        curr_target = handler(G, payload, curr_target, trace, step_fn)


# ---------------------
# Helpers to build sequences easily
# ---------------------


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

    SHA → AL → RA → ZHIR → NUL → THOL
    """
    return seq(
        Glyph.SHA, Glyph.AL, Glyph.RA, Glyph.ZHIR, Glyph.NUL, Glyph.THOL
    )
