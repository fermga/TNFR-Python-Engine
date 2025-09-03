"""TNFR programming language."""

from __future__ import annotations
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager
from collections import deque

from .constants import get_param
from .grammar import apply_glyph_with_grammar
from .constants_glyphs import GLYPHS_CANONICAL_SET
from .types import Glyph
from .collections_utils import ensure_collection
from .glyph_history import ensure_history

# Basic types
Node = Any
AdvanceFn = Callable[[Any], None]  # normalmente dynamics.step

# ---------------------
# DSL constructs
# ---------------------


@dataclass
class WAIT:
    """Wait a number of steps without applying glyphs.

    Attributes:
        steps: number of steps the system advances.
    """

    steps: int = 1


@dataclass
class TARGET:
    """Select the subset of nodes for subsequent glyphs.

    Attributes:
        nodes: iterable of nodes; ``None`` for all nodes.
    """

    nodes: Optional[Iterable[Node]] = None  # None = all nodes


@dataclass
class THOL:
    """THOL block that opens self-organisation.

    Attributes:
        body: sequence of tokens to execute within the block.
        repeat: how many times to repeat ``body``.
        force_close: ``Glyph.SHA`` or ``Glyph.NUL`` to force closure.
    """

    body: Sequence[Any]
    repeat: int = 1  # number of times to repeat the body
    force_close: Optional[Glyph] = None  # None → automatic closure; SHA or NUL to force


Token = Union[Glyph, WAIT, TARGET, THOL]

# ---------------------
# Internal utilities
# ---------------------


@contextmanager
def _forced_selector(G, glyph: Glyph):
    """Temporarily override the glyph selector to force ``glyph``.

    The canonical grammar is enforced before applying.
    """
    prev = G.graph.get("glyph_selector")

    def selector_forced(_G, _n):
        return glyph

    G.graph["glyph_selector"] = selector_forced
    try:
        yield
    finally:
        if prev is None:
            G.graph.pop("glyph_selector", None)
        else:
            G.graph["glyph_selector"] = prev


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


def _apply_glyph_to_targets(G, g: Glyph | str, nodes: Optional[Iterable[Node]] = None):
    """Apply ``g`` to ``nodes`` (or all nodes) respecting the grammar.

    ``nodes`` may be any iterable of nodes, including a ``NodeView`` or other
    iterables. To avoid unnecessary materialisation, iteration happens over the
    iterable as-is; callers must materialise it if indexing is needed.
    """
    nodes_iter = _all_nodes(G) if nodes is None else nodes
    w = _window(G)
    apply_glyph_with_grammar(G, nodes_iter, g, w)


def _advance(G, step_fn: Optional[AdvanceFn] = None):
    if step_fn is None:
        from .dynamics import step as step_fn
    step_fn(G)


# ---------------------
# Sequence compilation → list of atomic operations
# ---------------------


def _flatten(seq: Sequence[Token]) -> List[Tuple[str, Any]]:
    """Return list of operations ``(op, payload)``.
    ``op`` ∈ { 'GLYPH', 'WAIT', 'TARGET', 'THOL' }.

    Implemented iteratively using an explicit stack to avoid deep recursion
    when ``THOL`` blocks are nested.
    """
    ops: List[Tuple[str, Any]] = []
    stack: List[Token] = list(reversed(seq))

    while stack:
        item = stack.pop()
        if isinstance(item, TARGET):
            ops.append(("TARGET", item))
            continue
        if isinstance(item, WAIT):
            ops.append(("WAIT", item.steps))
            continue
        if isinstance(item, THOL):
            # Open THOL block and push its body iteratively
            ops.append(("THOL", Glyph.THOL.value))

            repeats = max(1, int(item.repeat))
            if item.force_close is not None and not isinstance(item.force_close, Glyph):
                raise ValueError("force_close must be a Glyph")
            closing = (
                item.force_close
                if isinstance(item.force_close, Glyph)
                and item.force_close in {Glyph.SHA, Glyph.NUL}
                else None
            )
            # Explicit closure must run at the end, so push onto stack before
            # expanding body.
            if closing is not None:
                stack.append(closing)

            # Insert bodies in order so the first repetition is processed first.
            for _ in range(repeats):
                for tok in reversed(item.body):
                    stack.append(tok)
            continue

        # item should be a glyph
        g = item.value if isinstance(item, Glyph) else str(item)
        if g not in GLYPHS_CANONICAL_SET:
            raise ValueError(f"Non-canonical glyph: {g}")
        ops.append(("GLYPH", g))
    return ops


# ---------------------
# Handlers for atomic tokens
# ---------------------


def _record_trace(trace: deque, G, op: str, **data) -> None:
    trace.append({"t": float(G.graph.get("_t", 0.0)), "op": op, **data})


def _handle_target(G, payload: TARGET, _curr_target, trace: deque, _step_fn):
    """Handle a ``TARGET`` token and return the active node set.

    Returns
    -------
    Collection[Node]
        Collection of nodes to be used for subsequent operations.
    """
    nodes_src = _all_nodes(G) if payload.nodes is None else payload.nodes
    curr_target = tuple(ensure_collection(nodes_src))
    _record_trace(trace, G, "TARGET", n=len(curr_target))
    return curr_target


def _handle_wait(
    G, steps: int, curr_target, trace: deque, step_fn: Optional[AdvanceFn]
):
    steps = max(1, int(steps))
    for _ in range(steps):
        _advance(G, step_fn)
    _record_trace(trace, G, "WAIT", k=steps)
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
    _advance(G, step_fn)
    _record_trace(trace, G, label, g=g)
    return curr_target


# ---------------------
# Public API
# ---------------------


def play(G, sequence: Sequence[Token], step_fn: Optional[AdvanceFn] = None) -> None:
    """Execute a canonical sequence on graph ``G``.

    Rules:
      - Use ``TARGET(nodes=...)`` to change the subset of application.
      - ``WAIT(k)`` advances ``k`` steps with the current selector (no forced glyph).
      - ``THOL([...], repeat=r, force_close=…)`` opens a self-organising block,
        repeats the body and optionally forces closure with SHA/NUL.
      - Glyphs are applied via ``enforce_canonical_grammar``.
    """
    ops = _flatten(sequence)
    curr_target: Optional[List[Node]] = None

    # Traza de programa en history
    history = ensure_history(G)
    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    trace = history.get("program_trace")
    if not isinstance(trace, deque) or trace.maxlen != maxlen:
        trace = deque(trace or [], maxlen=maxlen)
        history["program_trace"] = trace

    handlers = {
        "TARGET": _handle_target,
        "WAIT": _handle_wait,
        "GLYPH": _handle_glyph,
        "THOL": lambda G, g, curr_target, trace, step_fn: _handle_glyph(
            G, g or Glyph.THOL.value, curr_target, trace, step_fn, label="THOL"
        ),
    }

    for op, payload in ops:
        handler = handlers.get(op)
        if handler is None:
            raise ValueError(f"Unknown operation: {op}")
        curr_target = handler(G, payload, curr_target, trace, step_fn)


# ---------------------
# Helpers to build sequences easily
# ---------------------


def seq(*tokens: Token) -> List[Token]:
    return list(tokens)


def block(*tokens: Token, repeat: int = 1, close: Optional[Glyph] = None) -> THOL:
    return THOL(body=list(tokens), repeat=repeat, force_close=close)


def target(nodes: Optional[Iterable[Node]] = None) -> TARGET:
    return TARGET(nodes=nodes)


def wait(steps: int = 1) -> WAIT:
    return WAIT(steps=max(1, int(steps)))


def basic_canonical_example() -> List[Token]:
    """Reference canonical sequence.

    SHA → AL → RA → ZHIR → NUL → THOL
    """
    return seq(Glyph.SHA, Glyph.AL, Glyph.RA, Glyph.ZHIR, Glyph.NUL, Glyph.THOL)
