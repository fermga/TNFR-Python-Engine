"""Execution ordinals independent of physical time and retained telemetry.

The ordinal distinguishes admitted runtime invocations in the tracked epoch,
including ones that later fail. Between calls its value counts those admitted
invocations; inside a call it identifies that invocation. Failed calls can have
partially changed state, so their identifier must not be reused. This is
execution provenance, not a clock, an operator count, or a completion certificate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral
from typing import Any

from .compat.dataclass import dataclass

RUNTIME_STEP_NEXT_KEY = "_runtime_step_next"
RUNTIME_STEP_UNSET = object()


@dataclass(slots=True)
class _ActiveRuntimeStep:
    graph: Mapping[str, Any]
    active: bool = True


_ACTIVE_GRAPHS: ContextVar[tuple[_ActiveRuntimeStep, ...]] = ContextVar(
    "tnfr_active_runtime_graphs", default=()
)


def _ordinal(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def resolve_runtime_step_index(
    next_index: Any = RUNTIME_STEP_UNSET,
    *,
    fallback: int = 0,
) -> int:
    """Resolve raw owner fields; consume legacy fallback only without markers."""
    if next_index is RUNTIME_STEP_UNSET:
        return _ordinal(fallback, "legacy step index")
    return _ordinal(next_index, RUNTIME_STEP_NEXT_KEY)


def prepare_runtime_step(graph: Mapping[str, Any]) -> int:
    """Validate an invocation before history, callbacks or nodal writes."""
    ordinal = resolve_runtime_step_index(
        graph.get(RUNTIME_STEP_NEXT_KEY, RUNTIME_STEP_UNSET),
    )
    if any(scope.active and scope.graph is graph for scope in _ACTIVE_GRAPHS.get()):
        raise RuntimeError("Recursive runtime.step on the same graph is unsupported")
    return ordinal


@contextmanager
def runtime_step_scope(graph: MutableMapping[str, Any], ordinal: int) -> Iterator[None]:
    """Reserve one admitted ordinal and expose it throughout its callbacks.

    The persisted value is the active ordinal inside the scope and its
    successor after exit. The reentrancy guard is execution-local, so copying
    a graph does not copy a stale active-call marker. A copy may continue at
    the ordinal captured with its state; it does not certify completion of
    the source call or replay its unfinished work.
    Inherited async contexts share a liveness record, so they reject reentry
    while the source is active but can resume after its scope has exited.

    The first tracked call starts at zero; prior telemetry cannot reconstruct
    unobserved execution. Late failure consumes the ordinal without promising
    rollback. Private markers are runtime-owned; concurrent mutation and caller
    modification of them are outside this execution contract.
    """
    if prepare_runtime_step(graph) != ordinal:
        raise RuntimeError("Runtime step ordinal changed during admission")
    graph[RUNTIME_STEP_NEXT_KEY] = ordinal
    scope = _ActiveRuntimeStep(graph)
    live_scopes = tuple(item for item in _ACTIVE_GRAPHS.get() if item.active)
    token = _ACTIVE_GRAPHS.set((*live_scopes, scope))
    try:
        yield
    finally:
        try:
            graph[RUNTIME_STEP_NEXT_KEY] = ordinal + 1
        finally:
            scope.active = False
            _ACTIVE_GRAPHS.reset(token)
