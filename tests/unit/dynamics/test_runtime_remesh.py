"""Runtime remeshing delegation tests."""

from __future__ import annotations

import pytest

from tnfr import operators
from tnfr.dynamics import runtime


class _Recorder:
    """Capture invocations to verify delegation occurs."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, graph: object) -> None:
        self.calls.append(graph)


class _ExplosiveError(RuntimeError):
    """Raised to ensure _maybe_remesh propagates delegate errors."""


def test_maybe_remesh_delegates_to_operator(monkeypatch, graph_canon) -> None:
    """_maybe_remesh should invoke the remesh operator exactly once."""

    G = graph_canon()
    recorder = _Recorder()

    monkeypatch.setattr(operators, "apply_remesh_if_globally_stable", recorder)
    monkeypatch.setattr(runtime, "apply_remesh_if_globally_stable", recorder)

    runtime._maybe_remesh(G)

    assert recorder.calls == [G]


def test_maybe_remesh_propagates_delegate_errors(monkeypatch, graph_canon) -> None:
    """Exceptions from the remesh delegate must bubble up."""

    G = graph_canon()

    def blow_up(graph):  # type: ignore[no-untyped-def]
        raise _ExplosiveError("remesh failure")

    monkeypatch.setattr(operators, "apply_remesh_if_globally_stable", blow_up)
    monkeypatch.setattr(runtime, "apply_remesh_if_globally_stable", blow_up)

    with pytest.raises(_ExplosiveError):
        runtime._maybe_remesh(G)
