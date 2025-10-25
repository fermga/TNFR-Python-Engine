"""Tests for callback registration in :mod:`tnfr.cli.execution`."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.cli import execution


@pytest.fixture(name="callback_names")
def _callback_names() -> tuple[str, ...]:
    return (
        "register_sigma_callback",
        "register_metrics_callbacks",
        "register_trace",
        "_metrics_step",
        "validate_canon",
    )


@pytest.mark.parametrize("graph_factory", [nx.Graph])
def test_register_callbacks_and_observer_invokes_all(
    monkeypatch: pytest.MonkeyPatch,
    callback_names: tuple[str, ...],
    graph_factory: type[nx.Graph],
) -> None:
    """``register_callbacks_and_observer`` should invoke each helper once."""

    calls: dict[str, int] = {name: 0 for name in callback_names}
    graph = graph_factory()

    def make_stub(func_name: str):
        def _stub(*args, **kwargs):
            calls[func_name] += 1
            return None

        return _stub

    for name in callback_names:
        monkeypatch.setattr(execution, name, make_stub(name))

    execution.register_callbacks_and_observer(graph)

    assert calls == {name: 1 for name in callback_names}
