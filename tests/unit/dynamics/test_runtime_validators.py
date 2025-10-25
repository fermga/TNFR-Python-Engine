"""Tests for runtime validator orchestration."""

import pytest

from tnfr.dynamics import runtime


def test_run_validators_invokes_utils(monkeypatch, graph_canon):
    """Ensure ``_run_validators`` delegates to ``tnfr.utils.run_validators``."""

    G = graph_canon()
    called_with = {}

    def recorder(graph):
        called_with["graph"] = graph

    monkeypatch.setattr("tnfr.utils.run_validators", recorder)

    runtime._run_validators(G)

    assert called_with["graph"] is G


def test_run_validators_propagates_exceptions(monkeypatch, graph_canon):
    """Exceptions raised by validators should propagate to callers."""

    G = graph_canon()
    sentinel = RuntimeError("validator boom")

    def failing_validator(graph):  # pragma: no cover - raising stub
        raise sentinel

    monkeypatch.setattr("tnfr.utils.run_validators", failing_validator)

    with pytest.raises(RuntimeError) as excinfo:
        runtime._run_validators(G)

    assert excinfo.value is sentinel
