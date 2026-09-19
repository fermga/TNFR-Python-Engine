"""Strict discrete-domain regressions for public runtime iteration counts."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.dynamics.runtime import run
from tnfr.errors import TNFRValueError
from tnfr.sense import sigma_rose


@pytest.mark.parametrize("steps", (True, False, 1.5, "2"))
def test_runtime_run_rejects_non_integral_step_counts_before_mutation(steps) -> None:
    graph = nx.Graph()
    graph.graph["marker"] = {"value": 1}
    before = deepcopy(dict(graph.graph))

    with pytest.raises(TypeError, match="non-negative integer"):
        run(graph, steps)

    assert graph.graph == before


def test_runtime_run_rejects_negative_step_count_before_mutation() -> None:
    graph = nx.Graph()
    before = deepcopy(dict(graph.graph))

    with pytest.raises(ValueError, match="non-negative integer"):
        run(graph, -1)

    assert graph.graph == before


@pytest.mark.parametrize("steps", (True, False, 1.5, "2"))
def test_sigma_rose_rejects_non_integral_windows_even_without_history(steps) -> None:
    graph = nx.Graph()

    with pytest.raises(TNFRValueError, match="non-negative integer"):
        sigma_rose(graph, steps=steps)

    assert "history" not in graph.graph


def test_sigma_rose_zero_window_is_empty_instead_of_all_history() -> None:
    graph = nx.Graph()
    graph.graph["history"] = {
        "sigma_counts": [
            {"t": 0, "AL": 2},
            {"t": 1, "IL": 3},
        ]
    }

    zero = sigma_rose(graph, steps=0)
    latest = sigma_rose(graph, steps=1)

    assert all(value == 0 for value in zero.values())
    assert latest["IL"] == 3
    assert latest["AL"] == 0
