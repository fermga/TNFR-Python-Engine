from __future__ import annotations

from functools import cached_property
from typing import Any, Iterable

import pytest

from tnfr.metrics.common import (
    compute_coherence,
    ensure_neighbors_map,
    merge_graph_weights,
)


class _MockNodeView:
    def __init__(self, graph: "_MockGraph") -> None:
        self._graph = graph

    def __iter__(self) -> Iterable[str]:
        return iter(self._graph._order)

    def __call__(self, data: bool = False):
        if data:
            return [(node, self._graph._nodes[node]) for node in self._graph._order]
        return list(self._graph._order)

    def __getitem__(self, node: str) -> dict[str, Any]:
        return self._graph._nodes[node]


class _MockEdgeView:
    def __init__(self, graph: "_MockGraph") -> None:
        self._graph = graph

    def __iter__(self) -> Iterable[tuple[str, str]]:
        return iter(self._graph._edges)

    def __call__(self, data: bool = False):
        return list(self._graph._edges)


class _MockGraph:
    def __init__(self) -> None:
        self._order = ("a", "b")
        self._nodes = {
            "a": {"dnfr": 0.1, "dEPI": 0.05, "frequency": 1.0},
            "b": {"dnfr": 0.2, "dEPI": 0.1, "frequency": 2.0},
        }
        self._adj = {"a": {"b": {}}, "b": {"a": {}}}
        self._edges = (("a", "b"),)
        self.graph: dict[str, Any] = {"DNFR_WEIGHTS": {"dnfr": 0.8}}

    @cached_property
    def nodes(self) -> _MockNodeView:
        return _MockNodeView(self)

    @cached_property
    def edges(self) -> _MockEdgeView:
        return _MockEdgeView(self)

    def number_of_nodes(self) -> int:
        return len(self._order)

    def neighbors(self, node: str):
        return iter(self._adj[node])

    def __getitem__(self, node: str) -> dict[str, Any]:
        return self._adj[node]

    def __iter__(self):
        return iter(self._order)


@pytest.fixture()
def mock_graph() -> _MockGraph:
    return _MockGraph()


def test_compute_coherence_supports_cached_views(mock_graph: _MockGraph) -> None:
    coherence = compute_coherence(mock_graph)
    expected = pytest.approx(1.0 / (1.0 + 0.15 + 0.075))
    assert coherence == expected


def test_neighbors_map_uses_iterable_neighbours(mock_graph: _MockGraph) -> None:
    neighbors = ensure_neighbors_map(mock_graph)
    assert tuple(neighbors["a"]) == ("b",)


def test_merge_graph_weights_reads_graph_mapping(mock_graph: _MockGraph) -> None:
    weights = merge_graph_weights(mock_graph, "DNFR_WEIGHTS")
    assert weights["dnfr"] == 0.8
