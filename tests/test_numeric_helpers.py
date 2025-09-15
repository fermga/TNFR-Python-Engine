import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.helpers.numeric import neighbor_mean
from tnfr.observers import phase_sync


def test_neighbor_mean_returns_default_when_no_neighbors():
    G = nx.Graph()
    G.add_node(0)

    assert neighbor_mean(G, 0, ("EPI",), default=2.5) == pytest.approx(2.5)


def test_neighbor_mean_averages_existing_values():
    G = nx.Graph()
    G.add_nodes_from(
        (
            (0, {}),
            (1, {"EPI": 1.0}),
            (2, {"EPI": 3.0}),
        )
    )
    G.add_edges_from(((0, 1), (0, 2)))

    assert neighbor_mean(G, 0, ("EPI",), default=0.0) == pytest.approx(2.0)


def test_phase_sync_statistics_fallback(monkeypatch):
    monkeypatch.setattr("tnfr.observers.get_numpy", lambda: None)

    G = nx.Graph()
    G.add_nodes_from(
        (
            (0, {"theta": 0.0}),
            (1, {"theta": 0.1}),
            (2, {"theta": -0.1}),
        )
    )

    # 0 variance would yield 1; this setup triggers the statistics branch.
    diffs = [0.0, 0.1, -0.1]
    expected_var = sum(d * d for d in diffs) / len(diffs)
    assert phase_sync(G, R=1.0, psi=0.0) == pytest.approx(1.0 / (1.0 + expected_var))
