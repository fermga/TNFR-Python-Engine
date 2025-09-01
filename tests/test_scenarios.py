"""Pruebas de scenarios."""
import pytest
import networkx as nx

from tnfr.scenarios import build_graph


def test_build_graph_valid_topologies():
    n = 6
    seed = 1
    references = {
        "ring": nx.cycle_graph(n),
        "complete": nx.complete_graph(n),
        "erdos": nx.gnp_random_graph(n, 3.0 / n, seed=seed),
    }
    for topology, ref_graph in references.items():
        G = build_graph(n=n, topology=topology, seed=seed)
        assert nx.is_isomorphic(G, ref_graph)


def test_build_graph_invalid_topology():
    with pytest.raises(ValueError):
        build_graph(n=5, topology="invalid", seed=1)

