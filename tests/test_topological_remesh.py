"""Pruebas de remeshing topológico."""

import networkx as nx

from tnfr.constants import attach_defaults
from tnfr.operators import apply_topological_remesh


def _graph_with_epi(graph_canon, n=6):
    G = graph_canon()
    for i in range(n):
        G.add_node(i)
    attach_defaults(G)
    for i in G.nodes():
        G.nodes[i]["EPI"] = float(i)
    return G


def test_remesh_community_reduces_nodes_and_preserves_connectivity(graph_canon):
    G = _graph_with_epi(graph_canon, n=6)
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (2, 3),
        ]
    )
    attach_defaults(G)
    apply_topological_remesh(G, mode="community")
    assert nx.is_connected(G)
    assert G.number_of_nodes() < 6
    ev = G.graph.get("history", {}).get("remesh_events", [])
    assert ev and ev[-1].get("mode") == "community"


def test_remesh_knn_preserves_connectivity(graph_canon):
    G = _graph_with_epi(graph_canon, n=5)
    apply_topological_remesh(G, mode="knn", k=2, p_rewire=1.0, seed=1)
    assert nx.is_connected(G)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() >= 4


def test_remesh_mst_returns_tree(graph_canon):
    G = _graph_with_epi(graph_canon, n=5)
    apply_topological_remesh(G, mode="mst")
    assert nx.is_tree(G)
    assert G.number_of_nodes() == 5
