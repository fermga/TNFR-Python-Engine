"""Pruebas de remeshing topológico."""
import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.operators import apply_topological_remesh
from tnfr.glyph_history import ensure_history


def _graph_with_epi(graph_canon, n=6):
    G = graph_canon()
    for i in range(n):
        G.add_node(i)
    inject_defaults(G)
    for i in G.nodes():
        G.nodes[i]["EPI"] = float(i)
    return G


def _edge_set(G):
    return {tuple(sorted(edge)) for edge in G.edges()}


def test_remesh_community_reduces_nodes_and_preserves_connectivity(
    graph_canon,
):
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
    inject_defaults(G)
    apply_topological_remesh(G, mode="community")
    assert nx.is_connected(G)
    assert G.number_of_nodes() < 6
    ev = ensure_history(G).get("remesh_events", [])
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


def test_remesh_respects_graph_random_seed(graph_canon):
    base = _graph_with_epi(graph_canon, n=6)
    base.graph["RANDOM_SEED"] = 1234

    G1 = base.copy()
    G2 = base.copy()

    apply_topological_remesh(G1, mode="knn", k=2, p_rewire=0.3)
    apply_topological_remesh(G2, mode="knn", k=2, p_rewire=0.3)

    assert _edge_set(G1) == _edge_set(G2)


def test_remesh_sequences_depend_on_mode_and_k(graph_canon):
    base = _graph_with_epi(graph_canon, n=6)
    base.graph["RANDOM_SEED"] = 4321

    knn2 = base.copy()
    knn3 = base.copy()
    mst = base.copy()

    apply_topological_remesh(knn2, mode="knn", k=2, p_rewire=0.3)
    apply_topological_remesh(knn3, mode="knn", k=3, p_rewire=0.3)
    apply_topological_remesh(mst, mode="mst")

    edges_knn2 = _edge_set(knn2)
    edges_knn3 = _edge_set(knn3)
    edges_mst = _edge_set(mst)

    assert edges_knn2 != edges_knn3
    assert edges_knn2 != edges_mst
