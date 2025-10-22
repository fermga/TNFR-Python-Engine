import networkx as nx
import pytest

from tnfr.constants import (
    EPI_PRIMARY,
    SI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
)
from tnfr.metrics import coherence_matrix, local_phase_sync_weighted
from tnfr.utils import ensure_node_index_map


def make_graph(graph_canon, offset=0):
    G = graph_canon()
    G.add_node(offset)
    G.add_node(offset + 1)
    G.add_edge(offset, offset + 1)
    G.nodes[offset][THETA_PRIMARY] = 0.0
    G.nodes[offset + 1][THETA_PRIMARY] = 0.0
    return G


def test_local_phase_sync_independent_graphs(graph_canon):
    G1 = make_graph(graph_canon, 0)
    G2 = make_graph(graph_canon, 10)

    nodes1, W1 = coherence_matrix(G1)
    nodes2, W2 = coherence_matrix(G2)

    r1 = local_phase_sync_weighted(G1, nodes1[0], nodes_order=nodes1, W_row=W1)
    r2 = local_phase_sync_weighted(G2, nodes2[0], nodes_order=nodes2, W_row=W2)

    assert r1 == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)

    map1 = ensure_node_index_map(G1)
    map2 = ensure_node_index_map(G2)
    assert map1 is not map2
    assert set(map1.keys()) == set(nodes1)
    assert set(map2.keys()) == set(nodes2)

    r1_again = local_phase_sync_weighted(G1, nodes1[0], nodes_order=nodes1, W_row=W1)
    assert r1_again == pytest.approx(r1)
    assert ensure_node_index_map(G1) is map1


def test_node_index_map_invalidation(graph_canon):
    G = make_graph(graph_canon, 0)
    coherence_matrix(G)
    mapping1 = ensure_node_index_map(G)

    G.add_node(2)
    coherence_matrix(G)
    mapping2 = ensure_node_index_map(G)

    assert mapping1 is not mapping2
    assert set(mapping2.keys()) == set(G.nodes())


def test_use_numpy_parameter_matches_loops():
    G = nx.cycle_graph(3)
    for n in G.nodes:
        G.nodes[n][THETA_PRIMARY] = 0.0
    nodes_l, W_l = coherence_matrix(G, use_numpy=False)
    nodes_v, W_v = coherence_matrix(G, use_numpy=True)
    nodes_p, W_p = coherence_matrix(G, use_numpy=False, n_jobs=2)
    assert nodes_l == nodes_v
    assert W_l == W_v
    assert nodes_p == nodes_l
    assert W_p == W_l


def test_coherence_parallel_consistency_neighbors_and_all(graph_canon):
    G = graph_canon()
    G.add_nodes_from(range(6))
    G.add_edges_from((i, (i + 1) % 6) for i in range(6))

    for n in G.nodes:
        node = G.nodes[n]
        node[THETA_PRIMARY] = n * 0.15
        node[EPI_PRIMARY] = float(n) / 5.0
        node[VF_PRIMARY] = float(n) / 10.0
        node[SI_PRIMARY] = min(1.0, max(0.0, (n % 3) / 2.0))

    nodes_serial, W_serial = coherence_matrix(G, use_numpy=False, n_jobs=1)
    nodes_parallel, W_parallel = coherence_matrix(G, use_numpy=False, n_jobs=3)

    assert nodes_serial == nodes_parallel
    assert W_serial == W_parallel

    cfg = G.graph["COHERENCE"]
    cfg["scope"] = "all"
    cfg["n_jobs"] = 3

    nodes_serial_all, W_serial_all = coherence_matrix(G, use_numpy=False, n_jobs=1)
    nodes_parallel_all, W_parallel_all = coherence_matrix(G, use_numpy=False)

    assert nodes_serial_all == nodes_parallel_all
    assert W_serial_all == W_parallel_all
