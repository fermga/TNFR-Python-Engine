import pytest
import networkx as nx

from tnfr.constants import ALIAS_THETA
from tnfr.metrics import coherence_matrix, local_phase_sync_weighted


def make_graph(offset=0):
    G = nx.Graph()
    G.add_node(offset)
    G.add_node(offset + 1)
    G.add_edge(offset, offset + 1)
    G.nodes[offset][ALIAS_THETA[0]] = 0.0
    G.nodes[offset + 1][ALIAS_THETA[0]] = 0.0
    return G


def test_local_phase_sync_independent_graphs():
    G1 = make_graph(0)
    G2 = make_graph(10)

    nodes1, W1 = coherence_matrix(G1)
    nodes2, W2 = coherence_matrix(G2)

    r1 = local_phase_sync_weighted(G1, nodes1[0], nodes_order=nodes1, W_row=W1)
    r2 = local_phase_sync_weighted(G2, nodes2[0], nodes_order=nodes2, W_row=W2)

    assert r1 == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)

    key = "_lpsw_cache"
    assert G1.graph[key]["nodes"] == tuple(nodes1)
    assert G2.graph[key]["nodes"] == tuple(nodes2)
    assert G1.graph[key] is not G2.graph[key]

    r1_again = local_phase_sync_weighted(
        G1, nodes1[0], nodes_order=nodes1, W_row=W1
    )
    assert r1_again == pytest.approx(r1)
