import networkx as nx
import pytest

from tnfr.constants import THETA_PRIMARY
from tnfr.metrics import (
    coherence_matrix,
    local_phase_sync,
    local_phase_sync_unweighted,
)


def make_graph():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.nodes[0][THETA_PRIMARY] = 0.0
    G.nodes[1][THETA_PRIMARY] = 0.0
    return G


def test_local_phase_sync_unweighted():
    G = make_graph()
    r = local_phase_sync_unweighted(G, 0)
    assert r == pytest.approx(1.0)


def test_local_phase_sync_with_weights():
    G = make_graph()
    nodes, W = coherence_matrix(G)
    r = local_phase_sync(G, nodes[0], nodes_order=nodes, W_row=W)
    assert r == pytest.approx(1.0)
