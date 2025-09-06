import networkx as nx

from tnfr import dynamics, operators
from tnfr.helpers import increment_edge_version


def test_cached_nodes_and_A_reuse_and_invalidate():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    data1 = dynamics._prepare_dnfr_data(G)
    nodes1 = data1["nodes"]
    data2 = dynamics._prepare_dnfr_data(G)
    assert nodes1 is data2["nodes"]
    G.add_edge(0, 2)
    increment_edge_version(G)
    data3 = dynamics._prepare_dnfr_data(G)
    assert data3["nodes"] is not nodes1


def test_cached_nodes_and_A_invalidate_on_node_addition():
    G = nx.Graph()
    G.add_edge(0, 1)
    data1 = dynamics._prepare_dnfr_data(G)
    nodes1 = data1["nodes"]
    G.add_node(2)
    increment_edge_version(G)
    data2 = dynamics._prepare_dnfr_data(G)
    assert data2["nodes"] is not nodes1


def test_node_offset_map_updates_on_node_addition():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    offset0 = operators._node_offset(G, 0)
    mapping1 = G.graph["_node_offset_map"]
    assert offset0 == 0
    offset0_again = operators._node_offset(G, 0)
    assert offset0_again == 0
    assert mapping1 is G.graph["_node_offset_map"]
    G.add_node(2)
    offset2 = operators._node_offset(G, 2)
    mapping2 = G.graph["_node_offset_map"]
    assert mapping2 is not mapping1
    assert offset2 == 2
