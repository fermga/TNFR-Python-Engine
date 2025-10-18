"""Pruebas enfocadas en los helpers públicos de cache."""

from tnfr.utils import (
    edge_version_update,
    ensure_node_index_map,
    ensure_node_offset_map,
)


def test_edge_version_update_scopes_mutations(graph_canon):
    G = graph_canon()
    start = int(G.graph.get("_edge_version", 0))
    with edge_version_update(G):
        assert G.graph["_edge_version"] == start + 1
    assert G.graph["_edge_version"] == start + 2


def test_node_offset_map_updates_on_node_addition(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    mapping1 = ensure_node_offset_map(G)
    assert mapping1[0] == 0
    mapping1_again = ensure_node_offset_map(G)
    assert mapping1 is mapping1_again
    G.add_node(2)
    mapping2 = ensure_node_offset_map(G)
    assert mapping2 is not mapping1
    assert mapping2[2] == 2


def test_node_offset_map_updates_on_node_replacement(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    mapping1 = ensure_node_offset_map(G)
    G.remove_node(0)
    G.add_node(2)
    mapping2 = ensure_node_offset_map(G)
    assert mapping2 is not mapping1
    assert 0 not in mapping2 and 2 in mapping2


def test_node_maps_order(graph_canon):
    G = graph_canon()
    G.add_nodes_from([2, 0, 1])
    idx_map = ensure_node_index_map(G)
    assert idx_map == {2: 0, 0: 1, 1: 2}
    G.graph["SORT_NODES"] = True
    offset_map = ensure_node_offset_map(G)
    assert offset_map == {0: 0, 1: 1, 2: 2}
    assert ensure_node_index_map(G) is idx_map
