import pytest

import tnfr.import_utils as import_utils
from tnfr import dynamics
from tnfr.helpers.cache import (
    cached_nodes_and_A,
    increment_edge_version,
    edge_version_update,
    ensure_node_offset_map,
    cached_node_list,
    ensure_node_index_map,
    _ensure_node_map,
)


def test_edge_version_update_scopes_mutations(graph_canon):
    G = graph_canon()
    start = int(G.graph.get("_edge_version", 0))
    with edge_version_update(G):
        assert G.graph["_edge_version"] == start + 1
    assert G.graph["_edge_version"] == start + 2


def test_cached_nodes_and_A_reuse_and_invalidate(graph_canon):
    G = graph_canon()
    G.add_edges_from([(0, 1), (1, 2)])
    data1 = dynamics._prepare_dnfr_data(G)
    nodes1 = data1["nodes"]
    data2 = dynamics._prepare_dnfr_data(G)
    assert nodes1 is data2["nodes"]
    G.add_edge(0, 2)
    increment_edge_version(G)
    data3 = dynamics._prepare_dnfr_data(G)
    assert data3["nodes"] is not nodes1


def test_cached_nodes_and_A_invalidate_on_node_addition(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    data1 = dynamics._prepare_dnfr_data(G)
    nodes1 = data1["nodes"]
    G.add_node(2)
    increment_edge_version(G)
    data2 = dynamics._prepare_dnfr_data(G)
    assert data2["nodes"] is not nodes1


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


def test__ensure_node_map_creates_multiple_maps(graph_canon):
    G = graph_canon()
    G.add_nodes_from([2, 0, 1])
    mapping = _ensure_node_map(G, attrs=("idx", "offset"), sort=False)
    cache = G.graph["_node_list_cache"]
    assert mapping is cache.idx
    assert cache.offset == {2: 0, 0: 1, 1: 2}
    assert mapping == {2: 0, 0: 1, 1: 2}


def test_node_maps_order(graph_canon):
    G = graph_canon()
    G.add_nodes_from([2, 0, 1])
    idx_map = ensure_node_index_map(G)
    assert idx_map == {2: 0, 0: 1, 1: 2}
    G.graph["SORT_NODES"] = True
    offset_map = ensure_node_offset_map(G)
    assert offset_map == {0: 0, 1: 1, 2: 2}
    assert ensure_node_index_map(G) is idx_map


def test_cache_node_list_updates_on_dirty(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    nodes1 = cached_node_list(G)
    nodes2 = cached_node_list(G)
    assert nodes1 is nodes2
    G.graph["_node_list_dirty"] = True
    nodes3 = cached_node_list(G)
    assert nodes3 is not nodes1


def test_cache_node_list_invalidate_on_node_replacement(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    nodes1 = cached_node_list(G)
    G.remove_node(0)
    G.add_node(2)
    nodes2 = cached_node_list(G)
    assert nodes2 is not nodes1
    assert set(nodes2) == {1, 2}


def test_cache_node_list_cache_updated_on_node_set_change(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    nodes1 = cached_node_list(G)
    cache1 = G.graph["_node_list_cache"]
    G.add_node(2)
    nodes2 = cached_node_list(G)
    cache2 = G.graph["_node_list_cache"]
    assert nodes2 is not nodes1
    assert cache2 is not cache1
    assert set(nodes2) == {0, 1, 2}
    assert G.graph["_node_list_len"] == 3


def test_cached_nodes_and_A_returns_none_without_numpy(monkeypatch, graph_canon):
    monkeypatch.setattr(import_utils, "cached_import", lambda *a, **k: None)
    monkeypatch.setattr(
        "tnfr.helpers.cache.cached_import", import_utils.cached_import
    )
    G = graph_canon()
    G.add_edge(0, 1)
    nodes, A = cached_nodes_and_A(G)
    assert A is None
    assert nodes == [0, 1]


def test_cached_nodes_and_A_requires_numpy(monkeypatch, graph_canon):
    monkeypatch.setattr(import_utils, "cached_import", lambda *a, **k: None)
    monkeypatch.setattr(
        "tnfr.helpers.cache.cached_import", import_utils.cached_import
    )
    G = graph_canon()
    G.add_edge(0, 1)
    with pytest.raises(RuntimeError):
        cached_nodes_and_A(G, require_numpy=True)
