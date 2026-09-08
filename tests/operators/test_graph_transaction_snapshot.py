"""Direct rollback coverage for mutable graph runtime state."""

from __future__ import annotations

from collections import deque

import networkx as nx
import pytest

from tnfr.operators.network_stage import GraphTransactionSnapshot


def test_snapshot_restores_array_shapes_dtypes_and_graph_node_aliases() -> None:
    np = pytest.importorskip("numpy")
    graph = nx.Graph()
    direct = np.array([1.0, 2.0])
    tuple_member = np.array([3.0, 4.0])
    tuple_cache = ("marker", tuple_member)
    node_only = np.array([5.0, 6.0])
    graph.add_node(
        0,
        direct_cache_alias=direct,
        tuple_cache_alias=tuple_member,
        node_only_array=node_only,
        marker="before",
    )
    graph.graph.update(
        _direct_cache=direct,
        _tuple_cache=tuple_cache,
        ordinary_marker="before",
    )
    snapshot = GraphTransactionSnapshot(graph)

    direct.dtype = np.int32
    direct.resize((5,), refcheck=False)
    direct[:] = 9
    direct.flags.writeable = False
    tuple_member.resize((3,), refcheck=False)
    tuple_member[:] = 8.0
    node_only.resize((3,), refcheck=False)
    node_only[:] = 7.0
    graph.nodes[0]["marker"] = "after"
    graph.graph["ordinary_marker"] = "after"

    snapshot.restore(graph)

    assert graph.graph["_direct_cache"] is direct
    assert graph.nodes[0]["direct_cache_alias"] is direct
    assert direct.shape == (2,)
    assert direct.dtype == np.dtype(float)
    assert direct.flags.writeable
    assert np.array_equal(direct, np.array([1.0, 2.0]))
    assert graph.graph["_tuple_cache"] is tuple_cache
    assert graph.graph["_tuple_cache"][1] is tuple_member
    assert graph.nodes[0]["tuple_cache_alias"] is tuple_member
    assert tuple_member.shape == (2,)
    assert np.array_equal(tuple_member, np.array([3.0, 4.0]))
    restored_node_only = graph.nodes[0]["node_only_array"]
    assert restored_node_only.shape == (2,)
    assert np.array_equal(restored_node_only, np.array([5.0, 6.0]))
    assert graph.nodes[0]["marker"] == "before"
    assert graph.graph["ordinary_marker"] == "before"


def test_snapshot_rebinds_array_when_tuple_member_refuses_resize() -> None:
    np = pytest.importorskip("numpy")

    class RefusesResize(np.ndarray):
        def resize(self, *args, **kwargs) -> None:
            raise RuntimeError("in-place resize refused")

    array = np.ndarray.__new__(RefusesResize, shape=(2,), dtype=float)
    array[:] = (1.0, 2.0)
    tuple_cache = ("marker", array)
    graph = nx.Graph()
    graph.add_node(0, array_alias=array, marker="before")
    graph.graph.update(
        _tuple_cache=tuple_cache,
        ordinary_marker="before",
    )
    snapshot = GraphTransactionSnapshot(graph)

    np.ndarray.resize(array, (3,), refcheck=False)
    array[:] = 9.0
    graph.nodes[0]["marker"] = "after"
    graph.graph["ordinary_marker"] = "after"

    snapshot.restore(graph)

    restored_tuple = graph.graph["_tuple_cache"]
    restored_array = restored_tuple[1]
    assert restored_tuple is not tuple_cache
    assert restored_array is not array
    assert isinstance(restored_array, RefusesResize)
    assert restored_array.shape == (2,)
    assert np.array_equal(restored_array, np.array([1.0, 2.0]))
    assert graph.nodes[0]["array_alias"] is restored_array
    assert graph.nodes[0]["marker"] == "before"
    assert graph.graph["ordinary_marker"] == "before"


def test_snapshot_restores_standard_container_size_and_type_changes() -> None:
    list_cache = [1, 2]
    dict_cache = {"a": 1}
    set_cache = {1, 2}
    deque_cache = deque((1, 2), maxlen=3)
    graph = nx.Graph()
    graph.graph.update(
        list_cache=list_cache,
        dict_cache=dict_cache,
        set_cache=set_cache,
        deque_cache=deque_cache,
    )
    snapshot = GraphTransactionSnapshot(graph)

    list_cache[:] = ["changed"]
    dict_cache.clear()
    dict_cache["changed"] = 9
    set_cache.clear()
    set_cache.add(9)
    deque_cache.clear()
    deque_cache.append(9)
    graph.graph["list_cache"] = {"replacement": True}
    graph.graph["dict_cache"] = ["replacement"]
    graph.graph["set_cache"] = ("replacement",)
    graph.graph["deque_cache"] = deque((9,), maxlen=1)

    snapshot.restore(graph)

    assert graph.graph["list_cache"] is list_cache
    assert graph.graph["dict_cache"] is dict_cache
    assert graph.graph["set_cache"] is set_cache
    assert graph.graph["deque_cache"] is deque_cache
    assert list_cache == [1, 2]
    assert dict_cache == {"a": 1}
    assert set_cache == {1, 2}
    assert tuple(deque_cache) == (1, 2)
    assert deque_cache.maxlen == 3