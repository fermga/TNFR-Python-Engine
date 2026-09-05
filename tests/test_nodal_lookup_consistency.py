"""Node lookup preserves ordered structure without repeated digest work."""

import gc
import hashlib
import pickle
import weakref
from dataclasses import dataclass, field

import networkx as nx
import pytest

from tnfr.node import NodeNX
from tnfr.operators.jitter import random_jitter
from tnfr.utils import cache


def _integer_checksum(nodes):
    hasher = hashlib.blake2b(digest_size=16)
    for node in nodes:
        hasher.update(hashlib.blake2b(str(node).encode(), digest_size=16).digest())
    return hasher.hexdigest()


def test_warm_offset_sweep_does_not_rehash_unchanged_nodes(monkeypatch):
    graph = nx.empty_graph(1100)
    expected = cache.ensure_node_offset_map(graph)
    calls = 0
    original = cache._iter_node_digests

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "_iter_node_digests", counted)
    for node in (0, 17, 1099):
        assert cache.ensure_node_offset_map(graph)[node] == expected[node]
    assert calls == 0


@pytest.mark.parametrize("seed_record", ["explicit", "legacy_pair", "legacy_set"])
def test_checksum_snapshot_is_reused_after_explicit_or_legacy_population(monkeypatch, seed_record):
    graph = nx.path_graph(5)
    expected = cache.node_set_checksum(graph, tuple(graph))
    if seed_record == "legacy_pair":
        graph.graph[cache.NODE_SET_CHECKSUM_KEY] = (expected[:16], expected)
    elif seed_record == "legacy_set":
        graph.graph[cache.NODE_SET_CHECKSUM_KEY] = (expected[:16], expected, frozenset(graph))
    assert cache.node_set_checksum(graph) == expected

    def reject_rehash(*args, **kwargs):
        raise AssertionError("unchanged checksum must reuse its validated snapshot")

    monkeypatch.setattr(cache, "_iter_node_digests", reject_rehash)
    assert cache.node_set_checksum(graph) == expected
    assert cache.node_set_checksum(graph, iter(graph)) == expected


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_remove_reinsert_updates_order_and_offsets_without_size_change(graph_type):
    graph = graph_type()
    graph.add_nodes_from([0, 1, 2])
    assert cache.cached_node_list(graph) == (0, 1, 2)
    cache.ensure_node_offset_map(graph)
    graph.remove_node(0)
    graph.add_node(0)
    assert cache.cached_node_list(graph) == (1, 2, 0)
    assert cache.ensure_node_offset_map(graph) == {1: 0, 2: 1, 0: 2}
    assert cache.ensure_node_index_map(graph) == {1: 0, 2: 1, 0: 2}


def test_same_size_replacement_is_detected():
    graph = nx.path_graph(3)
    cache.ensure_node_offset_map(graph)
    graph.remove_node(1)
    graph.add_node(9)
    assert cache.cached_node_list(graph) == (0, 2, 9)
    assert cache.ensure_node_offset_map(graph) == {0: 0, 2: 1, 9: 2}


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_equal_cross_type_replacement_returns_actual_label_and_checksum(replacement):
    graph = nx.Graph()
    graph.add_node(1)
    cache.ensure_node_offset_map(graph)
    previous = cache.node_set_checksum(graph)
    graph.remove_node(1)
    graph.add_node(replacement)
    assert cache.cached_node_list(graph)[0] is replacement
    assert next(iter(cache.ensure_node_offset_map(graph))) is replacement
    current = cache.node_set_checksum(graph)
    assert current != previous
    fresh = nx.Graph()
    fresh.add_node(replacement)
    cache.clear_node_repr_cache()
    assert current == cache.node_set_checksum(fresh)


@pytest.mark.parametrize("first,second", [(True, 1.0), (1.0, True), (1, True), (1, 1.0)])
def test_digest_cache_does_not_mix_equal_cross_type_labels(first, second):
    cache.clear_node_repr_cache()
    first_repr, first_digest = cache._node_repr_digest(first)
    second_repr, second_digest = cache._node_repr_digest(second)
    assert first_repr != second_repr
    assert first_digest != second_digest


@pytest.mark.parametrize("first,second", [((True,), (1.0,)),
                                         ((1, (True,)), (1.0, (1.0,)))])
def test_equal_nested_tuple_labels_keep_distinct_serialization(first, second):
    assert first == second
    cache.clear_node_repr_cache()
    first_repr, first_digest = cache._node_repr_digest(first)
    second_repr, second_digest = cache._node_repr_digest(second)
    assert first_repr != second_repr
    assert first_digest != second_digest


@dataclass(frozen=True)
class EqualImmutableLabel:
    value: int
    display: str = field(compare=False)


def test_equal_immutable_custom_label_replacement_keeps_new_representation():
    old = EqualImmutableLabel(1, "old")
    new = EqualImmutableLabel(1, "new")
    assert old == new and hash(old) == hash(new)
    graph = nx.Graph()
    graph.add_node(old)
    before = cache.node_set_checksum(graph)
    cache.ensure_node_offset_map(graph)
    graph.remove_node(old)
    graph.add_node(new)
    assert cache.cached_node_list(graph)[0] is new
    assert cache.node_set_checksum(graph) != before
    assert cache._node_repr_digest(new) == cache._node_repr_digest.__wrapped__(new)


def test_identity_digest_cache_retains_bounded_diagnostics_interfaces():
    cache.clear_node_repr_cache()
    node = EqualImmutableLabel(1, "same object")
    assert cache._node_repr_digest(node) == cache._node_repr_digest(node)
    assert cache._node_repr_digest.cache_info().hits == 1
    for number in range(1030):
        cache._node_repr_digest(EqualImmutableLabel(number, str(number)))
    assert cache._node_repr_digest.cache_info().currsize == 1024
    assert cache._node_repr_digest.cache_parameters() == {"maxsize": 1024, "typed": True}
    cache._node_repr_digest.cache_clear()
    assert cache._node_repr_digest.cache_info().currsize == 0


def test_explicit_equal_value_snapshots_recompute_for_new_node_objects(monkeypatch):
    graph = nx.Graph()
    old = int("1000")
    new = int("1000")
    assert old == new and old is not new
    expected = cache.node_set_checksum(graph, (old,))
    calls = 0
    original = cache._iter_node_digests

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "_iter_node_digests", counted)
    assert cache.node_set_checksum(graph, (new,)) == expected
    assert calls == 1


class SameRepresentation:
    def __repr__(self):
        return "indistinguishable-label"


def test_distinct_equal_representation_nodes_do_not_hide_replacement():
    first, second, replacement = SameRepresentation(), SameRepresentation(), SameRepresentation()
    graph = nx.Graph()
    graph.add_nodes_from([first, second])
    cache.ensure_node_offset_map(graph)
    graph.remove_node(first)
    graph.add_node(replacement)
    assert cache.cached_node_list(graph) == (second, replacement)
    assert cache.ensure_node_offset_map(graph) == {second: 0, replacement: 1}


@pytest.mark.parametrize("nodes,sorted_nodes", [([2, 0, 1], [0, 1, 2]),
                                               ([10, 2, 1], [1, 10, 2]),
                                               ([2, "a", 1], ["a", 1, 2])])
def test_sort_policy_toggle_rebuilds_offsets_only(nodes, sorted_nodes):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    original_idx = cache.ensure_node_index_map(graph)
    for sort, order in [(False, nodes), (True, sorted_nodes), (False, nodes)]:
        graph.graph["SORT_NODES"] = sort
        assert cache.ensure_node_offset_map(graph) == dict(zip(order, range(len(order))))
        assert cache.cached_node_list(graph) == tuple(nodes)
        assert cache.ensure_node_index_map(graph) is original_idx


def test_graph_copy_detaches_node_maps_and_sort_mode():
    graph = nx.Graph()
    graph.add_nodes_from([2, 0, 1])
    source_offsets = cache.ensure_node_offset_map(graph)
    source_cache = graph.graph["_node_list_cache"]
    copied = graph.copy()
    copied.graph["SORT_NODES"] = True
    assert cache.ensure_node_offset_map(copied) == {0: 0, 1: 1, 2: 2}
    assert copied.graph["_node_list_cache"] is not source_cache
    assert cache.ensure_node_offset_map(graph) is source_offsets
    assert source_offsets == {2: 0, 0: 1, 1: 2}


def test_node_constructor_on_copy_preserves_original_adapter_cache():
    graph = nx.path_graph(2)
    original = NodeNX(graph, 0)
    original_cache = graph.graph["_node_cache"]
    copied = graph.copy()
    detached = NodeNX(copied, 0)
    assert original_cache[0] is original
    assert copied.graph["_node_cache"] is not original_cache
    assert NodeNX.from_graph(copied, 0) is detached
    assert NodeNX.from_graph(graph, 0) is original


@pytest.mark.parametrize("factory", [NodeNX, NodeNX.from_graph])
def test_empty_adapter_cache_copy_does_not_insert_into_parent(factory):
    graph = nx.path_graph(2)
    graph.graph["_node_cache"] = {}
    copied = graph.copy()
    created = factory(copied, 0)
    assert graph.graph["_node_cache"] == {}
    assert copied.graph["_node_cache"][0] is created


def test_filtered_view_checks_visible_nodes_and_parent_remains_correct():
    graph = nx.empty_graph(4)
    visible = {0, 1}
    view = nx.subgraph_view(graph, filter_node=lambda node: node in visible)
    assert cache.cached_node_list(graph) == (0, 1, 2, 3)
    assert cache.ensure_node_offset_map(view) == {0: 0, 1: 1}
    visible.remove(0)
    visible.add(3)
    assert cache.cached_node_list(view) == (1, 3)
    assert cache.ensure_node_offset_map(view) == {1: 0, 3: 1}
    assert cache.cached_node_list(graph) == (0, 1, 2, 3)
    assert cache.ensure_node_offset_map(graph) == {0: 0, 1: 1, 2: 2, 3: 3}


@pytest.mark.parametrize("initial_presorted", [False, True])
def test_checksum_mode_and_order_are_part_of_cache_identity(initial_presorted):
    graph = nx.Graph()
    graph.add_nodes_from([2, 0, 1])
    cache.node_set_checksum(graph, presorted=initial_presorted)
    assert cache.node_set_checksum(graph, presorted=False) == _integer_checksum([0, 1, 2])
    assert cache.node_set_checksum(graph, presorted=True) == _integer_checksum([2, 0, 1])
    graph.remove_node(2)
    graph.add_node(2)
    assert cache.node_set_checksum(graph, presorted=True) == _integer_checksum([0, 1, 2])


def test_explicit_subset_checksum_cannot_replace_full_graph_checksum():
    graph = nx.path_graph(3)
    cache.node_set_checksum(graph, (0, 1))
    assert cache.node_set_checksum(graph) == _integer_checksum([0, 1, 2])


@pytest.mark.parametrize("explicit_nodes", [False, True])
def test_store_false_removes_record_even_on_cache_hit(explicit_nodes):
    graph = nx.path_graph(3)
    expected = cache.node_set_checksum(graph)
    nodes = tuple(graph) if explicit_nodes else None
    assert cache.node_set_checksum(graph, nodes, store=False) == expected
    assert cache.NODE_SET_CHECKSUM_KEY not in graph.graph


def test_external_cache_reference_does_not_keep_graph_alive():
    graph = nx.path_graph(3)
    cache.cached_node_list(graph)
    node_cache = graph.graph["_node_list_cache"]
    ref = weakref.ref(graph)
    del graph
    gc.collect()
    assert ref() is None
    assert node_cache.nodes == (0, 1, 2)


def test_node_lookup_cache_remains_picklable_and_rebinds_after_restore():
    graph = nx.path_graph(3)
    cache.ensure_node_offset_map(graph)
    restored = pickle.loads(pickle.dumps(graph))
    inherited = restored.graph["_node_list_cache"]
    assert cache.ensure_node_offset_map(restored) == {0: 0, 1: 1, 2: 2}
    assert restored.graph["_node_list_cache"] is not inherited
    assert restored.graph["_node_list_cache"].owner() is restored
    assert graph.graph["_node_list_cache"].owner() is graph


@pytest.mark.parametrize("sort", [False, True])
def test_jitter_reordered_graph_matches_fresh_current_order(sort):
    graph = nx.Graph(RANDOM_SEED=7, SORT_NODES=sort)
    graph.add_nodes_from([2, 0, 1])
    cache.ensure_node_offset_map(graph)
    graph.remove_node(2)
    graph.add_node(2)
    fresh = nx.Graph(RANDOM_SEED=7, SORT_NODES=sort)
    fresh.add_nodes_from(graph)
    for node in graph:
        assert random_jitter(NodeNX(graph, node), 0.1) == random_jitter(NodeNX(fresh, node), 0.1)
