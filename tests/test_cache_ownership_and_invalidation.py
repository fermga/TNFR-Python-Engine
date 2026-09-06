"""Cache results must preserve graph ownership and explicit request semantics."""

from pathlib import Path
import pickle

import networkx as nx
import numpy as np
import pytest

from tnfr.cache import (
    CacheLevel, GraphChangeTracker, PersistentTNFRCache, TNFRHierarchicalCache,
    cache_tnfr_computation, cached_nodes_and_A, edge_version_cache,
    increment_edge_version, invalidate_function_cache,
)
from tnfr.metrics.buffer_cache import ensure_numpy_buffers
from tnfr.utils.unified_cache import UnifiedLRUCache


def test_copied_graph_does_not_reuse_original_cache_or_scratch_arrays():
    original = nx.path_graph(3)
    first = ensure_numpy_buffers(original, key_prefix="ownership", count=3, buffer_count=1)
    first[0][:] = 1.0
    copied = original.copy()
    second = ensure_numpy_buffers(copied, key_prefix="ownership", count=3, buffer_count=1)
    second[0][:] = 2.0
    assert second[0] is not first[0]
    np.testing.assert_array_equal(first[0], np.ones(3))
    assert original.graph["_tnfr_cache_manager"] is not copied.graph["_tnfr_cache_manager"]


def test_invalidation_of_copy_preserves_original_cache():
    original = nx.path_graph(3)
    value = edge_version_cache(original, "ownership", object)
    copied = original.copy()
    increment_edge_version(copied)
    assert edge_version_cache(original, "ownership", object) is value


def test_graph_view_does_not_reuse_parent_graph_result():
    graph = nx.path_graph(3)
    assert edge_version_cache(graph, "size", lambda: len(graph)) == 3
    view = graph.subgraph([0, 1])
    assert edge_version_cache(view, "size", lambda: len(view)) == 2
    assert edge_version_cache(graph, "size", lambda: len(graph)) == 3


@pytest.mark.parametrize("weak", [False, True])
def test_copied_node_adapter_binds_to_the_requested_graph(weak):
    from tnfr.node import NodeNX

    graph = nx.path_graph(2)
    original = NodeNX.from_graph(graph, 0, use_weak_cache=weak)
    copied = graph.copy()
    detached = NodeNX.from_graph(copied, 0, use_weak_cache=weak)
    assert detached.G is copied
    assert detached is not original
    assert NodeNX.from_graph(graph, 0, use_weak_cache=weak) is original


def test_operator_on_copy_does_not_mutate_original_through_cached_adapter():
    from tnfr.node import NodeNX
    from tnfr.operators import apply_glyph_obj
    from tnfr.types import Glyph

    graph = nx.Graph()
    graph.add_node(0, EPI=0.5, **{"νf": 1.0, "ΔNFR": 0.2, "θ": 0.0})
    NodeNX.from_graph(graph, 0)
    copied = graph.copy()
    apply_glyph_obj(NodeNX.from_graph(copied, 0), Glyph.IL)
    assert graph.nodes[0]["ΔNFR"] == 0.2
    assert abs(copied.nodes[0]["ΔNFR"]) < 0.2


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, np.dtype([("x", "i4")])])
def test_buffer_key_includes_dtype(dtype):
    graph = nx.path_graph(2)
    first = ensure_numpy_buffers(graph, key_prefix="typed", count=2, buffer_count=1)
    second = ensure_numpy_buffers(graph, key_prefix="typed", count=2, buffer_count=1, dtype=dtype)
    assert second[0].dtype == np.dtype(dtype)
    assert first[0].dtype == np.dtype(float)


def test_equivalent_buffer_dtype_requests_reuse_storage():
    graph = nx.Graph()
    first = ensure_numpy_buffers(graph, key_prefix="typed", count=2, buffer_count=1, dtype="f8")
    second = ensure_numpy_buffers(graph, key_prefix="typed", count=2, buffer_count=1, dtype=np.float64)
    assert first is second


@pytest.mark.parametrize("sparse_first", [True, False])
def test_adjacency_key_distinguishes_sparse_and_dense_requests(sparse_first):
    graph = nx.path_graph(3)
    cached_nodes_and_A(graph, prefer_sparse=sparse_first)
    nodes, adjacency = cached_nodes_and_A(graph, prefer_sparse=not sparse_first)
    assert nodes == (0, 1, 2)
    if sparse_first:
        np.testing.assert_array_equal(adjacency, nx.to_numpy_array(graph, weight=None))
    else:
        assert adjacency is None


def test_adjacency_key_preserves_explicit_node_order():
    graph = nx.path_graph(3)
    cached_nodes_and_A(graph, nodes=(0, 1, 2))
    nodes, adjacency = cached_nodes_and_A(graph, nodes=(1, 2, 0))
    assert nodes == (1, 2, 0)
    np.testing.assert_array_equal(adjacency, nx.to_numpy_array(graph, nodelist=nodes, weight=None))


@pytest.mark.parametrize("weighted", [False, True])
def test_lru_clear_releases_capacity_and_removal_resources(weighted):
    removed = []
    locks = {"old": object()}
    cache = UnifiedLRUCache(maxsize=2, getsizeof=len if weighted else None,
                            locks=locks, eviction_callbacks=lambda k, v: removed.append(k))
    cache["old"] = "ab"
    cache.clear()
    assert cache.currsize == 0
    assert cache.get_stats().size == 0
    assert locks == {}
    assert removed == ["old"]
    cache["new"] = "cd"
    assert cache["new"] == "cd"


@pytest.mark.parametrize("disk_flag,level", [(False, CacheLevel.GRAPH_STRUCTURE), (True, CacheLevel.TEMPORARY)])
def test_persistent_memory_only_write(disk_flag, level, tmp_path):
    cache = PersistentTNFRCache(cache_dir=tmp_path)
    cache.set_persistent("key", 42, level, {"graph_topology"}, persist_to_disk=disk_flag)
    assert cache.get_persistent("key", level) == 42
    assert not list(tmp_path.rglob("*.pkl"))


def test_persistent_invalidation_survives_memory_clear_and_reopen(tmp_path):
    cache = PersistentTNFRCache(cache_dir=tmp_path)
    level = CacheLevel.GRAPH_STRUCTURE
    cache.set_persistent("key", 42, level, {"graph_topology"})
    assert cache.invalidate_by_dependency("graph_topology") == 1
    assert cache.get_persistent("key", level) is None
    reopened = PersistentTNFRCache(cache_dir=tmp_path)
    assert reopened.get_persistent("key", level) is None


def test_memory_only_replacement_cannot_revive_an_older_disk_snapshot(tmp_path):
    cache = PersistentTNFRCache(cache_dir=tmp_path)
    level = CacheLevel.GRAPH_STRUCTURE
    cache.set_persistent("key", 1, level, set())
    cache.set_persistent("key", 2, level, set(), persist_to_disk=False)
    assert cache.get_persistent("key", level) == 2
    assert PersistentTNFRCache(cache_dir=tmp_path).get_persistent("key", level) is None


@pytest.mark.parametrize("dependencies", [set(), {"node_epi"}])
def test_function_invalidation_targets_its_custom_cache_only(dependencies):
    cache = TNFRHierarchicalCache()
    calls = {"first": 0, "second": 0}

    @cache_tnfr_computation(CacheLevel.TEMPORARY, dependencies, cache_instance=cache)
    def first(value):
        calls["first"] += 1
        return value

    @cache_tnfr_computation(CacheLevel.TEMPORARY, dependencies, cache_instance=cache)
    def second(value):
        calls["second"] += 1
        return value

    first(1)
    second(1)
    assert invalidate_function_cache(first) == 1
    first(1)
    second(1)
    assert calls == {"first": 2, "second": 1}


def test_keyword_named_graph_participates_in_dependency_hash():
    cache = TNFRHierarchicalCache()

    @cache_tnfr_computation(CacheLevel.TEMPORARY, {"node_epi"}, cache_instance=cache)
    def read(*, network):
        return network.nodes[0]["EPI"]

    graph = nx.Graph()
    graph.add_node(0, EPI=1.0)
    assert read(network=graph) == 1.0
    graph.nodes[0]["EPI"] = 2.0
    assert read(network=graph) == 2.0


def test_separately_constructed_cached_closures_have_distinct_values():
    cache = TNFRHierarchicalCache()

    def make_reader(value):
        @cache_tnfr_computation(CacheLevel.TEMPORARY, set(), cache_instance=cache)
        def read():
            return value
        return read

    first, second = make_reader(1), make_reader(2)
    assert first() == 1
    assert second() == 2


@pytest.mark.parametrize("mutation", ["add_nodes_from", "add_edges_from", "remove_nodes_from", "remove_edges_from", "clear_edges", "clear"])
def test_graph_tracker_covers_bulk_topology_mutation(mutation):
    graph = nx.path_graph(3)
    cache = TNFRHierarchicalCache()
    GraphChangeTracker(cache).track_graph_changes(graph)
    cache.set("metric", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})
    arguments = {"add_nodes_from": ([3, 4],), "add_edges_from": ([(0, 2)],),
                 "remove_nodes_from": ([2],), "remove_edges_from": ([(0, 1)],),
                 "clear_edges": (), "clear": ()}
    getattr(graph, mutation)(*arguments[mutation])
    assert cache.get("metric", CacheLevel.GRAPH_STRUCTURE) is None


def test_graph_tracker_preserves_multigraph_key_arguments_and_return():
    graph = nx.MultiGraph()
    tracker = GraphChangeTracker(TNFRHierarchicalCache())
    tracker.track_graph_changes(graph)
    assert graph.add_edge(0, 1, "named", weight=2) == "named"
    graph.remove_edge(0, 1, "named")
    assert graph.number_of_edges() == 0


def test_tracker_invalidates_a_partially_applied_bulk_operation():
    graph = nx.Graph()
    cache = TNFRHierarchicalCache()
    tracker = GraphChangeTracker(cache)
    tracker.track_graph_changes(graph)
    cache.set("metric", 0, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})
    with pytest.raises(nx.NetworkXError):
        graph.add_edges_from([(0, 1), (2,)])
    assert graph.has_edge(0, 1)
    assert cache.get("metric", CacheLevel.GRAPH_STRUCTURE) is None


def test_hot_path_configuration_is_owned_and_merged_across_calls():
    from tnfr.cache import configure_hot_path_caches

    graph = nx.Graph()
    configure_hot_path_caches(graph, buffer_max_entries=256)
    configure_hot_path_caches(graph, trig_cache_size=16)
    assert graph.graph["_tnfr_cache_config"]["overrides"]["_edge_version_state"] == 256
    copied = graph.copy()
    configure_hot_path_caches(copied, buffer_max_entries=512)
    assert graph.graph["_cache_config"]["buffer_max_entries"] == 256
    assert copied.graph["_cache_config"]["buffer_max_entries"] == 512


def test_graph_tracker_invalidates_canonical_property_dependency():
    cache = TNFRHierarchicalCache()
    tracker = GraphChangeTracker(cache)
    cache.set("metric", 1, CacheLevel.NODE_PROPERTIES, {"node_phase"})
    tracker.on_node_property_change(0, "theta", 0.0, 1.0)
    assert cache.get("metric", CacheLevel.NODE_PROPERTIES) is None


def test_memory_profile_reports_retained_bytes_by_level_and_invalidation():
    cache = TNFRHierarchicalCache(max_memory_mb=1)
    cache.set("temporary", list(range(16)), CacheLevel.TEMPORARY, {"node_epi"})
    cache.set("derived", list(range(32)), CacheLevel.DERIVED_METRICS, {"node_phase"})
    warm = cache.memory_profile()
    assert warm["retained_bytes"] > 0
    assert warm["retained_bytes_by_level"][CacheLevel.TEMPORARY.value] > 0
    assert warm["retained_bytes_by_level"][CacheLevel.DERIVED_METRICS.value] > 0
    assert cache.invalidate_by_dependency("node_epi") == 1
    cold = cache.memory_profile()
    assert cold["retained_bytes"] < warm["retained_bytes"]
    assert cold["retained_bytes_by_level"][CacheLevel.TEMPORARY.value] == 0


class _WriteMarkerOnUnpickle:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return Path.write_text, (Path(self.path), "outer pickle executed")


@pytest.mark.parametrize("extension", [b"\x82\x01", b"\x83\x00\x01", b"\x84\x00\x00\x01\x00"])
def test_secure_shelve_rejects_extension_opcodes_before_unpickling(tmp_path, monkeypatch, extension):
    from tnfr.cache import SecurityError, create_secure_shelve_layer
    from tnfr.utils import cache_layers

    layer = create_secure_shelve_layer(str(tmp_path / "extensions"), secret=b"secret")
    calls = []

    def unexpected_unpickler(*args, **kwargs):
        calls.append(True)
        raise AssertionError("outer unpickler must not receive extension opcodes")

    monkeypatch.setattr(cache_layers, "_EnvelopeUnpickler", unexpected_unpickler)
    try:
        # A minimal opcode probe, without registering or constructing a callable.
        layer._shelf.dict[b"extension"] = b"\x80\x04" + extension + b"."
        with pytest.raises(SecurityError):
            layer.load("extension")
        assert calls == []
    finally:
        layer.close()


def test_secure_shelve_rejects_outer_pickle_before_execution(tmp_path):
    from tnfr.cache import SecurityError, create_secure_shelve_layer

    marker = tmp_path / "unpickle-marker.txt"
    layer = create_secure_shelve_layer(str(tmp_path / "secure-cache"), secret=b"test-secret")
    try:
        # Simulate tampering with the outer shelve bytes; the payload is harmless.
        layer._shelf.dict[b"tampered"] = pickle.dumps(_WriteMarkerOnUnpickle(marker))
        with pytest.raises(SecurityError):
            layer.load("tampered")
        assert not marker.exists()
    finally:
        layer.close()


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_secure_shelve_roundtrip_bytes_and_objects(tmp_path, protocol):
    from tnfr.cache import create_secure_shelve_layer

    layer = create_secure_shelve_layer(str(tmp_path / "secure-cache"), secret=b"test-secret", protocol=protocol)
    try:
        for name, value in [("bytes", b"payload"), ("object", {"value": [1, 2]})]:
            layer.store(name, value)
            assert layer.load(name) == value
    finally:
        layer.close()


class _RedisMemoryClient:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)


def test_secure_redis_requires_signed_bytes_even_with_nonbinary_client():
    from tnfr.cache import SecurityError, create_secure_redis_layer

    client = _RedisMemoryClient()
    layer = create_secure_redis_layer(client, secret=b"secret")
    client.data[layer._format_key("unsigned")] = "unverified text"
    with pytest.raises(SecurityError):
        layer.load("unsigned")


def test_secure_redis_roundtrip_preserves_raw_bytes_and_objects():
    from tnfr.cache import create_secure_redis_layer

    layer = create_secure_redis_layer(_RedisMemoryClient(), secret=b"secret")
    for name, value in [("bytes", b"raw data"), ("object", {"value": [1, 2]})]:
        layer.store(name, value)
        assert layer.load(name) == value


@pytest.mark.parametrize("backend", ["shelve", "redis"])
def test_signature_authenticates_raw_versus_pickle_interpretation(backend, tmp_path):
    from tnfr.cache import SecurityError, create_secure_redis_layer, create_secure_shelve_layer
    from tnfr.utils.cache_layers import _SIGNATURE_PREFIX

    marker = tmp_path / "mode-marker.txt"
    client = _RedisMemoryClient()
    layer = (create_secure_shelve_layer(str(tmp_path / "signed"), secret=b"secret")
             if backend == "shelve" else create_secure_redis_layer(client, secret=b"secret"))
    try:
        # Signing raw bytes grants no permission to execute them as a pickle.
        layer.store("raw", pickle.dumps(_WriteMarkerOnUnpickle(marker)))
        if backend == "shelve":
            envelope = layer._shelf["raw"]
        else:
            envelope = client.data[layer._format_key("raw")]
        offset = len(_SIGNATURE_PREFIX)
        tampered = envelope[:offset] + b"\x01" + envelope[offset + 1:]
        if backend == "shelve":
            layer._shelf["raw"] = tampered
        else:
            client.data[layer._format_key("raw")] = tampered
        with pytest.raises(SecurityError):
            layer.load("raw")
        assert not marker.exists()
    finally:
        layer.close()


def test_unsigned_trusted_shelve_keeps_legacy_object_roundtrip(tmp_path, monkeypatch):
    from tnfr.cache import ShelveCacheLayer

    monkeypatch.setenv("TNFR_ALLOW_UNSIGNED_PICKLE", "1")
    layer = ShelveCacheLayer(str(tmp_path / "legacy"))
    try:
        layer.store("object", {"value": [1, 2]})
        assert layer.load("object") == {"value": [1, 2]}
    finally:
        layer.close()


def test_old_signed_envelopes_are_rejected_for_rebuilding(tmp_path):
    from tnfr.cache import SecurityError, create_hmac_signer, create_secure_shelve_layer

    payload = pickle.dumps({"legacy": True})
    signature = create_hmac_signer(b"secret")(payload)
    envelope = b"TNFRSIG1\x01" + len(signature).to_bytes(4, "big") + signature + payload
    layer = create_secure_shelve_layer(str(tmp_path / "legacy-signed"), secret=b"secret")
    try:
        layer._shelf["old"] = envelope
        with pytest.raises(SecurityError, match="legacy"):
            layer.load("old")
    finally:
        layer.close()
