import hashlib
import timeit

from tnfr.helpers.cache import (
    node_set_checksum,
    _stable_json,
    increment_edge_version,
    _node_repr,
    _hash_node,
)


def build_graph(graph_canon):
    G = graph_canon()
    G.add_node(("foo", 1))
    G.add_node(("foo", 2))
    return G


def test_node_set_checksum_object_stable(graph_canon):
    checksum1 = node_set_checksum(build_graph(graph_canon))
    checksum2 = node_set_checksum(build_graph(graph_canon))
    assert checksum1 == checksum2


def _reference_checksum(G):
    nodes = sorted(G.nodes(), key=_node_repr)
    hasher = hashlib.blake2b(digest_size=16)
    for n in nodes:
        d = hashlib.blake2b(
            _stable_json(n).encode("utf-8"), digest_size=16
        ).digest()
        hasher.update(d)
    return hasher.hexdigest()


def test_node_set_checksum_compatibility(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2, 3])
    assert node_set_checksum(G) == _reference_checksum(G)


def test_node_set_checksum_iterable_equivalence(graph_canon):
    G = graph_canon()
    G.add_nodes_from([3, 1, 2])
    gen = (n for n in G.nodes())
    assert node_set_checksum(G, gen) == node_set_checksum(G)


def test_node_set_checksum_presorted_performance(graph_canon):
    G = graph_canon()
    G.add_nodes_from(range(1000))
    nodes = list(G.nodes())
    nodes.sort(key=_node_repr)
    t_unsorted = timeit.timeit(lambda: node_set_checksum(G, nodes), number=1)
    t_presorted = timeit.timeit(
        lambda: node_set_checksum(G, nodes, presorted=True), number=1
    )
    assert t_presorted <= t_unsorted * 3.0


def test_node_set_checksum_no_store_does_not_cache(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    node_set_checksum(G, store=False)
    assert "_node_set_checksum_cache" not in G.graph


def test_node_set_checksum_cache_token_is_prefix(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    checksum = node_set_checksum(G)
    token, stored_checksum = G.graph["_node_set_checksum_cache"]
    assert stored_checksum == checksum
    assert token == checksum[:16]
    assert len(token) == 16


def test_node_repr_cache_cleared_on_increment(graph_canon):
    nxG = graph_canon()
    _node_repr("foo")
    assert _node_repr.cache_info().currsize > 0
    increment_edge_version(nxG)
    assert _node_repr.cache_info().currsize == 0


def test_hash_node_cache_cleared_on_increment(graph_canon):
    nxG = graph_canon()
    obj = ("foo", 1)
    _hash_node(obj, _node_repr(obj))
    assert _hash_node.cache_info().currsize > 0
    increment_edge_version(nxG)
    assert _hash_node.cache_info().currsize == 0


def test_hash_node_matches_manual():
    obj = ("a", 1)
    obj_repr = _node_repr(obj)
    manual = hashlib.blake2b(obj_repr.encode("utf-8"), digest_size=16).digest()
    assert _hash_node(obj, obj_repr) == manual
