import hashlib
import networkx as nx
import timeit

from tnfr.helpers.cache import (
    node_set_checksum,
    _stable_json,
    increment_edge_version,
    _node_repr,
    _hash_node,
)


def build_graph():
    G = nx.Graph()
    G.add_node(("foo", 1))
    G.add_node(("foo", 2))
    return G


def test_node_set_checksum_object_stable():
    checksum1 = node_set_checksum(build_graph())
    checksum2 = node_set_checksum(build_graph())
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


def test_node_set_checksum_compatibility():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    assert node_set_checksum(G) == _reference_checksum(G)


def test_node_set_checksum_iterable_equivalence():
    G = nx.Graph()
    G.add_nodes_from([3, 1, 2])
    gen = (n for n in G.nodes())
    assert node_set_checksum(G, gen) == node_set_checksum(G)


def test_node_set_checksum_presorted_performance():
    G = nx.Graph()
    G.add_nodes_from(range(1000))
    nodes = list(G.nodes())
    nodes.sort(key=_node_repr)
    t_unsorted = timeit.timeit(lambda: node_set_checksum(G, nodes), number=1)
    t_presorted = timeit.timeit(
        lambda: node_set_checksum(G, nodes, presorted=True), number=1
    )
    assert t_presorted <= t_unsorted * 3.0


def test_node_set_checksum_no_store_does_not_cache():
    G = nx.Graph()
    G.add_nodes_from([1, 2])
    node_set_checksum(G, store=False)
    assert "_node_set_checksum_cache" not in G.graph


def test_node_repr_cache_cleared_on_increment():
    nxG = nx.Graph()
    _node_repr("foo")
    assert _node_repr.cache_info().currsize > 0
    increment_edge_version(nxG)
    assert _node_repr.cache_info().currsize == 0


def test_hash_node_cache_cleared_on_increment():
    nxG = nx.Graph()
    _hash_node(("foo", 1))
    assert _hash_node.cache_info().currsize > 0
    increment_edge_version(nxG)
    assert _hash_node.cache_info().currsize == 0


def test_hash_node_matches_manual():
    obj = ("a", 1)
    manual = hashlib.blake2b(
        _node_repr(obj).encode("utf-8"), digest_size=16
    ).digest()
    assert _hash_node(obj) == manual
