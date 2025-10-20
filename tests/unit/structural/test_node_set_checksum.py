import hashlib
import timeit
from unittest.mock import patch

from tnfr.utils import (
    NODE_SET_CHECKSUM_KEY,
    clear_node_repr_cache,
    node_set_checksum,
    stable_json,
)
from tnfr.utils import increment_edge_version


def build_graph(graph_canon):
    G = graph_canon()
    G.add_node(("foo", 1))
    G.add_node(("foo", 2))
    return G


def test_node_set_checksum_object_stable(graph_canon):
    checksum1 = node_set_checksum(build_graph(graph_canon))
    checksum2 = node_set_checksum(build_graph(graph_canon))
    assert checksum1 == checksum2


def _sorting_key(node):
    try:
        return stable_json(node)
    except TypeError:
        return repr(node)


def _reference_checksum(G):
    nodes = sorted(G.nodes(), key=_sorting_key)
    hasher = hashlib.blake2b(digest_size=16)
    for n in nodes:
        d = hashlib.blake2b(
            stable_json(n).encode("utf-8"), digest_size=16
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
    nodes.sort(key=_sorting_key)
    t_unsorted = timeit.timeit(lambda: node_set_checksum(G, nodes), number=1)
    t_presorted = timeit.timeit(
        lambda: node_set_checksum(G, nodes, presorted=True), number=1
    )
    assert t_presorted <= t_unsorted * 3.0


def test_node_set_checksum_no_store_does_not_cache(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    node_set_checksum(G, store=False)
    assert NODE_SET_CHECKSUM_KEY not in G.graph


def test_node_set_checksum_cache_token_is_prefix(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    checksum = node_set_checksum(G)
    token, stored_checksum, nodes_snapshot = G.graph[NODE_SET_CHECKSUM_KEY]
    assert stored_checksum == checksum
    assert token == checksum[:16]
    assert len(token) == 16
    assert nodes_snapshot == frozenset(G.nodes())


def test_node_set_checksum_uses_cached_result_without_rehash(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    node_set_checksum(G)
    with patch("tnfr.utils.cache._node_repr_digest") as mock_digest:
        assert node_set_checksum(G) == G.graph[NODE_SET_CHECKSUM_KEY][1]
        mock_digest.assert_not_called()


def test_increment_edge_version_clears_node_repr_cache(graph_canon):
    nxG = graph_canon()
    nxG.add_nodes_from([1, 2, 3])
    with patch("tnfr.utils.cache.stable_json", wraps=stable_json) as mock_stable_json:
        clear_node_repr_cache()
        node_set_checksum(nxG, store=False)
        first_call_count = mock_stable_json.call_count
        assert first_call_count == len(nxG.nodes())

        node_set_checksum(nxG, store=False)
        assert mock_stable_json.call_count == first_call_count

        increment_edge_version(nxG)
        node_set_checksum(nxG, store=False)
        assert mock_stable_json.call_count == first_call_count + len(nxG.nodes())
