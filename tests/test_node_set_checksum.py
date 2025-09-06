import hashlib
import networkx as nx

from tnfr.helpers import node_set_checksum, _stable_json


def build_graph():
    class Foo:
        def __init__(self, value):
            self.value = value

    G = nx.Graph()
    G.add_node(Foo(1))
    G.add_node(Foo(2))
    return G


def test_node_set_checksum_object_stable():
    checksum1 = node_set_checksum(build_graph())
    checksum2 = node_set_checksum(build_graph())
    assert checksum1 == checksum2


def _reference_checksum(G):
    hasher = hashlib.blake2b(digest_size=16)

    def serialise(n):
        return repr(_stable_json(n))

    serialised = [serialise(n) for n in G.nodes()]
    serialised.sort()
    for i, node_repr in enumerate(serialised):
        if i:
            hasher.update(b"|")
        hasher.update(node_repr.encode("utf-8"))
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
