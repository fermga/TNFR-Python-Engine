import networkx as nx
import json
import hashlib

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


def _old_checksum(G):
    sha1 = hashlib.sha1()

    def serialise(n):
        return json.dumps(_stable_json(n), sort_keys=True, ensure_ascii=False)

    for i, node_repr in enumerate(sorted(serialise(n) for n in G.nodes())):
        if i:
            sha1.update(b"|")
        sha1.update(node_repr.encode("utf-8"))
    return sha1.hexdigest()


def test_node_set_checksum_compatibility():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    assert node_set_checksum(G) == _old_checksum(G)
