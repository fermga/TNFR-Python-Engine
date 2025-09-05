import networkx as nx

from tnfr.helpers import node_set_checksum


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
