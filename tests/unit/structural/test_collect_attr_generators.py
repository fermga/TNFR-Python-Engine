import networkx as nx
import numpy as np

from tnfr.alias import collect_attr, set_attr
from tnfr.constants import get_aliases

ALIAS_THETA = get_aliases("THETA")


def test_collect_attr_with_generator_numpy():
    G = nx.path_graph(3)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, float(n))

    nodes_gen = (n for n in G.nodes)
    result = collect_attr(G, nodes_gen, ALIAS_THETA, 0.0, np=np)

    assert np.array_equal(result, np.array([0.0, 1.0, 2.0], dtype=float))
