import gc
import networkx as nx

from tnfr.node import NodoNX
from tnfr.operators import random_jitter


def test_random_jitter_cache_cleared_on_node_removal():
    G = nx.Graph()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    random_jitter(n0, 0.1)
    cache = G.graph.get("_rnd_cache")
    assert cache is not None
    assert len(cache) == 1

    del n0
    G.remove_node(0)
    gc.collect()

    assert len(cache) == 0
