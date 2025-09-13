import time
import networkx as nx
import numpy as np
import pytest

from tnfr.alias import set_attr, collect_attr
from tnfr.constants import get_aliases

ALIAS_THETA = get_aliases("THETA")


def _naive_collect(G):
    return np.array(
        [collect_attr(G, [n], ALIAS_THETA, 0.0, np=np)[0] for n in G.nodes],
        dtype=float,
    )


@pytest.mark.slow
def test_collect_attr_performance():
    G = nx.gnp_random_graph(300, 0.1, seed=1)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)

    start = time.perf_counter()
    for _ in range(5):
        collect_attr(G, G.nodes, ALIAS_THETA, 0.0, np=np)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(5):
        _naive_collect(G)
    t_naive = time.perf_counter() - start

    assert t_opt <= t_naive
