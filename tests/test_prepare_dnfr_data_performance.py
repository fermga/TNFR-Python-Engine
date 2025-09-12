import time
import networkx as nx
import pytest

from tnfr.constants import get_aliases
from tnfr.dynamics import _prepare_dnfr_data
from tnfr.helpers.cache import cached_nodes_and_A
from tnfr.alias import get_attr, set_attr

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


def _naive_prepare(G):
    nodes, _ = cached_nodes_and_A(G, cache_size=1)
    theta = [get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in nodes]
    epi = [get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes]
    vf = [get_attr(G.nodes[n], ALIAS_VF, 0.0) for n in nodes]
    return theta, epi, vf


@pytest.mark.slow
def test_prepare_dnfr_data_performance():
    G = nx.gnp_random_graph(300, 0.1, seed=1)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_EPI, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)

    start = time.perf_counter()
    for _ in range(5):
        _prepare_dnfr_data(G)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(5):
        _naive_prepare(G)
    t_naive = time.perf_counter() - start

    assert t_opt <= t_naive
