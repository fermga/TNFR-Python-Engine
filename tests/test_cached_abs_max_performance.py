import time
import networkx as nx
import pytest

from tnfr.alias import set_attr_with_max, set_attr, recompute_abs_max
from tnfr.constants import ALIAS_VF


@pytest.mark.slow
def test_cached_abs_max_update_performance():
    G_opt = nx.gnp_random_graph(500, 0.1, seed=1)
    G_naive = G_opt.copy()

    for n in G_opt.nodes:
        set_attr_with_max(G_opt, n, ALIAS_VF, 0.0, cache="_vfmax")
        set_attr(G_naive.nodes[n], ALIAS_VF, 0.0)
    recompute_abs_max(G_naive, ALIAS_VF, key="_vfmax")

    nodes = list(G_opt.nodes)
    values = [float(i) for i in range(len(nodes))]

    start = time.perf_counter()
    for n, v in zip(nodes, values):
        set_attr_with_max(G_opt, n, ALIAS_VF, v, cache="_vfmax")
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    for n, v in zip(nodes, values):
        set_attr(G_naive.nodes[n], ALIAS_VF, v)
        recompute_abs_max(G_naive, ALIAS_VF, key="_vfmax")
    t_naive = time.perf_counter() - start

    assert t_opt <= t_naive
