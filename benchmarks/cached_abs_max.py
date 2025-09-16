"""Benchmark cached updates against full recomputation of |νf| maxima."""

import time
import networkx as nx

from tnfr.alias import (
    set_attr_with_max,
    set_attr,
    multi_recompute_abs_max,
)
from tnfr.constants import get_aliases

ALIAS_VF = get_aliases("VF")
ALIAS_MAP = {"_vfmax": ALIAS_VF}


def run():
    """Run the benchmark and print the elapsed times."""
    G_opt = nx.gnp_random_graph(500, 0.1, seed=1)
    G_naive = G_opt.copy()

    for n in G_opt.nodes:
        set_attr_with_max(G_opt, n, ALIAS_VF, 0.0, cache="_vfmax")
        set_attr(G_naive.nodes[n], ALIAS_VF, 0.0)
    multi_recompute_abs_max(G_naive, ALIAS_MAP)

    nodes = list(G_opt.nodes)
    values = [float(i) for i in range(len(nodes))]

    start = time.perf_counter()
    for n, v in zip(nodes, values):
        set_attr_with_max(G_opt, n, ALIAS_VF, v, cache="_vfmax")
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    for n, v in zip(nodes, values):
        set_attr(G_naive.nodes[n], ALIAS_VF, v)
        multi_recompute_abs_max(G_naive, ALIAS_MAP)
    t_naive = time.perf_counter() - start

    print(f"cached update: {t_opt:.6f}s, full recompute: {t_naive:.6f}s")


if __name__ == "__main__":
    run()
