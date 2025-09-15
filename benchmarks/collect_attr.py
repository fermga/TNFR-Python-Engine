"""Benchmark for :func:`tnfr.alias.collect_attr` performance.

The optimized path relies on NumPy. When the optional dependency is not
available the script prints an informative message and exits without
executing the benchmark.
"""

import time
import networkx as nx

from tnfr.alias import set_attr, collect_attr
from tnfr.constants import get_aliases
from tnfr.import_utils import cached_import

np = cached_import("numpy")

ALIAS_THETA = get_aliases("THETA")


def _naive_collect(G):
    values = [collect_attr(G, [n], ALIAS_THETA, 0.0)[0] for n in G.nodes]
    if np is not None:
        return np.array(values, dtype=float)
    return values


def run():
    """Run the benchmark and print the elapsed times."""
    if np is None:
        print("NumPy not available; install 'tnfr[numpy]' to run this benchmark.")
        return

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

    print(f"optimized: {t_opt:.6f}s, naive: {t_naive:.6f}s")


if __name__ == "__main__":
    run()
