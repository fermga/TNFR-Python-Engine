"""Benchmark for ensure_history trim performance."""

import time

from tnfr.glyph_history import HistoryDict, ensure_history
from tnfr.constants import inject_defaults
import networkx as nx


def run():
    """Run the benchmark and print the elapsed times."""
    G = nx.Graph()
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 1000
    hist = {f"k{i}": [] for i in range(2000)}
    G.graph["history"] = HistoryDict(hist, maxlen=2000)

    start = time.perf_counter()
    ensure_history(G)
    t_bulk = time.perf_counter() - start

    hist2 = HistoryDict({f"k{i}": [] for i in range(2000)}, maxlen=2000)
    start = time.perf_counter()
    while len(hist2) > 1000:
        hist2.pop_least_used()
    t_loop = time.perf_counter() - start

    print(f"bulk: {t_bulk:.6f}s, manual loop: {t_loop:.6f}s")


if __name__ == "__main__":
    run()
