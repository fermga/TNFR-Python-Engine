"""Benchmark for HistoryDict.increment performance."""

import time
from collections import Counter

from tnfr.glyph_history import HistoryDict


def run():
    """Run the benchmark and print the elapsed times."""
    n = 5000
    hist = HistoryDict()
    start = time.perf_counter()
    for i in range(n):
        hist.get_increment(f"k{i}")
    t_hist = time.perf_counter() - start

    counter = Counter()
    start = time.perf_counter()
    for i in range(n):
        counter[f"k{i}"] += 1
    t_counter = time.perf_counter() - start

    print(f"HistoryDict: {t_hist:.6f}s, Counter: {t_counter:.6f}s")


if __name__ == "__main__":
    run()
