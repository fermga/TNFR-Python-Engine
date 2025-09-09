import time
from collections import Counter
import pytest

from tnfr.glyph_history import HistoryDict


@pytest.mark.slow
def test_increment_performance():
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

    assert t_hist <= t_counter * 10
