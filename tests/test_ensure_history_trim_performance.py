import time
import pytest

from tnfr.glyph_history import HistoryDict, ensure_history
from tnfr.constants import inject_defaults


@pytest.mark.slow
def test_ensure_history_trim_performance(graph_canon):
    G = graph_canon()
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 1000
    G.graph["HISTORY_COMPACT_EVERY"] = 100
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

    assert t_bulk <= t_loop
