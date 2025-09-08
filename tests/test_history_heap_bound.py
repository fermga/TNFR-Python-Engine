import heapq
import timeit

from tnfr.glyph_history import HistoryDict


def test_heap_size_stays_bounded_under_churn():
    hist = HistoryDict({f"k{i}": [] for i in range(10)}, compact_every=5)
    for i in range(1000):
        _ = hist.get_increment(f"k{i % 10}")
    assert len(hist._heap) <= len(hist._counts) + hist._compact_every


def test_heap_size_skewed_churn():
    hist = HistoryDict({f"k{i}": [] for i in range(10)}, compact_every=5)
    for i in range(10_000):
        key = "k0" if i % 2 == 0 else f"k{(i % 9) + 1}"
        _ = hist.get_increment(key)
    assert len(hist._heap) <= len(hist._counts) + hist._compact_every


def test_prune_heap_discards_stale_entries():
    hist = HistoryDict({f"k{i}": [] for i in range(3)}, compact_every=1)
    for i in range(30):
        hist.get_increment(f"k{i % 3}")
    # Inject stale entries
    hist._heap.extend([(0, "k0"), (0, "k1"), (0, "k2")])
    heapq.heapify(hist._heap)
    hist._prune_heap()
    assert len(hist._heap) <= len(hist._counts) + hist._compact_every
    for key, cnt in hist._counts.items():
        assert (cnt, key) in hist._heap


def test_prune_heap_performance():
    hist = HistoryDict({f"k{i}": [] for i in range(100)}, compact_every=5)
    def churn():
        for i in range(5_000):
            hist.get_increment(f"k{i % 100}")
    t = timeit.timeit(churn, number=1)
    assert t < 1.0
