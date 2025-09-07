from tnfr.glyph_history import HistoryDict


def test_heap_size_stays_bounded_under_churn():
    hist = HistoryDict({f"k{i}": [] for i in range(10)}, compact_every=5)
    for i in range(1000):
        _ = hist.get_increment(f"k{i % 10}")
    assert len(hist._heap) <= len(hist._counts) + hist._compact_every
