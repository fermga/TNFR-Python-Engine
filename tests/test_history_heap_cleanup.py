"""Pruebas de compactación de _heap en HistoryDict."""

from tnfr.glyph_history import HistoryDict


def test_heap_compaction_single_key():
    hist = HistoryDict()
    hist["a"] = 0
    for _ in range(100):
        _ = hist["a"]
    assert len(hist._heap) <= len(hist) * 2


def test_heap_compaction_many_keys():
    hist = HistoryDict()
    for i in range(10):
        hist[f"k{i}"] = i
    for i in range(1000):
        _ = hist[f"k{i % 10}"]
    assert len(hist._heap) <= len(hist) * 2


def test_get_does_not_track_usage():
    hist = HistoryDict()
    hist["a"] = 1
    counts_before = dict(hist._counts)
    heap_before = list(hist._heap)
    assert hist.get("a") == 1
    assert hist._counts == counts_before
    assert hist._heap == heap_before


def test_heap_compaction_after_deletions():
    hist = HistoryDict()
    for i in range(10):
        hist[f"k{i}"] = i
        _ = hist[f"k{i}"]
    for _ in range(5):
        hist.pop_least_used()
        assert len(hist._heap) <= len(hist) * 2
