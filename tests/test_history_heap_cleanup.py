"""Pruebas de compactación de _heap en HistoryDict."""
from tnfr.helpers import HistoryDict


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
