"""Pruebas de compactación de _heap en HistoryDict."""

from tnfr.glyph_history import HistoryDict
import pytest


def test_heap_compaction_single_key():
    hist = HistoryDict()
    hist["a"] = 0
    for _ in range(100):
        _ = hist.get_increment("a")
    assert len(hist._heap) <= len(hist) + hist._compact_every


def test_heap_compaction_many_keys():
    hist = HistoryDict()
    for i in range(10):
        hist[f"k{i}"] = i
    for i in range(1000):
        _ = hist.get_increment(f"k{i % 10}")
    assert len(hist._heap) <= len(hist) + hist._compact_every


def test_get_increment_tracks_usage():
    hist = HistoryDict()
    hist["a"] = 1
    counts_before = dict(hist._counts)
    heap_before = list(hist._heap)
    assert hist.get_increment("a") == 1
    assert hist._counts["a"] == counts_before["a"] + 1
    assert hist._heap != heap_before


def test_get_missing_key_no_usage():
    hist = HistoryDict()
    counts_before = dict(hist._counts)
    heap_before = list(hist._heap)
    assert hist.get("missing") is None
    assert hist._counts == counts_before
    assert hist._heap == heap_before


def test_heap_compaction_after_deletions():
    hist = HistoryDict()
    for i in range(10):
        hist[f"k{i}"] = i
        _ = hist.get_increment(f"k{i}")
    for _ in range(5):
        hist.pop_least_used()
        assert len(hist._heap) <= len(hist) + hist._compact_every


def test_pop_least_used_empty_message():
    hist = HistoryDict()
    with pytest.raises(
        KeyError, match="HistoryDict is empty; cannot pop least used"
    ):
        hist.pop_least_used()
