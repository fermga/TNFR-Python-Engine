"""Unit tests validating the HistoryDict usage counter behavior."""

import pytest

from tnfr.glyph_history import HistoryDict


def test_counter_single_key():
    hist = HistoryDict()
    hist["a"] = 0
    for _ in range(100):
        _ = hist.get_increment("a")
    assert hist._counts["a"] == 100


def test_counter_many_keys():
    hist = HistoryDict()
    for i in range(10):
        hist[f"k{i}"] = i
    for i in range(1000):
        _ = hist.get_increment(f"k{i % 10}")
    assert set(hist._counts) == set(hist)


def test_get_increment_tracks_usage():
    hist = HistoryDict()
    hist["a"] = 1
    counts_before = dict(hist._counts)
    assert hist.get_increment("a") == 1
    assert hist._counts["a"] == counts_before.get("a", 0) + 1


def test_get_missing_key_no_usage():
    hist = HistoryDict()
    counts_before = dict(hist._counts)
    assert hist.get("missing") is None
    assert hist._counts == counts_before


def test_counts_after_deletions():
    hist = HistoryDict()
    for i in range(10):
        hist[f"k{i}"] = i
        _ = hist.get_increment(f"k{i}")
    for _ in range(5):
        hist.pop_least_used()
        assert set(hist._counts) == set(hist)


def test_pop_least_used_empty_message():
    hist = HistoryDict()
    with pytest.raises(KeyError, match="HistoryDict is empty; cannot pop least used"):
        hist.pop_least_used()
