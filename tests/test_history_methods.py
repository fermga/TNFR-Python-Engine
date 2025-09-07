from collections import deque

from tnfr.glyph_history import HistoryDict


def test_getitem_pure_access():
    hist = HistoryDict({"a": 1})
    val = hist["a"]
    assert val == 1
    assert hist._counts.get("a", 0) == 0


def test_get_increment_tracks_usage():
    hist = HistoryDict({"a": 1})
    val = hist.get_increment("a")
    assert val == 1
    assert hist._counts["a"] == 1
    assert hist.get("missing", 42) == 42
    assert "missing" not in hist
    assert "missing" not in hist._counts


def test_setdefault_inserts_and_converts_without_usage():
    hist = HistoryDict(maxlen=2)
    val = hist.setdefault("a", [1])
    assert isinstance(val, deque)
    assert list(val) == [1]
    assert hist._counts.get("a", 0) == 0
    val2 = hist.setdefault("a", [2])
    assert val2 is val
    assert hist._counts.get("a", 0) == 0


def test_pop_least_used_removes_least_frequent_key():
    hist = HistoryDict({"a": 1, "b": 2})
    hist.get_increment("a")
    hist.get_increment("b")
    hist.get_increment("b")
    val = hist.pop_least_used()
    assert val == 1
    assert "a" not in hist
    assert "a" not in hist._counts


def test_pop_least_used_batch_removes_k_keys():
    hist = HistoryDict()
    for i in range(5):
        hist[f"k{i}"] = i
        for _ in range(i):
            hist.get_increment(f"k{i}")
    hist.pop_least_used_batch(2)
    assert set(hist) == {"k2", "k3", "k4"}
    assert set(hist._counts) == {"k2", "k3", "k4"}
