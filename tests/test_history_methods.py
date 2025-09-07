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
