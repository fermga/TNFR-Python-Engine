from tnfr.glyph_history import HistoryDict


def test_setitem_initializes_count():
    hist = HistoryDict()
    hist["a"] = 1
    assert hist._counts["a"] == 0


def test_setitem_overwrite_preserves_count():
    hist = HistoryDict({"a": 1})
    hist.get_increment("a")
    hist["a"] = 2
    assert hist["a"] == 2
    assert hist._counts["a"] == 1


def test_eviction_updates_counts():
    hist = HistoryDict({"a": 1, "b": 2, "c": 3})
    hist.get_increment("a")
    hist.get_increment("b")
    hist.get_increment("b")
    val = hist.pop_least_used()
    assert val == 3
    assert "c" not in hist
    assert "c" not in hist._counts


def test_get_increment_updates_count():
    hist = HistoryDict()
    hist.get_increment("a")
    assert hist._counts["a"] == 1


def test_setdefault_initializes_count():
    hist = HistoryDict()
    hist.setdefault("a", [])
    assert hist._counts["a"] == 0

