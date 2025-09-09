from tnfr.glyph_history import HistoryDict


def test_setitem_inserts_tracks_heap_index():
    hist = HistoryDict()
    hist["a"] = 1
    assert hist._heap_index["a"] == 0
    assert hist._heap[0] == (0, "a")


def test_setitem_updates_existing_heap_entry():
    hist = HistoryDict({"a": 1})
    hist.get_increment("a")
    before_len = len(hist._heap)
    hist["a"] = 2
    assert hist["a"] == 2
    assert len(hist._heap) == before_len
    idx = hist._heap_index["a"]
    assert hist._heap[idx] == (1, "a")


def test_eviction_updates_heap_index():
    hist = HistoryDict({"a": 1, "b": 2, "c": 3})
    hist.get_increment("a")
    hist.get_increment("b")
    hist.get_increment("b")
    val = hist.pop_least_used()
    assert val == 3
    assert "c" not in hist._heap_index
    assert len(hist._heap_index) == len(hist._counts)

