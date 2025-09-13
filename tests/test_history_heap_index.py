from tnfr.glyph_history import HistoryDict


def test_setitem_creates_heap_entry():
    hist = HistoryDict()
    hist["a"] = 1
    assert (0, "a") in hist._heap


def test_setitem_updates_existing_heap_entry():
    hist = HistoryDict({"a": 1})
    hist.get_increment("a")
    before_len = len(hist._heap)
    hist["a"] = 2
    assert hist["a"] == 2
    assert len(hist._heap) == before_len
    assert (1, "a") in hist._heap


def test_eviction_updates_counts():
    hist = HistoryDict({"a": 1, "b": 2, "c": 3})
    hist.get_increment("a")
    hist.get_increment("b")
    hist.get_increment("b")
    val = hist.pop_least_used()
    assert val == 3
    assert "c" not in hist
    assert "c" not in hist._counts


def test_get_increment_pushes_heap_entry():
    hist = HistoryDict()
    hist.get_increment("a")
    assert (1, "a") in hist._heap


def test_setdefault_creates_heap_entry():
    hist = HistoryDict()
    hist.setdefault("a", [])
    assert (0, "a") in hist._heap


def test_prune_heap_keeps_heap_consistent():
    hist = HistoryDict({f"k{i}": [] for i in range(3)}, compact_every=1)
    for _ in range(5):
        hist.get_increment("k0")
    for key, cnt in hist._counts.items():
        assert (cnt, key) in hist._heap


def test_prune_heap_prunes_in_place():
    hist = HistoryDict({f"k{i}": [] for i in range(3)}, compact_every=1)
    original_heap = id(hist._heap)
    for _ in range(5):
        hist.get_increment("k0")
    assert id(hist._heap) == original_heap
    assert len(hist._heap) <= len(hist._counts) + hist._compact_every
    for key, cnt in hist._counts.items():
        assert (cnt, key) in hist._heap
