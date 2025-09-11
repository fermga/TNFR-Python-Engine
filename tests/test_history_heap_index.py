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


def test_get_increment_updates_heap_index():
    hist = HistoryDict()
    hist.get_increment("a")
    assert hist._heap_index["a"] == 0
    assert hist._heap[0] == (1, "a")


def test_setdefault_tracks_heap_index():
    hist = HistoryDict()
    hist.setdefault("a", [])
    assert hist._heap_index["a"] == 0
    assert hist._heap[0] == (0, "a")


def test_prune_heap_keeps_index_consistent():
    hist = HistoryDict({f"k{i}": [] for i in range(3)}, compact_every=1)
    for _ in range(5):
        hist.get_increment("k0")
    assert len(hist._heap_index) == len(hist._counts)
    for k, idx in hist._heap_index.items():
        assert hist._heap[idx][1] == k


def test_prune_heap_prunes_in_place():
    hist = HistoryDict({f"k{i}": [] for i in range(3)}, compact_every=1)
    original_heap = id(hist._heap)
    for _ in range(5):
        hist.get_increment("k0")
    assert id(hist._heap) == original_heap
    assert len(hist._heap) <= len(hist._counts) + hist._compact_every
    for k, idx in hist._heap_index.items():
        assert hist._heap[idx][1] == k
