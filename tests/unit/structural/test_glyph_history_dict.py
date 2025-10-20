"""Centralized tests for ``HistoryDict``.

Glyph histories revolve around three concepts:
* **Series**: sequences of events stored per key (lists or ``deque``).
* **Counters**: the ``_counts`` map measuring how often each series is used.
* **Windows**: size limits (``maxlen``) validated in ``test_glyph_history_windowing.py``.

This module focuses on the interaction between series and counters to ensure each responsibility remains isolated before window policies are applied."""



from collections import deque
import timeit

from tnfr.glyph_history import HistoryDict


def test_series_assignment_initializes_and_preserves_counters():
    hist = HistoryDict()
    hist["a"] = 1
    assert hist._counts["a"] == 0

    hist.get_increment("a")
    hist["a"] = 2
    assert hist["a"] == 2
    assert hist._counts["a"] == 1


def test_series_getitem_is_pure_access():
    hist = HistoryDict({"a": 1})
    assert hist["a"] == 1
    assert hist._counts.get("a", 0) == 0


def test_series_setdefault_converts_iterables_without_touching_counters():
    hist = HistoryDict(maxlen=2)
    val = hist.setdefault("a", [1])
    assert isinstance(val, deque)
    assert val == deque([1])
    assert hist._counts.get("a", 0) == 0

    existing = deque([3])
    val2 = hist.setdefault("b", existing)
    assert val2 is existing
    assert hist._counts.get("b", 0) == 0


def test_counters_increment_and_ignore_missing_entries():
    hist = HistoryDict({"a": 1})
    assert hist.get_increment("a") == 1
    assert hist._counts["a"] == 1

    assert hist.get("missing", 42) == 42
    assert "missing" not in hist
    assert "missing" not in hist._counts


def test_counters_remain_in_sync_with_series_after_eviction():
    hist = HistoryDict({"a": 1, "b": 2, "c": 3})
    hist.get_increment("a")
    hist.get_increment("b")
    hist.get_increment("b")

    removed = hist.pop_least_used()
    assert removed == 3
    assert "c" not in hist
    assert "c" not in hist._counts


def test_eviction_prefers_least_used_key():
    hist = HistoryDict({"a": 1, "b": 2, "c": 3})
    for _ in range(3):
        hist.get_increment("a")
    for _ in range(2):
        hist.get_increment("b")

    expected = min(hist._counts, key=hist._counts.get)
    hist.pop_least_used()
    assert expected not in hist
    assert expected not in hist._counts


def test_eviction_batch_discards_multiple_series():
    hist = HistoryDict()
    for i in range(5):
        hist[f"k{i}"] = i
        for _ in range(i):
            hist.get_increment(f"k{i}")

    hist.pop_least_used_batch(2)
    assert set(hist) == {"k2", "k3", "k4"}
    assert set(hist._counts) == {"k2", "k3", "k4"}


def test_counters_stay_bounded_under_churn():
    hist = HistoryDict({f"k{i}": [] for i in range(10)})
    for i in range(1000):
        _ = hist.get_increment(f"k{i % 10}")
    assert len(hist._counts) == len(hist)


def test_eviction_performance_remains_linearish():
    hist = HistoryDict({f"k{i}": [] for i in range(100)})
    for i in range(5_000):
        hist.get_increment(f"k{i % 100}")

    def churn() -> None:
        for _ in range(100):
            hist.pop_least_used()

    duration = timeit.timeit(churn, number=1)
    assert duration < 1.0
