import timeit

from tnfr.glyph_history import HistoryDict


def test_counts_stay_bounded_under_churn():
    hist = HistoryDict({f"k{i}": [] for i in range(10)})
    for i in range(1000):
        _ = hist.get_increment(f"k{i % 10}")
    assert len(hist._counts) == len(hist)


def test_pop_least_used_discards_minimum_count():
    hist = HistoryDict({f"k{i}": [] for i in range(3)})
    for i in range(30):
        hist.get_increment(f"k{i % 3}")
    hist.get_increment("k0")
    expected = min(hist._counts, key=hist._counts.get)
    hist.pop_least_used()
    assert expected not in hist
    assert expected not in hist._counts


def test_pop_least_used_performance():
    hist = HistoryDict({f"k{i}": [] for i in range(100)})
    for i in range(5_000):
        hist.get_increment(f"k{i % 100}")

    def churn():
        for _ in range(100):
            hist.pop_least_used()

    t = timeit.timeit(churn, number=1)
    assert t < 1.0

