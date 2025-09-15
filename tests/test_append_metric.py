from tnfr.glyph_history import append_metric, HistoryDict


def test_append_metric_plain_dict():
    hist: dict[str, list[int]] = {}
    append_metric(hist, "a", 1)
    append_metric(hist, "a", 2)
    assert hist["a"] == [1, 2]


def test_append_metric_historydict():
    hist = HistoryDict()
    append_metric(hist, "a", 1)
    append_metric(hist, "a", 2)
    assert list(hist["a"]) == [1, 2]
    assert hist._counts.get("a", 0) == 0
