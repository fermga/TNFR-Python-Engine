"""Integration tests for ``glyph_history`` window handling, ensuring series limits and usage counters stay consistent."""



from collections import deque

import pytest

import tnfr.glyph_history as glyph_history
from tnfr.constants import get_aliases, inject_defaults
from tnfr.glyph_history import HistoryDict, append_metric, ensure_history, _ensure_history

ALIAS_EPI_KIND = get_aliases("EPI_KIND")


def _make_node(history: list[str], current: str | None = None, window: int = 10) -> dict[str, object]:
    nd: dict[str, object] = {}
    for glyph in history:
        glyph_history.push_glyph(nd, glyph, window)
    if current is not None:
        nd[ALIAS_EPI_KIND[0]] = current
    return nd


# ---------------------------------------------------------------------------
# _ensure_history helpers
# ---------------------------------------------------------------------------

def test_ensure_history_skips_zero_window():
    nd: dict[str, object] = {}
    window, hist = _ensure_history(nd, 0)
    assert window == 0
    assert hist is None
    assert "glyph_history" not in nd


def test_ensure_history_creates_zero_when_requested():
    nd: dict[str, object] = {}
    window, hist = _ensure_history(nd, 0, create_zero=True)
    assert window == 0
    assert isinstance(hist, deque)
    assert hist.maxlen == 0
    assert "glyph_history" in nd


def test_ensure_history_positive_window_converts_input():
    nd: dict[str, object] = {}
    window, hist = _ensure_history(nd, 2)
    assert window == 2
    assert isinstance(hist, deque)
    assert hist.maxlen == 2
    hist.append("A")
    assert list(nd["glyph_history"]) == ["A"]


def test_ensure_history_discards_non_iterable_input():
    nd: dict[str, object] = {"glyph_history": "ABC"}
    _, hist = _ensure_history(nd, 2)
    assert isinstance(hist, deque)
    assert list(hist) == []


def test_ensure_history_accepts_iterable_input():
    nd: dict[str, object] = {"glyph_history": ["A", "B"]}
    _, hist = _ensure_history(nd, 2)
    assert list(hist) == ["A", "B"]


# ---------------------------------------------------------------------------
# push_glyph
# ---------------------------------------------------------------------------

def test_push_glyph_negative_window_raises():
    nd: dict[str, object] = {}
    with pytest.raises(ValueError):
        glyph_history.push_glyph(nd, "A", window=-1)


def test_push_glyph_zero_window_drops_entries():
    nd: dict[str, object] = {}
    glyph_history.push_glyph(nd, "A", window=0)
    assert list(nd["glyph_history"]) == []
    glyph_history.push_glyph(nd, "B", window=0)
    assert list(nd["glyph_history"]) == []


def test_push_glyph_positive_window_keeps_recent_items():
    nd: dict[str, object] = {}
    glyph_history.push_glyph(nd, "A", window=2)
    glyph_history.push_glyph(nd, "B", window=2)
    assert list(nd["glyph_history"]) == ["A", "B"]
    glyph_history.push_glyph(nd, "C", window=2)
    assert list(nd["glyph_history"]) == ["B", "C"]


def test_push_glyph_accepts_existing_list_history():
    nd: dict[str, object] = {"glyph_history": ["A"]}
    glyph_history.push_glyph(nd, "B", window=2)
    assert list(nd["glyph_history"]) == ["A", "B"]


# ---------------------------------------------------------------------------
# recent_glyph
# ---------------------------------------------------------------------------

def test_recent_glyph_window_one_prefers_history_over_current():
    nd = _make_node(["Y"], current="X")
    assert not glyph_history.recent_glyph(nd, "X", window=1)
    assert glyph_history.recent_glyph(nd, "Y", window=1)


def test_recent_glyph_window_zero_checks_current_only():
    nd = _make_node(["A", "B"], current="B")
    assert not glyph_history.recent_glyph(nd, "B", window=0)


def test_recent_glyph_window_zero_does_not_create_history():
    nd: dict[str, object] = {}
    assert not glyph_history.recent_glyph(nd, "B", window=0)
    assert "glyph_history" not in nd


def test_recent_glyph_window_negative_raises():
    nd = _make_node(["A", "B"], current="B")
    with pytest.raises(ValueError):
        glyph_history.recent_glyph(nd, "B", window=-1)


def test_recent_glyph_history_lookup_with_window():
    nd = _make_node(["A", "B"], current="C")
    assert glyph_history.recent_glyph(nd, "B", window=2)
    assert glyph_history.recent_glyph(nd, "A", window=2)
    assert glyph_history.recent_glyph(nd, "A", window=3)


def test_recent_glyph_discards_non_iterable_history():
    nd = {"glyph_history": 1}  # type: ignore[assignment]
    assert not glyph_history.recent_glyph(nd, "A", window=1)
    assert list(nd["glyph_history"]) == []


# ---------------------------------------------------------------------------
# append_metric
# ---------------------------------------------------------------------------

def test_append_metric_updates_plain_dict_series():
    hist: dict[str, list[int]] = {}
    append_metric(hist, "a", 1)
    append_metric(hist, "a", 2)
    assert hist["a"] == [1, 2]


def test_append_metric_respects_historydict_counters():
    hist = HistoryDict()
    append_metric(hist, "a", 1)
    append_metric(hist, "a", 2)
    assert list(hist["a"]) == [1, 2]
    assert hist._counts.get("a", 0) == 0


# ---------------------------------------------------------------------------
# ensure_history integration (window handling)
# ---------------------------------------------------------------------------

def test_history_maxlen_and_cleanup(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    append_metric(hist, "a", 1)
    append_metric(hist, "b", 2)
    append_metric(hist, "c", 3)

    ensure_history(G)  # force cleanup
    assert len(hist) == 2

    series = hist.setdefault("series", [])
    series.extend([1, 2, 3])
    assert isinstance(series, deque)
    assert series.maxlen == 2
    assert series == deque([2, 3], maxlen=2)


def test_history_least_used_is_preserved(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", [])  # zero length but still in use
    hist.setdefault("b", []).append(1)
    hist.setdefault("c", []).append(1)
    _ = hist.get_increment("a")
    _ = hist.get_increment("a")

    ensure_history(G)
    assert len(hist) == 2
    assert "a" in hist


def test_history_trim_uses_least_used_counter(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    for key in ["a", "b", "c", "d"]:
        append_metric(hist, key, 1)
    _ = hist.get_increment("a")
    _ = hist.get_increment("a")
    _ = hist.get_increment("b")

    ensure_history(G)
    assert set(hist.keys()) == {"a", "b"}


def test_history_maxlen_override_respected(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)

    hist = ensure_history(G)
    assert not isinstance(hist, HistoryDict)

    G.graph["HISTORY_MAXLEN"] = 3
    hist = ensure_history(G)
    assert isinstance(hist, HistoryDict)
    assert hist._maxlen == 3


def test_history_not_trimmed_when_equal_maxlen(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", []).append(1)
    hist.setdefault("b", []).append(2)

    ensure_history(G)
    assert len(hist) == 2
    assert set(hist.keys()) == {"a", "b"}


def test_history_not_trimmed_when_below_maxlen(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", []).append(1)

    ensure_history(G)
    assert len(hist) == 1
    assert "a" in hist


def test_history_negative_maxlen_raises(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["HISTORY_MAXLEN"] = -1

    with pytest.raises(ValueError):
        ensure_history(G)
