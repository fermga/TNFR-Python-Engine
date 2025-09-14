from collections import deque

from tnfr.glyph_history import _normalize_history_input, _ensure_history


def test_normalize_history_input_filters_and_lists():
    assert _normalize_history_input('abc') == []
    assert _normalize_history_input(['a', 'b']) == ['a', 'b']


def test_ensure_history_uses_normalized_list():
    nd: dict[str, object] = {'glyph_history': 'abc'}
    w, hist = _ensure_history(nd, 2)
    assert w == 2
    assert isinstance(hist, deque)
    assert list(hist) == []
