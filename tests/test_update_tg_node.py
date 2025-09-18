from collections import defaultdict
import builtins

from tnfr.metrics.glyph_timing import _update_tg_node, GlyphTiming
from tnfr.glyph_history import push_glyph


def test_update_tg_node_accumulates_and_resets():
    nd = {}
    push_glyph(nd, "A", window=5)
    tg_total = defaultdict(float)
    tg_by_node = defaultdict(lambda: defaultdict(list))
    g, latent = _update_tg_node(1, nd, 1.0, tg_total, tg_by_node)
    assert g == "A" and not latent
    assert tg_total == {}
    push_glyph(nd, "B", window=5)
    g, latent = _update_tg_node(1, nd, 2.0, tg_total, tg_by_node)
    assert g == "B"
    assert tg_total["A"] == 1.0
    assert tg_by_node[1]["A"] == [1.0]
    st = nd["_Tg"]
    assert isinstance(st, GlyphTiming)
    assert st.curr == "B" and st.run == 2.0


def test_update_tg_node_no_float_call(monkeypatch):
    nd = {}
    push_glyph(nd, "A", window=5)
    tg_total = defaultdict(float)
    tg_total["A"] = 0.0
    tg_by_node = defaultdict(lambda: defaultdict(list))
    _update_tg_node(1, nd, 1.0, tg_total, tg_by_node)
    push_glyph(nd, "B", window=5)

    def fake_float(x):  # pragma: no cover
        raise AssertionError("float() should not be called")

    monkeypatch.setattr(builtins, "float", fake_float)
    _update_tg_node(1, nd, 2.0, tg_total, tg_by_node)
