from collections import defaultdict

from tnfr.metrics.core import _update_tg_node, TgCurr, TgRun
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
    assert st[TgCurr] == "B" and st[TgRun] == 2.0
