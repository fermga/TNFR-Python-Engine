import networkx as nx

from tnfr.dynamics import step, default_glyph_selector, parametric_glyph_selector


def _make_graph():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)
    G.graph["GRAMMAR_CANON"] = {"enabled": False}
    G.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, VF_MIN=0.0, VF_MAX=1.0)
    return G


def test_default_selector_does_not_compute_norms():
    G = _make_graph()
    G.graph["glyph_selector"] = default_glyph_selector
    step(G, use_Si=False, apply_glyphs=True)
    assert "_sel_norms" not in G.graph


def test_parametric_selector_computes_norms():
    G = _make_graph()
    G.graph["glyph_selector"] = parametric_glyph_selector
    step(G, use_Si=False, apply_glyphs=True)
    assert "_sel_norms" in G.graph
    assert "dnfr_max" in G.graph["_sel_norms"]
    assert "accel_max" in G.graph["_sel_norms"]
