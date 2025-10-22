from __future__ import annotations

from typing import Sequence

from tnfr.alias import set_attr
from tnfr.dynamics import selectors
from tnfr.glyph_history import ensure_history


def _configure_basic_graph(graph_canon) -> tuple:
    G = graph_canon()
    G.graph["GRAMMAR_CANON"] = {"enabled": False}
    G.graph["AL_MAX_LAG"] = 10
    G.graph["EN_MAX_LAG"] = 10
    return G, (selectors.ALIAS_SI, selectors.ALIAS_DNFR, selectors.ALIAS_D2EPI)


def _inject_nodes(G, alias_triplet: Sequence[Sequence[str]], samples: Sequence[tuple[float, float, float]]) -> None:
    si_alias, dnfr_alias, accel_alias = alias_triplet
    for idx, (si, dnfr, accel) in enumerate(samples):
        G.add_node(idx)
        nd = G.nodes[idx]
        set_attr(nd, si_alias, si)
        set_attr(nd, dnfr_alias, dnfr)
        set_attr(nd, accel_alias, accel)


def test_default_selector_prepare_uses_preselection(graph_canon):
    G, aliases = _configure_basic_graph(graph_canon)
    G.graph["SELECTOR_THRESHOLDS"] = {"si_hi": 0.7, "si_lo": 0.3, "dnfr_hi": 0.5}
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}
    _inject_nodes(G, aliases, ((0.85, 0.10, 0.0), (0.10, 0.80, 0.0)))

    selector = selectors.DefaultGlyphSelector()
    selector.prepare(G, list(G.nodes))

    assert selector.select(G, 0) == "IL"
    assert selector.select(G, 1) == "OZ"


def test_apply_selector_supports_custom_instance(monkeypatch, graph_canon):
    class CustomSelector(selectors.AbstractSelector):
        def __init__(self) -> None:
            self.prepared_nodes: list[int] = []
            self.calls: list[int] = []

        def prepare(self, graph, nodes) -> None:
            self.prepared_nodes = list(nodes)

        def select(self, graph, node) -> str:
            self.calls.append(node)
            return "RA"

    custom = CustomSelector()
    G, aliases = _configure_basic_graph(graph_canon)
    _inject_nodes(G, aliases, ((0.5, 0.2, 0.0), (0.4, 0.3, 0.0)))

    history = ensure_history(G)
    history["since_AL"] = {n: 0 for n in G.nodes}
    history["since_EN"] = {n: 0 for n in G.nodes}

    applied: list[tuple[int, str]] = []

    def fake_apply_glyph(graph, node, glyph, *, window=None):
        applied.append((node, getattr(glyph, "value", glyph)))

    monkeypatch.setattr(selectors, "apply_glyph", fake_apply_glyph)
    monkeypatch.setattr(selectors, "on_applied_glyph", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        selectors, "enforce_canonical_grammar", lambda graph, node, glyph: glyph
    )

    G.graph["glyph_selector"] = custom
    selector = selectors._apply_selector(G)
    selectors._apply_glyphs(G, selector, history)

    assert custom.prepared_nodes == [0, 1]
    assert custom.calls == [0, 1]
    assert applied == [(0, "RA"), (1, "RA")]


def test_apply_selector_instantiates_selector_class(graph_canon):
    G, aliases = _configure_basic_graph(graph_canon)
    _inject_nodes(G, aliases, ((0.5, 0.2, 0.0),))

    G.graph["glyph_selector"] = selectors.ParametricGlyphSelector
    selector = selectors._apply_selector(G)

    assert isinstance(selector, selectors.ParametricGlyphSelector)
    assert isinstance(G.graph["glyph_selector"], selectors.ParametricGlyphSelector)
