from __future__ import annotations

import tnfr.dynamics as dynamics
from tnfr.alias import set_attr
from tnfr.glyph_history import ensure_history


def _make_default_graph(graph_canon):
    G = graph_canon()
    G.graph["GRAMMAR_CANON"] = {"enabled": False}
    G.graph["SELECTOR_THRESHOLDS"] = {"si_hi": 0.8, "si_lo": 0.3, "dnfr_hi": 0.5}
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}

    samples = [
        (0.9, 0.2, 0.1),
        (0.2, 0.8, 0.0),
        (0.5, 0.4, 0.2),
    ]
    for idx, (si, dnfr, accel) in enumerate(samples):
        G.add_node(idx)
        nd = G.nodes[idx]
        set_attr(nd, dynamics.ALIAS_SI, si)
        set_attr(nd, dynamics.ALIAS_DNFR, dnfr)
        set_attr(nd, dynamics.ALIAS_D2EPI, accel)
    return G


def _make_param_graph(graph_canon):
    G = graph_canon()
    G.graph["GRAMMAR_CANON"] = {"enabled": False}
    G.graph["glyph_selector"] = dynamics.parametric_glyph_selector
    G.graph["SELECTOR_THRESHOLDS"] = {
        "si_hi": 0.75,
        "si_lo": 0.25,
        "dnfr_hi": 0.55,
        "dnfr_lo": 0.25,
        "accel_hi": 0.6,
        "accel_lo": 0.2,
    }
    G.graph["GLYPH_SELECTOR_MARGIN"] = 0.05
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}

    samples = [
        (0.85, 0.30, 0.10, ["RA"]),
        (0.20, 0.70, 0.80, ["NAV"]),
        (0.45, 0.65, 0.10, []),
        (0.50, 0.10, 0.70, []),
    ]
    for idx, (si, dnfr, accel, history) in enumerate(samples):
        G.add_node(idx)
        nd = G.nodes[idx]
        set_attr(nd, dynamics.ALIAS_SI, si)
        set_attr(nd, dynamics.ALIAS_DNFR, dnfr)
        set_attr(nd, dynamics.ALIAS_D2EPI, accel)
        if history:
            nd["glyph_history"] = history[:]
    return G


def _run_selector(G, monkeypatch):
    history = ensure_history(G)
    history["since_AL"] = {n: 0 for n in G.nodes}
    history["since_EN"] = {n: 0 for n in G.nodes}

    applied: list[tuple[int, str]] = []

    def fake_apply_glyph(G_local, node, glyph, *, window=None):
        value = getattr(glyph, "value", glyph)
        applied.append((node, value))

    monkeypatch.setattr(dynamics, "apply_glyph", fake_apply_glyph)
    monkeypatch.setattr(dynamics, "on_applied_glyph", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "enforce_canonical_grammar", lambda G_local, node, glyph: glyph)

    selector = dynamics._apply_selector(G)
    dynamics._apply_glyphs(G, selector, history)
    return applied, history


def test_default_selector_parallel_matches_sequential(monkeypatch, graph_canon):
    G_seq = _make_default_graph(graph_canon)
    applied_seq, hist_seq = _run_selector(G_seq, monkeypatch)

    G_par = _make_default_graph(graph_canon)
    G_par.graph["GLYPH_SELECTOR_N_JOBS"] = 2
    applied_par, hist_par = _run_selector(G_par, monkeypatch)

    assert applied_seq == applied_par
    assert hist_seq["since_AL"] == hist_par["since_AL"]
    assert hist_seq["since_EN"] == hist_par["since_EN"]


def test_param_selector_parallel_matches_sequential(monkeypatch, graph_canon):
    G_seq = _make_param_graph(graph_canon)
    applied_seq, hist_seq = _run_selector(G_seq, monkeypatch)

    G_par = _make_param_graph(graph_canon)
    G_par.graph["GLYPH_SELECTOR_N_JOBS"] = 3
    applied_par, hist_par = _run_selector(G_par, monkeypatch)

    assert applied_seq == applied_par
    assert hist_seq["since_AL"] == hist_par["since_AL"]
    assert hist_seq["since_EN"] == hist_par["since_EN"]


def test_selector_n_jobs_one_is_sequential(monkeypatch, graph_canon):
    G = _make_param_graph(graph_canon)
    G.graph["GLYPH_SELECTOR_N_JOBS"] = 1

    class FailExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ProcessPoolExecutor should not be used when n_jobs == 1")

    monkeypatch.setattr(dynamics, "ProcessPoolExecutor", FailExecutor)

    _run_selector(G, monkeypatch)
