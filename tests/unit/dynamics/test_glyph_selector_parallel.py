from __future__ import annotations

import tnfr.dynamics as dynamics
import tnfr.dynamics.selectors as selectors
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


def _run_selector(
    G,
    monkeypatch,
    *,
    since_al: dict[int, int] | None = None,
    since_en: dict[int, int] | None = None,
    apply_hook=None,
    enforce_hook=None,
    on_applied_hook=None,
):
    history = ensure_history(G)
    if since_al is None:
        history["since_AL"] = {n: 0 for n in G.nodes}
    else:
        history["since_AL"] = dict(since_al)
    if since_en is None:
        history["since_EN"] = {n: 0 for n in G.nodes}
    else:
        history["since_EN"] = dict(since_en)

    applied: list[tuple[int, str]] = []

    def fake_apply_glyph(G_local, node, glyph, *, window=None):
        value = getattr(glyph, "value", glyph)
        applied.append((node, value))
        if apply_hook is not None:
            apply_hook(G_local, node, value, window=window)

    monkeypatch.setattr(selectors, "apply_glyph", fake_apply_glyph)

    if on_applied_hook is None:
        monkeypatch.setattr(selectors, "on_applied_glyph", lambda *args, **kwargs: None)
    else:

        def wrapped_on_applied(G_local, node, glyph, *args, **kwargs):
            value = getattr(glyph, "value", glyph)
            on_applied_hook(G_local, node, value)

        monkeypatch.setattr(selectors, "on_applied_glyph", wrapped_on_applied)

    if enforce_hook is None:
        monkeypatch.setattr(
            selectors, "enforce_canonical_grammar", lambda G_local, node, glyph: glyph
        )
    else:

        def wrapped_enforce(G_local, node, glyph):
            value = getattr(glyph, "value", glyph)
            return enforce_hook(G_local, node, value)

        monkeypatch.setattr(selectors, "enforce_canonical_grammar", wrapped_enforce)

    selector = selectors._apply_selector(G)
    selectors._apply_glyphs(G, selector, history)
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
            raise AssertionError(
                "ProcessPoolExecutor should not be used when n_jobs == 1"
            )

    monkeypatch.setattr(selectors, "ProcessPoolExecutor", FailExecutor)

    _run_selector(G, monkeypatch)


def test_parallel_selector_is_deterministic(monkeypatch, graph_canon):
    G_first = _make_param_graph(graph_canon)
    G_first.graph["GLYPH_SELECTOR_N_JOBS"] = 3
    first_applied, first_hist = _run_selector(G_first, monkeypatch)

    G_second = _make_param_graph(graph_canon)
    G_second.graph["GLYPH_SELECTOR_N_JOBS"] = 3
    second_applied, second_hist = _run_selector(G_second, monkeypatch)

    assert first_applied == second_applied
    assert first_hist["since_AL"] == second_hist["since_AL"]
    assert first_hist["since_EN"] == second_hist["since_EN"]


def test_parallel_respects_since_counters(monkeypatch, graph_canon):
    G_seq = _make_default_graph(graph_canon)
    G_seq.graph["AL_MAX_LAG"] = 1
    G_seq.graph["EN_MAX_LAG"] = 1

    since_al = {0: 1, 1: 0, 2: 0}
    since_en = {0: 0, 1: 2, 2: 0}

    applied_seq, hist_seq = _run_selector(
        G_seq,
        monkeypatch,
        since_al=since_al,
        since_en=since_en,
    )

    G_par = _make_default_graph(graph_canon)
    G_par.graph["AL_MAX_LAG"] = 1
    G_par.graph["EN_MAX_LAG"] = 1
    G_par.graph["GLYPH_SELECTOR_N_JOBS"] = 2

    applied_par, hist_par = _run_selector(
        G_par,
        monkeypatch,
        since_al=since_al,
        since_en=since_en,
    )

    assert applied_seq == applied_par
    assert hist_seq["since_AL"] == hist_par["since_AL"]
    assert hist_seq["since_EN"] == hist_par["since_EN"]


def test_parallel_canonical_hooks_order(monkeypatch, graph_canon):
    def make_graph():
        G = _make_param_graph(graph_canon)
        G.graph["GRAMMAR_CANON"] = {"enabled": True}
        return G

    log_seq: list[tuple[str, int, str]] = []

    def enforce_hook_seq(G_local, node, glyph):
        log_seq.append(("enforce", node, glyph))
        if node % 2 == 0:
            return glyph
        return selectors.Glyph.NAV

    def on_applied_seq(G_local, node, glyph):
        log_seq.append(("on_applied", node, glyph))

    applied_seq, _ = _run_selector(
        make_graph(),
        monkeypatch,
        enforce_hook=enforce_hook_seq,
        on_applied_hook=on_applied_seq,
    )

    log_par: list[tuple[str, int, str]] = []

    def enforce_hook_par(G_local, node, glyph):
        log_par.append(("enforce", node, glyph))
        if node % 2 == 0:
            return glyph
        return selectors.Glyph.NAV

    def on_applied_par(G_local, node, glyph):
        log_par.append(("on_applied", node, glyph))

    G_parallel = make_graph()
    G_parallel.graph["GLYPH_SELECTOR_N_JOBS"] = 3
    applied_par, _ = _run_selector(
        G_parallel,
        monkeypatch,
        enforce_hook=enforce_hook_par,
        on_applied_hook=on_applied_par,
    )

    assert applied_seq == applied_par
    assert log_seq == log_par
