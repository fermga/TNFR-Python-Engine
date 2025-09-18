"""Regression tests for :mod:`tnfr.dynamics` run loop."""

from __future__ import annotations

from collections import deque

import tnfr.dynamics as dynamics
from tnfr.glyph_history import ensure_history


def test_run_stops_early_with_historydict(monkeypatch, graph_canon):
    """STOP_EARLY should break once the stability window stays above the limit."""

    G = graph_canon()
    G.graph["STOP_EARLY"] = {"enabled": True, "window": 2, "fraction": 0.8}
    G.graph["HISTORY_MAXLEN"] = 5
    # Pre-populate with values below the threshold so the loop needs fresh data.
    G.graph["history"] = {"stable_frac": [0.4, 0.5]}

    call_count = 0

    def fake_step(G, *, dt=None, use_Si=True, apply_glyphs=True):
        nonlocal call_count
        call_count += 1
        hist = ensure_history(G)
        series = hist.setdefault("stable_frac", [])
        series.append(0.95)

    monkeypatch.setattr(dynamics, "step", fake_step)

    dynamics.run(G, steps=5)

    assert call_count == 2
    hist = ensure_history(G)
    series = hist.get("stable_frac")
    assert isinstance(series, deque)
    assert list(series)[-2:] == [0.95, 0.95]


def test_step_preserves_since_mappings(monkeypatch, graph_canon):
    """``since_*`` history entries should stay as mappings when bounded."""

    G = graph_canon()
    G.add_node(0)
    G.graph["HISTORY_MAXLEN"] = 3

    def fake_update_nodes(
        G,
        *,
        dt,
        use_Si,
        apply_glyphs,
        step_idx,
        hist,
    ) -> None:
        h_al = hist.setdefault("since_AL", {})
        h_en = hist.setdefault("since_EN", {})
        h_al[0] = h_al.get(0, 0) + 1
        h_en[0] = h_en.get(0, 0) + 1

    monkeypatch.setattr(dynamics, "_update_nodes", fake_update_nodes)
    monkeypatch.setattr(dynamics, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(dynamics, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_validators", lambda G: None)

    dynamics.step(G)

    hist = ensure_history(G)
    since_al = hist["since_AL"]
    since_en = hist["since_EN"]
    assert isinstance(since_al, dict)
    assert isinstance(since_en, dict)
    assert since_al[0] == 1
    assert since_en[0] == 1
