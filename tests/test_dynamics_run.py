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
