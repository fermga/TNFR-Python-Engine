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

    def fake_step(G, *, dt=None, use_Si=True, apply_glyphs=True, n_jobs=None):
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
        job_overrides=None,
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


def test_step_respects_n_jobs_overrides(monkeypatch, graph_canon):
    """Explicit ``n_jobs`` overrides should reach every parallel component."""

    G = graph_canon()
    recorded = {}

    def fake_compute_delta_nfr(G, *, n_jobs=None):
        recorded["dnfr"] = n_jobs

    G.graph["compute_delta_nfr"] = fake_compute_delta_nfr

    def fake_compute_si(G, *, inplace=True, n_jobs=None):
        recorded["si"] = n_jobs

    def fake_update_epi_via_nodal_equation(G, *, dt=None, method=None, n_jobs=None):
        recorded["integrator"] = n_jobs

    def fake_coordinate_global_local_phase(G, *_args, n_jobs=None, **_kwargs):
        recorded["phase"] = n_jobs

    def fake_adapt_vf_by_coherence(G, *, n_jobs=None):
        recorded["vf"] = n_jobs

    monkeypatch.setattr(dynamics, "compute_Si", fake_compute_si)
    monkeypatch.setattr(
        dynamics, "update_epi_via_nodal_equation", fake_update_epi_via_nodal_equation
    )
    monkeypatch.setattr(
        dynamics, "coordinate_global_local_phase", fake_coordinate_global_local_phase
    )
    monkeypatch.setattr(dynamics, "adapt_vf_by_coherence", fake_adapt_vf_by_coherence)
    monkeypatch.setattr(dynamics, "_apply_glyphs", lambda G, selector, hist: None)
    monkeypatch.setattr(dynamics, "apply_canonical_clamps", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_node_sample", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(dynamics, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_validators", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_after_callbacks", lambda G, step_idx: None)

    overrides = {
        "dnfr": "3",
        "SI_N_JOBS": 4,
        "integrator": 5,
        "phase": "6",
        "vf_adapt": 7,
    }

    dynamics.step(G, n_jobs=overrides)

    assert recorded == {
        "dnfr": 3,
        "si": 4,
        "integrator": 5,
        "phase": 6,
        "vf": 7,
    }


def test_step_defaults_to_graph_jobs(monkeypatch, graph_canon):
    """Graph attributes remain the fallback when overrides are missing."""

    G = graph_canon()
    G.graph.update(
        {
            "DNFR_N_JOBS": "2",
            "SI_N_JOBS": 3,
            "INTEGRATOR_N_JOBS": "4",
            "PHASE_N_JOBS": 5,
            "VF_ADAPT_N_JOBS": 0,
        }
    )

    recorded = {}

    def fake_compute_delta_nfr(G, *, n_jobs=None):
        recorded["dnfr"] = n_jobs

    G.graph["compute_delta_nfr"] = fake_compute_delta_nfr

    def fake_compute_si(G, *, inplace=True, n_jobs=None):
        recorded["si"] = n_jobs

    def fake_update_epi_via_nodal_equation(G, *, dt=None, method=None, n_jobs=None):
        recorded["integrator"] = n_jobs

    def fake_coordinate_global_local_phase(G, *_args, n_jobs=None, **_kwargs):
        recorded["phase"] = n_jobs

    def fake_adapt_vf_by_coherence(G, *, n_jobs=None):
        recorded["vf"] = n_jobs

    monkeypatch.setattr(dynamics, "compute_Si", fake_compute_si)
    monkeypatch.setattr(
        dynamics, "update_epi_via_nodal_equation", fake_update_epi_via_nodal_equation
    )
    monkeypatch.setattr(
        dynamics, "coordinate_global_local_phase", fake_coordinate_global_local_phase
    )
    monkeypatch.setattr(dynamics, "adapt_vf_by_coherence", fake_adapt_vf_by_coherence)
    monkeypatch.setattr(dynamics, "_apply_glyphs", lambda G, selector, hist: None)
    monkeypatch.setattr(dynamics, "apply_canonical_clamps", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_node_sample", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(dynamics, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_validators", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_after_callbacks", lambda G, step_idx: None)

    dynamics.step(G)

    assert recorded == {
        "dnfr": 2,
        "si": 3,
        "integrator": 4,
        "phase": 5,
        "vf": None,
    }


def test_run_reuses_normalized_n_jobs(monkeypatch, graph_canon):
    """``run`` should normalize ``n_jobs`` once and reuse it across steps."""

    G = graph_canon()
    seen = []

    def fake_step(
        G,
        *,
        dt=None,
        use_Si=True,
        apply_glyphs=True,
        n_jobs=None,
    ) -> None:
        seen.append(n_jobs)

    monkeypatch.setattr(dynamics, "step", fake_step)

    overrides = {"dnfr": "2", "vf_adapt_n_jobs": 3}

    dynamics.run(G, steps=2, n_jobs=overrides)

    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert seen[0] == {"DNFR": "2", "VF_ADAPT": 3}
