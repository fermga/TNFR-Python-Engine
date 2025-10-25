"""Regression tests for :mod:`tnfr.dynamics` run loop."""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import pytest

import tnfr.dynamics as dynamics
import tnfr.dynamics.adaptation as adaptation
import tnfr.dynamics.coordination as coordination
import tnfr.dynamics.integrators as integrators
import tnfr.dynamics.runtime as runtime
import tnfr.dynamics.selectors as selectors
from tnfr.alias import get_attr
from tnfr.constants import DEFAULTS
from tnfr.glyph_history import ensure_history


class _RecordingIntegrator(integrators.AbstractIntegrator):
    def __init__(self, recorded: dict[str, Any]):
        self._recorded = recorded

    def integrate(
        self,
        graph,
        *,
        dt=None,
        t=None,
        method=None,
        n_jobs=None,
    ) -> None:
        self._recorded["integrator"] = n_jobs


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
    monkeypatch.setattr(runtime, "step", fake_step)

    dynamics.run(G, steps=5)

    assert call_count == 2
    hist = ensure_history(G)
    series = hist.get("stable_frac")
    assert isinstance(series, deque)
    assert list(series)[-2:] == [0.95, 0.95]


def test_run_stops_early_with_seed_deque(monkeypatch, graph_canon):
    """A seeded deque should remain intact when STOP_EARLY halts the loop."""

    G = graph_canon()
    G.graph["STOP_EARLY"] = {"enabled": True, "window": 2, "fraction": 0.8}
    G.graph["HISTORY_MAXLEN"] = 5
    seeded_series = deque([0.4, 0.5], maxlen=5)
    G.graph["history"] = {"stable_frac": seeded_series}

    call_count = 0

    def fake_step(G, *, dt=None, use_Si=True, apply_glyphs=True, n_jobs=None):
        nonlocal call_count
        call_count += 1
        hist = ensure_history(G)
        series = hist["stable_frac"]
        assert isinstance(series, deque)
        series.append(0.95)

    monkeypatch.setattr(dynamics, "step", fake_step)
    monkeypatch.setattr(runtime, "step", fake_step)

    dynamics.run(G, steps=5)

    assert call_count == 2
    hist = ensure_history(G)
    series = hist["stable_frac"]
    assert series is seeded_series
    assert isinstance(series, deque)
    assert len(series) == 4
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
    monkeypatch.setattr(runtime, "_update_nodes", fake_update_nodes)
    monkeypatch.setattr(dynamics, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(runtime, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(dynamics, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(runtime, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_validators", lambda G: None)
    monkeypatch.setattr(runtime, "_run_validators", lambda G: None)

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

    class _ClassRecordingIntegrator(integrators.AbstractIntegrator):
        def integrate(
            self,
            graph,
            *,
            dt=None,
            t=None,
            method=None,
            n_jobs=None,
        ) -> None:
            recorded["integrator"] = n_jobs

    def fake_compute_delta_nfr(G, *, n_jobs=None):
        recorded["dnfr"] = n_jobs

    G.graph["compute_delta_nfr"] = fake_compute_delta_nfr

    def fake_compute_si(G, *, inplace=True, n_jobs=None):
        recorded["si"] = n_jobs

    def fake_coordinate_global_local_phase(G, *_args, n_jobs=None, **_kwargs):
        recorded["phase"] = n_jobs

    def fake_adapt_vf_by_coherence(G, *, n_jobs=None):
        recorded["vf"] = n_jobs

    monkeypatch.setattr(dynamics, "compute_Si", fake_compute_si)
    monkeypatch.setattr(runtime, "compute_Si", fake_compute_si)
    G.graph["integrator"] = _ClassRecordingIntegrator
    monkeypatch.setattr(
        coordination,
        "coordinate_global_local_phase",
        fake_coordinate_global_local_phase,
    )
    monkeypatch.setattr(adaptation, "adapt_vf_by_coherence", fake_adapt_vf_by_coherence)
    monkeypatch.setattr(selectors, "_apply_glyphs", lambda G, selector, hist: None)
    monkeypatch.setattr(
        dynamics, "apply_canonical_clamps", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runtime, "apply_canonical_clamps", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_node_sample", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_update_node_sample", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(runtime, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(dynamics, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(runtime, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_validators", lambda G: None)
    monkeypatch.setattr(runtime, "_run_validators", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_after_callbacks", lambda G, step_idx: None)
    monkeypatch.setattr(runtime, "_run_after_callbacks", lambda G, step_idx: None)

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

    def fake_coordinate_global_local_phase(G, *_args, n_jobs=None, **_kwargs):
        recorded["phase"] = n_jobs

    def fake_adapt_vf_by_coherence(G, *, n_jobs=None):
        recorded["vf"] = n_jobs

    monkeypatch.setattr(dynamics, "compute_Si", fake_compute_si)
    monkeypatch.setattr(runtime, "compute_Si", fake_compute_si)
    G.graph["integrator"] = lambda graph: _RecordingIntegrator(recorded)
    monkeypatch.setattr(
        coordination,
        "coordinate_global_local_phase",
        fake_coordinate_global_local_phase,
    )
    monkeypatch.setattr(adaptation, "adapt_vf_by_coherence", fake_adapt_vf_by_coherence)
    monkeypatch.setattr(selectors, "_apply_glyphs", lambda G, selector, hist: None)
    monkeypatch.setattr(
        dynamics, "apply_canonical_clamps", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runtime, "apply_canonical_clamps", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_node_sample", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_update_node_sample", lambda *args, **kwargs: None)
    monkeypatch.setattr(dynamics, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(runtime, "_update_epi_hist", lambda G: None)
    monkeypatch.setattr(dynamics, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(runtime, "_maybe_remesh", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_validators", lambda G: None)
    monkeypatch.setattr(runtime, "_run_validators", lambda G: None)
    monkeypatch.setattr(dynamics, "_run_after_callbacks", lambda G, step_idx: None)
    monkeypatch.setattr(runtime, "_run_after_callbacks", lambda G, step_idx: None)

    dynamics.step(G)

    assert recorded == {
        "dnfr": 2,
        "si": 3,
        "integrator": 4,
        "phase": 5,
        "vf": None,
    }


def test_update_nodes_clamps_out_of_range_values(monkeypatch, graph_canon):
    """Nodes are clamped to canonical EPI and νf bounds during updates."""

    G = graph_canon()
    node_id = 0
    epi_hi = float(DEFAULTS["EPI_MAX"]) + 0.5
    vf_lo = float(DEFAULTS["VF_MIN"]) - 0.25
    G.add_node(node_id, EPI=epi_hi, nu_f=vf_lo)

    monkeypatch.setattr(runtime, "_prepare_dnfr", lambda *args, **kwargs: None)
    monkeypatch.setattr(selectors, "_apply_selector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(selectors, "_apply_glyphs", lambda *_args, **_kwargs: None)

    class _NoOpIntegrator(integrators.AbstractIntegrator):
        def integrate(
            self,
            graph,
            *,
            dt=None,
            t=None,
            method=None,
            n_jobs=None,
        ) -> None:
            return None

    monkeypatch.setattr(
        runtime, "_resolve_integrator_instance", lambda _graph: _NoOpIntegrator()
    )
    monkeypatch.setattr(
        coordination,
        "coordinate_global_local_phase",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adaptation, "adapt_vf_by_coherence", lambda *_args, **_kwargs: None
    )

    runtime._update_nodes(
        G,
        dt=0.1,
        use_Si=False,
        apply_glyphs=False,
        step_idx=0,
        hist={},
    )

    node_data = G.nodes[node_id]
    clamped_epi = get_attr(node_data, runtime.ALIAS_EPI, 0.0)
    clamped_vf = get_attr(node_data, runtime.ALIAS_VF, 0.0)

    assert clamped_epi == pytest.approx(float(DEFAULTS["EPI_MAX"]))
    assert clamped_vf == pytest.approx(float(DEFAULTS["VF_MIN"]))


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
    monkeypatch.setattr(runtime, "step", fake_step)

    overrides = {"dnfr": "2", "vf_adapt_n_jobs": 3}

    dynamics.run(G, steps=2, n_jobs=overrides)

    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert seen[0] == {"DNFR": "2", "VF_ADAPT": 3}


def test_run_rejects_negative_steps(graph_canon):
    """Negative step counts should be rejected explicitly."""

    G = graph_canon()

    with pytest.raises(ValueError, match="must be non-negative"):
        runtime.run(G, steps=-1)


def test_run_negative_steps_preserve_stop_state_and_history(graph_canon):
    """Rejecting negative steps must leave STOP_EARLY state and history untouched."""

    G = graph_canon()
    stop_cfg_before = copy.deepcopy(G.graph.get("STOP_EARLY"))
    assert stop_cfg_before is not None
    # ``history`` should not be created or mutated when ``run`` aborts early.
    G.graph.pop("history", None)

    with pytest.raises(ValueError, match="must be non-negative"):
        runtime.run(G, steps=-1)

    assert G.graph.get("STOP_EARLY") == stop_cfg_before
    assert "history" not in G.graph
