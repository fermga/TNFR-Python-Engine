"""Integration tests exercising runtime remesh and STOP_EARLY behaviour."""

from __future__ import annotations

from collections import deque
import random

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import get_aliases, inject_defaults
from tnfr.dynamics import integrators, runtime
from tnfr.dynamics.aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_VF
from tnfr.glyph_history import ensure_history
from tnfr.metrics import register_metrics_callbacks
from tnfr.structural import create_nfr


class _FixedStepIntegrator(integrators.AbstractIntegrator):
    """Integrator that advances time deterministically without altering EPI."""

    def __init__(self, dt: float):
        self._dt = float(dt)
        self.calls = 0
        self.last_dt = float(dt)

    def integrate(
        self,
        graph,
        *,
        dt=None,
        t=None,
        method=None,
        n_jobs=None,
    ) -> None:
        del t, method, n_jobs

        dt_step = self._dt if dt is None else float(dt)
        self.last_dt = dt_step
        self.calls += 1
        current_t = float(graph.graph.get("_t", 0.0))

        alias_depi = get_aliases("DEPI")
        for _, data in graph.nodes(data=True):
            set_attr(data, ALIAS_EPI, float(get_attr(data, ALIAS_EPI, 0.0)))
            set_attr(data, alias_depi, 0.0)
            set_attr(data, ALIAS_D2EPI, 0.0)

        graph.graph["_t"] = current_t + dt_step


def _stable_delta(graph, *, n_jobs=None):
    """ΔNFR hook that keeps nodal derivatives near zero for stability."""

    del n_jobs

    alias_depi = get_aliases("DEPI")
    for _, data in graph.nodes(data=True):
        set_attr(data, ALIAS_DNFR, 0.0)
        set_attr(data, alias_depi, 0.0)
        set_attr(data, ALIAS_SI, 0.98)
        set_attr(data, ALIAS_VF, float(get_attr(data, ALIAS_VF, 1.0)))


def test_runtime_run_triggers_remesh_and_stop_early():
    """Runtime should remesh once and honour STOP_EARLY gating with telemetry."""

    random.seed(0)
    dt = 0.05

    G = nx.Graph()
    inject_defaults(G)
    G.graph["DT"] = dt
    G.graph["HISTORY_MAXLEN"] = 0
    G.graph.setdefault("GLYPH_FACTORS", {}).pop("REMESH_alpha", None)
    metrics_cfg = G.graph.setdefault("METRICS", {})
    metrics_cfg.update({
        "enabled": True,
        "verbosity": "detailed",
        "attach_coherence_hooks": True,
        "attach_diagnosis_hooks": False,
    })

    nodes: list[str] = []
    for idx in range(4):
        _, node = create_nfr(
            f"n{idx}",
            epi=0.2 + 0.03 * idx,
            vf=1.0 + 0.05 * idx,
            theta=0.1 * idx,
            graph=G,
        )
        nodes.append(node)

    G.add_edges_from(zip(nodes, nodes[1:]))

    G.graph.update(
        {
            "STOP_EARLY": {"enabled": True, "window": 3, "fraction": 0.99},
            "REMESH_STABILITY_WINDOW": 3,
            "REMESH_REQUIRE_STABILITY": False,
            "REMESH_TAU_GLOBAL": 2,
            "REMESH_TAU_LOCAL": 1,
            "REMESH_COOLDOWN_WINDOW": 3,
            "REMESH_COOLDOWN_TS": 0.2,
            "REMESH_ALPHA": 0.65,
            "FRACTION_STABLE_REMESH": 0.9,
            "REMESH_LOG_EVENTS": True,
        }
    )

    integrator = _FixedStepIntegrator(dt)
    G.graph["integrator"] = integrator
    G.graph["compute_delta_nfr"] = _stable_delta

    hist_seed = ensure_history(G)
    hist_seed["stable_frac"] = [0.8, 0.98]
    stable_seed_len = len(hist_seed["stable_frac"])
    tau = max(G.graph["REMESH_TAU_GLOBAL"], G.graph["REMESH_TAU_LOCAL"])
    baseline = {node: float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)) for node in nodes}
    G.graph["_epi_hist"] = deque(
        [dict(baseline) for _ in range(tau + 1)],
        maxlen=max(2 * tau + 5, 64),
    )

    register_metrics_callbacks(G)

    runtime.run(G, steps=10, dt=dt)

    hist = ensure_history(G)
    events = hist.get("remesh_events", [])

    assert integrator.calls == 3, "STOP_EARLY should halt after the stability window"
    assert len(hist.get("C_steps", [])) == integrator.calls
    assert len(hist.get("stable_frac", [])) == stable_seed_len + integrator.calls
    assert hist["stable_frac"][-integrator.calls:] == pytest.approx([1.0] * integrator.calls)
    assert len(hist.get("phase_sync", [])) == integrator.calls
    assert len(hist.get("kuramoto_R", [])) == integrator.calls
    assert len(hist.get("glyph_load_disr", [])) == integrator.calls
    assert len(hist.get("sense_sigma_mag", [])) == integrator.calls
    assert len(hist.get("Si_mean", [])) == integrator.calls
    assert len(hist.get("Si_hi_frac", [])) == integrator.calls
    assert events, "remesh events must be recorded"

    event = events[-1]
    assert event["tau_global"] == G.graph["REMESH_TAU_GLOBAL"]
    assert event["tau_local"] == G.graph["REMESH_TAU_LOCAL"]
    assert event["alpha_source"] == "REMESH_ALPHA"
    assert event["stable_frac_last"] == hist["stable_frac"][-1]
    assert event.get("phase_sync_last") == pytest.approx(hist["phase_sync"][-2])
    assert event.get("glyph_disr_last") == pytest.approx(hist["glyph_load_disr"][-2])
    assert event["step"] == len(hist["C_steps"]) - 1

    remesh_step = G.graph["_last_remesh_step"]
    remesh_ts = G.graph["_last_remesh_ts"]
    assert remesh_step == stable_seed_len + integrator.calls - 1
    assert remesh_ts == pytest.approx(dt * integrator.calls)

    runtime.run(G, steps=5, dt=dt)

    hist = ensure_history(G)
    assert integrator.calls == 4, "A single extra step should run before STOP_EARLY triggers again"
    assert len(hist.get("remesh_events", [])) == 1, "Cooldown should block additional remeshes"
    assert len(hist.get("stable_frac", [])) == stable_seed_len + integrator.calls
    assert G.graph["_last_remesh_step"] == remesh_step
    assert G.graph["_last_remesh_ts"] == remesh_ts
    assert G.graph["_t"] == pytest.approx(dt * integrator.calls)
