import math
import random

import pytest

import tnfr.dynamics as dynamics
import tnfr.dynamics.adaptation as adaptation
import tnfr.dynamics.coordination as coordination
import tnfr.dynamics.integrators as integrators
import tnfr.dynamics.runtime as runtime
import tnfr.dynamics.selectors as selectors
from tnfr.alias import get_attr, set_attr
from tnfr.constants import get_aliases

ALIAS_THETA = get_aliases("THETA")


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


def _build_ring_graph(graph_factory, *, seed: int = 0, size: int = 8):
    rng = random.Random(seed)
    G = graph_factory()
    nodes = list(range(size))
    G.add_nodes_from(nodes)
    for idx in nodes:
        set_attr(G.nodes[idx], ALIAS_THETA, rng.uniform(-math.pi, math.pi))
    for idx in nodes:
        G.add_edge(idx, (idx + 1) % size)
    return G


def test_update_nodes_forwards_phase_jobs(monkeypatch, graph_canon):
    G = graph_canon()
    G.add_node(0)
    set_attr(G.nodes[0], ALIAS_THETA, 0.0)
    G.graph["PHASE_N_JOBS"] = 3

    captured = {}

    def fake_coordinate(G_inner, global_force, local_force, *, n_jobs=None):
        captured["n_jobs"] = n_jobs

    monkeypatch.setattr(
        coordination, "coordinate_global_local_phase", fake_coordinate
    )
    monkeypatch.setattr(dynamics, "_update_node_sample", lambda *a, **k: None)
    monkeypatch.setattr(dynamics, "_prepare_dnfr", lambda *a, **k: None)
    monkeypatch.setattr(selectors, "_apply_selector", lambda *a, **k: None)
    G.graph["integrator"] = _NoOpIntegrator()
    monkeypatch.setattr(adaptation, "adapt_vf_by_coherence", lambda *a, **k: None)
    monkeypatch.setattr(dynamics, "apply_canonical_clamps", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "apply_canonical_clamps", lambda *a, **k: None)

    dynamics._update_nodes(
        G,
        dt=0.1,
        use_Si=False,
        apply_glyphs=False,
        step_idx=1,
        hist={},
    )

    assert captured["n_jobs"] == 3


@pytest.mark.parametrize("phase_jobs", [None, 3])
def test_coordinate_phase_parallel_matches_serial(
    monkeypatch, graph_canon, phase_jobs
):
    graph_factory = graph_canon

    monkeypatch.setattr("tnfr.dynamics.get_numpy", lambda: None)
    monkeypatch.setattr("tnfr.metrics.trig_cache.get_numpy", lambda: None)
    monkeypatch.setattr("tnfr.metrics.trig.get_numpy", lambda: None)

    baseline = _build_ring_graph(graph_factory, seed=42, size=10)
    parallel = _build_ring_graph(graph_factory, seed=42, size=10)

    coordination.coordinate_global_local_phase(baseline, n_jobs=None)
    coordination.coordinate_global_local_phase(parallel, n_jobs=phase_jobs)

    for node in baseline.nodes:
        th_serial = get_attr(baseline.nodes[node], ALIAS_THETA, 0.0)
        th_parallel = get_attr(parallel.nodes[node], ALIAS_THETA, 0.0)
        assert th_parallel == pytest.approx(th_serial)

