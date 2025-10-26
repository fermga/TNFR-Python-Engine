import math
import random
from collections import deque

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

    monkeypatch.setattr(coordination, "coordinate_global_local_phase", fake_coordinate)
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
def test_coordinate_phase_parallel_matches_serial(monkeypatch, graph_canon, phase_jobs):
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


class _FakeArray(list):
    def __init__(self, iterable=()):
        super().__init__(float(item) for item in iterable)

    @property
    def size(self) -> int:
        return len(self)

    def _binary_op(self, other, op):
        if isinstance(other, _FakeArray):
            return _FakeArray(op(a, b) for a, b in zip(self, other))
        value = float(other)
        return _FakeArray(op(a, value) for a in self)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        if isinstance(other, _FakeArray):
            return other.__sub__(self)
        value = float(other)
        return _FakeArray(value - a for a in self)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)


class _FakeNumPy:
    def fromiter(self, iterable, dtype=float):
        return _FakeArray(iterable)

    def mean(self, array):
        seq = list(array)
        return float(sum(seq) / len(seq)) if seq else 0.0

    def arctan2(self, y, x):
        return math.atan2(y, x)

    def cos(self, array):
        if isinstance(array, _FakeArray):
            return _FakeArray(math.cos(item) for item in array)
        return math.cos(array)

    def sin(self, array):
        if isinstance(array, _FakeArray):
            return _FakeArray(math.sin(item) for item in array)
        return math.sin(array)


@pytest.mark.parametrize("numpy_present", [False, True])
def test_coordinate_phase_overrides_and_adaptive_history(
    monkeypatch, graph_canon, numpy_present
):
    def _fake_get_numpy():
        return _FakeNumPy() if numpy_present else None

    monkeypatch.setattr("tnfr.dynamics.get_numpy", _fake_get_numpy)
    monkeypatch.setattr(coordination, "get_numpy", _fake_get_numpy)

    graph = _build_ring_graph(graph_canon, seed=7, size=6)
    history = {
        "phase_state": ["Stable", "disSONANT"],
        "phase_R": [0.21, 0.33],
        "phase_disr": [0.12, 0.18],
    }
    graph.graph["history"] = history
    graph.graph["PHASE_K_GLOBAL"] = 0.05
    graph.graph["PHASE_K_LOCAL"] = 0.15

    initial_state = list(history["phase_state"])
    initial_R = list(history["phase_R"])
    initial_disr = list(history["phase_disr"])
    initial_kG_len = len(history.get("phase_kG", []))
    initial_kL_len = len(history.get("phase_kL", []))

    compute_calls: list[tuple[str, float, float]] = []

    def _fake_compute_state(G, cfg):
        compute_calls.append(("transition", 0.76, 0.22))
        return "transition", 0.76, 0.22

    monkeypatch.setattr(coordination, "_compute_state", _fake_compute_state)

    coordination.coordinate_global_local_phase(
        graph, global_force=0.42, local_force=0.27
    )

    assert compute_calls == []

    hist_state = history["phase_state"]
    assert isinstance(hist_state, deque)
    assert list(hist_state) == ["stable", "dissonant"]
    assert len(hist_state) == len(initial_state)

    assert isinstance(history["phase_R"], deque)
    assert list(history["phase_R"]) == initial_R
    assert isinstance(history["phase_disr"], deque)
    assert list(history["phase_disr"]) == initial_disr

    assert graph.graph["PHASE_K_GLOBAL"] == pytest.approx(0.42)
    assert graph.graph["PHASE_K_LOCAL"] == pytest.approx(0.27)

    assert len(history["phase_kG"]) == initial_kG_len + 1
    assert history["phase_kG"][initial_kG_len] == pytest.approx(0.42)
    assert len(history["phase_kL"]) == initial_kL_len + 1
    assert history["phase_kL"][initial_kL_len] == pytest.approx(0.27)

    state_len_before = len(hist_state)
    R_len_before = len(history["phase_R"])
    disr_len_before = len(history["phase_disr"])

    coordination.coordinate_global_local_phase(graph)

    assert len(compute_calls) == 1
    assert len(hist_state) == state_len_before + 1
    assert hist_state[-1] == "transition"
    assert len(history["phase_R"]) == R_len_before + 1
    assert history["phase_R"][-1] == pytest.approx(0.76)
    assert len(history["phase_disr"]) == disr_len_before + 1
    assert history["phase_disr"][-1] == pytest.approx(0.22)

    assert len(history["phase_kG"]) == initial_kG_len + 2
    assert history["phase_kG"][-1] == pytest.approx(graph.graph["PHASE_K_GLOBAL"])
    assert len(history["phase_kL"]) == initial_kL_len + 2
    assert history["phase_kL"][-1] == pytest.approx(graph.graph["PHASE_K_LOCAL"])


@pytest.mark.parametrize(
    "state,kG_start,kL_start,expected_kG,expected_kL",
    [
        ("dissonant", 0.11, 0.05, 0.37, 0.14),
        ("stable", 0.37, 0.23, 0.11, 0.05),
        ("Stable", 0.37, 0.23, 0.11, 0.05),
    ],
)
def test_smooth_adjust_k_tracks_state_targets(
    state, kG_start, kL_start, expected_kG, expected_kL
):
    cfg = {
        "kG_min": 0.11,
        "kG_max": 0.37,
        "kL_min": 0.05,
        "kL_max": 0.23,
        "up": 1.0,
        "down": 1.5,
    }

    midpoint = 0.5 * (cfg["kL_min"] + cfg["kL_max"])

    kG, kL = coordination._smooth_adjust_k(kG_start, kL_start, state, cfg)

    assert cfg["kG_min"] <= kG <= cfg["kG_max"]
    assert cfg["kL_min"] <= kL <= cfg["kL_max"]

    if state == "dissonant":
        assert kG == pytest.approx(expected_kG)
        assert kG >= kG_start
        assert kL == pytest.approx(midpoint)
        assert kL >= kL_start
    else:
        assert kG_start > expected_kG
        assert kL_start > expected_kL
        assert kG == pytest.approx(expected_kG)
        assert kL == pytest.approx(expected_kL)
        assert kG <= kG_start
        assert kL <= kL_start

