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


@pytest.mark.parametrize("bad_jobs", ["three", 1])
def test_coordinate_phase_invalid_jobs_stay_sequential(monkeypatch, graph_canon, bad_jobs):
    graph_factory = graph_canon

    monkeypatch.setattr(coordination, "get_numpy", lambda: None)

    baseline = _build_ring_graph(graph_factory, seed=11, size=4)
    baseline.graph["PHASE_ADAPT"] = {"enabled": False}
    baseline.graph["history"] = {}

    expected_thetas: dict[int, float] = {}
    coordination.coordinate_global_local_phase(baseline, n_jobs=None)
    for node in baseline.nodes:
        expected_thetas[node] = get_attr(baseline.nodes[node], ALIAS_THETA, 0.0)

    class _ExplodingExecutor:
        def __init__(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("ProcessPoolExecutor should not be constructed")

    monkeypatch.setattr(coordination, "ProcessPoolExecutor", _ExplodingExecutor)

    target = _build_ring_graph(graph_factory, seed=11, size=4)
    target.graph["PHASE_ADAPT"] = {"enabled": False}
    target.graph["history"] = {}

    coordination.coordinate_global_local_phase(target, n_jobs=bad_jobs)

    for node, expected in expected_thetas.items():
        assert get_attr(target.nodes[node], ALIAS_THETA, 0.0) == pytest.approx(expected)


def test_coordinate_phase_empty_graph_keeps_history(monkeypatch, graph_canon):
    monkeypatch.setattr(coordination, "get_numpy", lambda: None)

    graph = graph_canon()
    history = {
        "phase_state": deque(["stable"]),
        "phase_R": deque([0.25]),
        "phase_disr": deque([0.05]),
        "phase_kG": [0.2],
        "phase_kL": [0.1],
    }
    graph.graph["history"] = history
    graph.graph["PHASE_ADAPT"] = {"enabled": False}

    inert_snapshots = {
        key: list(value)
        for key, value in history.items()
        if key in {"phase_state", "phase_R", "phase_disr"}
    }
    k_lengths = {key: len(value) for key, value in history.items() if key.startswith("phase_k")}

    coordination.coordinate_global_local_phase(graph)

    assert graph.graph["history"] is history
    for key, snapshot in inert_snapshots.items():
        assert list(history[key]) == snapshot
    for key, length in k_lengths.items():
        assert len(history[key]) == length + 1


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


def test_coordinate_phase_handles_desynced_trig_cache(monkeypatch, graph_canon):
    graph = graph_canon()
    target = "beta"
    nodes = {"alpha": -0.4, target: 0.65, "gamma": 0.2}
    graph.add_nodes_from(nodes)
    for name, theta in nodes.items():
        set_attr(graph.nodes[name], ALIAS_THETA, theta)
    graph.add_edges_from([("alpha", target), (target, "gamma")])

    graph.graph["PHASE_K_GLOBAL"] = 0.05
    graph.graph["PHASE_K_LOCAL"] = 0.15
    graph.graph["PHASE_ADAPT"] = {
        "enabled": True,
        "kG_min": 0.05,
        "kG_max": 0.05,
        "kL_min": 0.15,
        "kL_max": 0.15,
        "up": 1.0,
        "down": 1.0,
    }
    history = {
        "phase_state": deque(["stable"]),
        "phase_R": deque([0.21]),
        "phase_disr": deque([0.05]),
        "phase_kG": [graph.graph["PHASE_K_GLOBAL"]],
        "phase_kL": [graph.graph["PHASE_K_LOCAL"]],
    }
    graph.graph["history"] = history

    fake_np = _FakeNumPy()
    monkeypatch.setattr(coordination, "get_numpy", lambda: fake_np)

    original_trig = coordination.get_trig_cache

    def _corrupted_trig_cache(G, *, np=None, cache_size=None):
        cache = original_trig(G, np=np, cache_size=cache_size)
        cache.theta.pop(target, None)
        cache.cos.pop(target, None)
        cache.sin.pop(target, None)
        return cache

    monkeypatch.setattr(coordination, "get_trig_cache", _corrupted_trig_cache)

    original_neighbors = coordination.ensure_neighbors_map

    def _corrupted_neighbors(G):
        mapping = dict(original_neighbors(G))
        mapping.pop(target, None)
        return mapping

    monkeypatch.setattr(coordination, "ensure_neighbors_map", _corrupted_neighbors)

    target_node_data = graph.nodes[target]
    fallback_calls: list[float] = []
    original_get_theta_attr = coordination.get_theta_attr

    def _tracking_theta_attr(data, default=None, **kwargs):
        value = original_get_theta_attr(data, default, **kwargs)
        if data is target_node_data:
            fallback_calls.append(float(value))
        return value

    monkeypatch.setattr(coordination, "get_theta_attr", _tracking_theta_attr)

    computed_state = ("transition", 0.72, 0.08)

    def _fake_compute_state(G, cfg):
        return computed_state

    monkeypatch.setattr(coordination, "_compute_state", _fake_compute_state)

    kG = graph.graph["PHASE_K_GLOBAL"]
    theta_values = list(nodes.values())
    mean_cos = sum(math.cos(angle) for angle in theta_values) / len(theta_values)
    mean_sin = sum(math.sin(angle) for angle in theta_values) / len(theta_values)
    expected_global = math.atan2(mean_sin, mean_cos)
    expected_target = nodes[target] + kG * (expected_global - nodes[target])

    state_len = len(history["phase_state"])
    R_len = len(history["phase_R"])
    disr_len = len(history["phase_disr"])
    kG_len = len(history["phase_kG"])
    kL_len = len(history["phase_kL"])

    coordination.coordinate_global_local_phase(graph)

    assert fallback_calls and fallback_calls[0] == pytest.approx(nodes[target])

    final_theta = get_attr(graph.nodes[target], ALIAS_THETA, 0.0)
    assert final_theta == pytest.approx(expected_target)

    assert len(history["phase_state"]) == state_len + 1
    assert history["phase_state"][-1] == computed_state[0]
    assert len(history["phase_R"]) == R_len + 1
    assert history["phase_R"][-1] == pytest.approx(computed_state[1])
    assert len(history["phase_disr"]) == disr_len + 1
    assert history["phase_disr"][-1] == pytest.approx(computed_state[2])
    assert len(history["phase_kG"]) == kG_len + 1
    assert history["phase_kG"][-1] == pytest.approx(kG)
    assert len(history["phase_kL"]) == kL_len + 1
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

