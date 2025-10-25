import copy
import math
from typing import Any

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import integrators as integrators_mod
from tnfr.dynamics.integrators import update_epi_via_nodal_equation

ALIAS_EPI = get_aliases("EPI")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_D2EPI = get_aliases("D2EPI")


def _build_sample_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph.update(
        {
            "DT": 0.15,
            "DT_MIN": 0.05,
            "INTEGRATOR_METHOD": "euler",
            "_t": 0.0,
            "GAMMA": {"type": "harmonic", "beta": 0.42, "omega": 0.31, "phi": 0.17},
        }
    )
    for idx in range(6):
        G.add_node(
            idx,
            VF=1.0 + 0.05 * idx,
            DNFR=0.2 * math.cos(0.3 * idx),
            EPI=0.5 * idx,
            DEPI=0.1 * math.sin(0.2 * idx),
            THETA=0.4 * idx,
        )
    return G


def _snapshot(G: nx.DiGraph) -> dict[int, tuple[float, float, float]]:
    return {
        node: (
            get_attr(data, ALIAS_EPI, 0.0),
            get_attr(data, ALIAS_DEPI, 0.0),
            get_attr(data, ALIAS_D2EPI, 0.0),
        )
        for node, data in G.nodes(data=True)
    }


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_parallel_integrator_matches_serial(method: str) -> None:
    base = _build_sample_graph()
    serial = copy.deepcopy(base)
    parallel = copy.deepcopy(base)

    kwargs = {"dt": 0.15, "t": 0.0, "method": method}
    update_epi_via_nodal_equation(serial, **kwargs, n_jobs=None)
    update_epi_via_nodal_equation(parallel, **kwargs, n_jobs=3)

    assert parallel.graph["_t"] == pytest.approx(serial.graph["_t"])
    serial_snapshot = _snapshot(serial)
    parallel_snapshot = _snapshot(parallel)
    for node in serial_snapshot:
        assert parallel_snapshot[node] == pytest.approx(serial_snapshot[node])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_parallel_fallback_without_numpy(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _build_sample_graph()
    base.graph["INTEGRATOR_METHOD"] = method

    np_missing_calls = 0

    def fake_get_numpy() -> None:
        nonlocal np_missing_calls
        np_missing_calls += 1
        return None

    monkeypatch.setattr(integrators_mod, "get_numpy", fake_get_numpy)

    serial = copy.deepcopy(base)
    parallel = copy.deepcopy(base)

    kwargs = {"dt": base.graph["DT"], "t": base.graph["_t"], "method": method}
    update_epi_via_nodal_equation(serial, **kwargs, n_jobs=None)
    update_epi_via_nodal_equation(parallel, **kwargs, n_jobs=3)

    assert np_missing_calls > 0
    serial_snapshot = _snapshot(serial)
    parallel_snapshot = _snapshot(parallel)
    for node in serial_snapshot:
        assert parallel_snapshot[node] == pytest.approx(serial_snapshot[node])


def test_evaluate_gamma_map_parallel_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    G = _build_sample_graph()
    nodes = list(G.nodes)
    t_val = 0.375

    def fake_eval_gamma(graph: nx.DiGraph, node: int, t: float) -> float:
        return float(graph.nodes[node]["VF"]) + t

    monkeypatch.setattr(integrators_mod, "eval_gamma", fake_eval_gamma)
    monkeypatch.setattr(integrators_mod, "_PARALLEL_GRAPH", None, raising=False)

    class _ImmediateFuture:
        def __init__(self, value: list[tuple[int, float]]):
            self._value = value

        def result(self) -> list[tuple[int, float]]:
            return self._value

    class SyncExecutor:
        instances: list["SyncExecutor"] = []

        def __init__(
            self,
            max_workers: int | None = None,
            *,
            mp_context: Any | None = None,
            initializer: Any | None = None,
            initargs: tuple[Any, ...] = (),
        ) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.initializer = initializer
            self.initargs = initargs
            self.tasks: list[tuple[list[int], float]] = []
            if initializer is not None:
                initializer(*initargs)
            SyncExecutor.instances.append(self)

        def __enter__(self) -> "SyncExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, fn, task: tuple[list[int], float]) -> _ImmediateFuture:
            self.tasks.append(task)
            return _ImmediateFuture(fn(task))

    SyncExecutor.instances = []
    monkeypatch.setattr(integrators_mod, "ProcessPoolExecutor", SyncExecutor)

    baseline = integrators_mod._evaluate_gamma_map(G, nodes, t_val, n_jobs=None)

    for node in nodes:
        expected_value = float(G.nodes[node]["VF"]) + t_val
        assert baseline[node] == pytest.approx(expected_value)

    for jobs in (None, 1, len(nodes) + 5, str(len(nodes) + 3)):
        prev_instances = len(SyncExecutor.instances)
        result = integrators_mod._evaluate_gamma_map(G, nodes, t_val, n_jobs=jobs)
        assert result == baseline
        if jobs in (None, 1):
            assert len(SyncExecutor.instances) == prev_instances

    assert len(SyncExecutor.instances) == 2
    for instance in SyncExecutor.instances:
        assert instance.initargs == (G,)
        assert instance.max_workers == len(nodes)
        for chunk, t in instance.tasks:
            assert t == t_val

    for instance in SyncExecutor.instances:
        chunked_nodes = [node for chunk, _ in instance.tasks for node in chunk]
        assert chunked_nodes == nodes

    assert integrators_mod._PARALLEL_GRAPH is G


def test_evaluate_gamma_map_single_node_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    G = _build_sample_graph()
    node = next(iter(G.nodes))
    t_val = 1.125

    def fake_eval_gamma(graph: nx.DiGraph, target: int, t: float) -> float:
        return float(graph.nodes[target]["VF"]) - t

    monkeypatch.setattr(integrators_mod, "eval_gamma", fake_eval_gamma)

    calls = {"executor": 0}

    class TrackingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            calls["executor"] += 1

        def __enter__(self) -> "TrackingExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, fn, task):
            raise AssertionError("Executor should not be used for single node")

    monkeypatch.setattr(integrators_mod, "ProcessPoolExecutor", TrackingExecutor)

    expected = {node: float(G.nodes[node]["VF"]) - t_val}
    result = integrators_mod._evaluate_gamma_map(G, [node], t_val, n_jobs=999)

    assert result == expected
    assert calls["executor"] == 0
