"""Unit tests for integrator routines driving node evolution."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants import inject_defaults
from tnfr.dynamics import integrators as integrators_mod
from tnfr.dynamics import runtime as runtime_mod
from tnfr.dynamics.integrators import _apply_increment_chunk
from tnfr.dynamics import update_epi_via_nodal_equation, validate_canon
from tnfr.initialization import init_node_attrs


@pytest.mark.parametrize(
    "hint,total",
    [
        (None, 10),
        ("invalid", 10),
        (0, 10),
        (-3, 10),
        (1, 10),
        (4, 1),
    ],
)
def test_normalise_jobs_rejects_invalid_and_degenerate_inputs(hint, total):
    """_normalise_jobs should fall back to serial execution when hints are unusable."""

    assert integrators_mod._normalise_jobs(hint, total) is None


def test_normalise_jobs_clamps_to_available_nodes():
    """Large worker hints must be capped by the available node count."""

    assert integrators_mod._normalise_jobs(32, 5) == 5
    assert integrators_mod._normalise_jobs(2.9, 8) == 2


@pytest.mark.parametrize(
    "nodes,chunk_size,expected",
    [
        ([], 3, []),
        (["a"], 5, [["a"]]),
        (list(range(5)), 10, [list(range(5))]),
        (list(range(5)), 2, [[0, 1], [2, 3], [4]]),
        (list(range(6)), 4, [[0, 1, 2, 3], [4, 5]]),
        (list(range(4)), 1, [[0], [1], [2], [3]]),
    ],
)
def test_chunk_nodes_preserves_order(nodes, chunk_size, expected):
    """Chunking helper should yield deterministic slices covering the full sequence."""

    chunks = list(integrators_mod._chunk_nodes(nodes, chunk_size))
    assert chunks == expected


@pytest.mark.parametrize(
    "method, ks",
    [
        ("euler", (0.2,)),
        ("rk4", (0.1, 0.2, 0.3, 0.4)),
    ],
)
def test_apply_increment_chunk_zero_dt_preserves_second_derivative(method, ks):
    """Scalar fallback should clamp the second derivative when the step is null."""

    chunk = [(0, 1.0, 0.4, ks)]

    results = _apply_increment_chunk(chunk, dt_step=0.0, method=method)

    assert len(results) == 1
    node, (epi, dEPI_dt, d2EPI) = results[0]
    assert node == 0
    assert epi == pytest.approx(1.0)
    assert dEPI_dt == pytest.approx(ks[-1])
    assert d2EPI == pytest.approx(0.0)


def test_prepare_integration_params_validations_and_dt_min():
    G = nx.path_graph(3)
    inject_defaults(G)

    with pytest.raises(TypeError):
        integrators_mod.prepare_integration_params(G, dt="bad")

    with pytest.raises(ValueError):
        integrators_mod.prepare_integration_params(G, dt=-0.1)

    G.graph["DT_MIN"] = 0.2
    dt_step, steps, _, _ = integrators_mod.prepare_integration_params(G, dt=0.5)
    assert steps > 1
    assert dt_step == pytest.approx(0.25)

    with pytest.raises(ValueError):
        integrators_mod.prepare_integration_params(G, method="heun")


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_epi_limits_preserved(method):
    G = nx.cycle_graph(6)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    G.graph["INTEGRATOR_METHOD"] = method
    G.graph["DT_MIN"] = 0.1
    G.graph["GAMMA"] = {"type": "none"}

    def const_dnfr(G):
        for i, n in enumerate(G.nodes()):
            nd = G.nodes[n]
            nd["ΔNFR"] = 5.0 if i % 2 == 0 else -5.0
            nd["νf"] = 1.0
            nd["EPI"] = 0.0

    const_dnfr(G)
    update_epi_via_nodal_equation(G, dt=1.0, method=method)
    validate_canon(G)

    e_min = G.graph["EPI_MIN"]
    e_max = G.graph["EPI_MAX"]
    for i, n in enumerate(G.nodes()):
        epi = G.nodes[n]["EPI"]
        if i % 2 == 0:
            assert epi == pytest.approx(e_max)
        else:
            assert epi == pytest.approx(e_min)
        assert e_min - 1e-6 <= epi <= e_max + 1e-6


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_update_epi_uses_shared_gamma_builder(method, monkeypatch):
    G = nx.path_graph(3)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    G.graph["DT_MIN"] = 0.2
    G.graph["GAMMA"] = {"type": "none"}

    def const_dnfr(G):
        for nd in G.nodes.values():
            nd["ΔNFR"] = 1.0
            nd["νf"] = 1.0

    const_dnfr(G)

    original_builder = integrators_mod._build_gamma_increments
    calls: list[tuple[float, float, str]] = []

    def spy_builder(
        G_arg,
        dt_step_arg,
        t_local_arg,
        *,
        method: str,
        n_jobs: int | None = None,
    ):
        calls.append((dt_step_arg, t_local_arg, method))
        return original_builder(
            G_arg,
            dt_step_arg,
            t_local_arg,
            method=method,
            n_jobs=n_jobs,
        )

    monkeypatch.setattr(
        integrators_mod,
        "_build_gamma_increments",
        spy_builder,
    )

    update_epi_via_nodal_equation(G, dt=0.6, method=method)

    assert len(calls) == 3
    assert all(call_method == method for _, _, call_method in calls)
    assert all(dt_step == pytest.approx(0.2) for dt_step, _, _ in calls)
    assert [t_local for _, t_local, _ in calls] == pytest.approx([0.0, 0.2, 0.4])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_update_epi_skips_eval_gamma_when_none(method, monkeypatch):
    G = nx.path_graph(2)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    G.graph["GAMMA"] = {"type": "none"}

    for nd in G.nodes.values():
        nd["ΔNFR"] = 1.0
        nd["νf"] = 1.0

    calls = 0

    def fake_eval_gamma(*args, **kwargs):
        nonlocal calls
        calls += 1
        return 0.0

    monkeypatch.setattr(integrators_mod, "eval_gamma", fake_eval_gamma)

    update_epi_via_nodal_equation(G, dt=0.3, method=method)

    assert calls == 0


def test_gamma_worker_requires_parallel_graph(monkeypatch):
    original_graph = integrators_mod._PARALLEL_GRAPH
    monkeypatch.setattr(integrators_mod, "_PARALLEL_GRAPH", None)

    with pytest.raises(RuntimeError, match="Parallel Γ worker"):
        integrators_mod._gamma_worker(([0], 0.0))

    integrators_mod._PARALLEL_GRAPH = original_graph


class _FakeArray:
    """Lightweight stand-in for numpy arrays used by increment tests."""

    def __init__(self, data):
        if isinstance(data, _FakeArray):
            self._data = data._data
            self.ndim = data.ndim
            self.shape = data.shape
            return

        data = list(data)
        if data and isinstance(data[0], (list, tuple)):
            rows = [list(row) for row in data]
            self._data = rows
            self.ndim = 2
            width = len(rows[0]) if rows else 0
            self.shape = (len(rows), width)
        else:
            self._data = [float(x) for x in data]
            self.ndim = 1
            self.shape = (len(self._data),)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        if self.ndim == 1:
            return iter(self._data)
        return iter(self._data)

    def __getitem__(self, key):
        if self.ndim == 1:
            return self._data[key]

        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, slice) and row_key == slice(None):
                if isinstance(col_key, int):
                    return _FakeArray(row[col_key] for row in self._data)
            raise TypeError("unsupported slice for _FakeArray")

        return self._data[key]

    def _binary_op(self, other, op):
        if isinstance(other, _FakeArray):
            other_values = other._data
        elif isinstance(other, (list, tuple)):
            other_values = other
        else:
            return _FakeArray(op(value, other) for value in self._data)

        return _FakeArray(op(a, b) for a, b in zip(self._data, other_values))

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        if isinstance(other, _FakeArray):
            return other.__sub__(self)
        if isinstance(other, (int, float)):
            return _FakeArray([other] * len(self._data)).__sub__(self)
        return _FakeArray(other).__sub__(self)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def zeros_like(self):
        return _FakeArray(0.0 for _ in self._data)


class _FakeNumpy:
    """Small shim exposing the subset of numpy used in the tests."""

    @staticmethod
    def asarray(data, dtype=float):  # noqa: ARG003 - dtype kept for API parity
        return _FakeArray(data)

    @staticmethod
    def zeros_like(arr):
        if not isinstance(arr, _FakeArray):
            arr = _FakeArray(arr)
        return arr.zeros_like()


def _use_fake_numpy(monkeypatch):
    """Force integrator helpers to rely on deterministic fake numpy arrays."""

    monkeypatch.setattr(integrators_mod, "get_numpy", lambda: _FakeNumpy())


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_apply_increments_vectorised(monkeypatch, method):
    np_mod = pytest.importorskip("numpy")

    G = nx.path_graph(3)
    epi_start = [0.5, -0.2, 1.3]
    dEPI_prev = [0.1, -0.05, 0.2]
    for idx, node in enumerate(G.nodes):
        nd = G.nodes[node]
        set_attr(nd, integrators_mod.ALIAS_EPI, epi_start[idx])
        set_attr(nd, integrators_mod.ALIAS_DEPI, dEPI_prev[idx])

    dt_step = 0.25
    if method == "rk4":
        staged = {
            0: (0.1, 0.2, 0.3, 0.4),
            1: (0.0, 0.1, 0.2, 0.3),
            2: (-0.2, -0.1, 0.0, 0.1),
        }
    else:
        staged = {
            0: (0.4,),
            1: (-0.2,),
            2: (0.1,),
        }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("chunk helper should not run when numpy is available")

    monkeypatch.setattr(integrators_mod, "get_numpy", lambda: np_mod)
    monkeypatch.setattr(integrators_mod, "_apply_increment_chunk", fail_if_called)

    results = integrators_mod._apply_increments(
        G,
        dt_step,
        staged,
        method=method,
        n_jobs=4,
    )

    expected: dict[int, tuple[float, float, float]] = {}
    for idx, node in enumerate(G.nodes):
        ks = staged[node]
        prev = dEPI_prev[idx]
        base = epi_start[idx]
        if method == "rk4":
            k1, k2, k3, k4 = ks
            epi = base + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            dEPI_dt = k4
        else:
            (k1,) = ks
            epi = base + dt_step * k1
            dEPI_dt = k1
        d2epi = (dEPI_dt - prev) / dt_step if dt_step != 0 else 0.0
        expected[node] = (epi, dEPI_dt, d2epi)

    for node, values in expected.items():
        assert results[node] == pytest.approx(values)


def test_apply_increments_rk4_rejects_incorrect_shape(monkeypatch):
    _use_fake_numpy(monkeypatch)

    G = nx.path_graph(1)
    node = next(iter(G.nodes))
    set_attr(G.nodes[node], integrators_mod.ALIAS_EPI, 0.3)
    set_attr(G.nodes[node], integrators_mod.ALIAS_DEPI, 0.05)

    increments = {node: (0.1, 0.2, 0.3)}

    with pytest.raises(ValueError, match="rk4 increments require four staged values"):
        integrators_mod._apply_increments(G, 0.1, increments, method="rk4")


def test_apply_increments_zero_dt_preserves_second_derivative(monkeypatch):
    _use_fake_numpy(monkeypatch)

    G = nx.path_graph(1)
    node = next(iter(G.nodes))
    set_attr(G.nodes[node], integrators_mod.ALIAS_EPI, -0.4)
    set_attr(G.nodes[node], integrators_mod.ALIAS_DEPI, 0.5)

    increments = {node: (0.1, 0.2, 0.3, 0.4)}

    results = integrators_mod._apply_increments(G, 0.0, increments, method="rk4")

    epi, dEPI_dt, d2EPI = results[node]
    assert epi == pytest.approx(-0.4 + (0.0 / 6.0) * (0.1 + 2 * 0.2 + 2 * 0.3 + 0.4))
    assert dEPI_dt == pytest.approx(0.4)
    assert d2EPI == pytest.approx(0.0)


def test_call_integrator_factory_rejects_keyword_only_arguments():
    G = nx.Graph()

    def kw_only_factory(*, graph):
        raise AssertionError("should not be invoked")

    with pytest.raises(TypeError, match="cannot require keyword-only arguments"):
        runtime_mod._call_integrator_factory(kw_only_factory, G)


def test_call_integrator_factory_rejects_multiple_positional_arguments():
    G = nx.Graph()

    def two_positional_factory(graph, extra):
        raise AssertionError("should not be invoked")

    with pytest.raises(TypeError, match="must accept at most one positional argument"):
        runtime_mod._call_integrator_factory(two_positional_factory, G)
