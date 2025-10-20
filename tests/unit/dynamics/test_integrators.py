"""Unit tests for integrator routines driving node evolution."""



from __future__ import annotations
import pytest

import networkx as nx
from tnfr.alias import set_attr
from tnfr.constants import inject_defaults
from tnfr.initialization import init_node_attrs
from tnfr.dynamics import update_epi_via_nodal_equation, validate_canon
from tnfr.dynamics import integrators as integrators_mod


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
