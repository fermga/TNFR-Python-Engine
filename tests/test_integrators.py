"""Pruebas de integrators."""

from __future__ import annotations
import pytest

import networkx as nx
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

    def spy_builder(G_arg, dt_step_arg, t_local_arg, *, method: str, **kwargs):
        calls.append((dt_step_arg, t_local_arg, method))
        return original_builder(
            G_arg, dt_step_arg, t_local_arg, method=method, **kwargs
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


def _build_increment_graph():
    G = nx.Graph()
    node_data = {
        "b": (1.0, 0.5),
        "a": (1.5, -0.2),
        "c": (2.0, 1.1),
    }
    for node, (vf, dnfr) in node_data.items():
        G.add_node(node)
        nd = G.nodes[node]
        nd["νf"] = vf
        nd["ΔNFR"] = dnfr
    return G, node_data


@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize("use_numpy", [True, False])
def test_collect_nodal_increments_matches_expected(method, use_numpy, monkeypatch):
    if use_numpy:
        np = pytest.importorskip("numpy")
    else:
        np = None

    G, node_data = _build_increment_graph()
    nodes = list(G.nodes())

    if method == "euler":
        gamma_maps = ({"b": 0.1, "c": -0.4},)
    else:
        gamma_maps = (
            {"b": 0.1, "a": -0.2, "c": 0.3},
            {"a": 0.5, "c": 0.2},
            {"b": -0.3},
            {"b": 0.4, "c": -0.1},
        )

    base = {n: vf * dnfr for n, (vf, dnfr) in node_data.items()}
    expected = {}
    for node in nodes:
        contributions = []
        for gm in gamma_maps:
            contributions.append(base[node] + gm.get(node, 0.0))
        expected[node] = tuple(contributions)

    with monkeypatch.context() as ctx:
        ctx.setattr(integrators_mod, "get_numpy", lambda: np)
        result = integrators_mod._collect_nodal_increments(
            G, gamma_maps, method=method
        )

    assert list(result.keys()) == nodes
    for node, values in result.items():
        assert values == pytest.approx(expected[node])
