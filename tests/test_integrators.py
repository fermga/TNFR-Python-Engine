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

    def spy_builder(G_arg, dt_step_arg, t_local_arg, *, method: str):
        calls.append((dt_step_arg, t_local_arg, method))
        return original_builder(G_arg, dt_step_arg, t_local_arg, method=method)

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
