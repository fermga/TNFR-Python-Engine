"""Regression coverage for canonical Gamma cache invalidation."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.dynamics import integrators
from tnfr.gamma import eval_gamma


def _graph() -> nx.Graph:
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.graph.update(
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
    )
    graph.nodes[0].update(
        {
            EPI_PRIMARY: 0.4,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.0,
            "theta": 0.0,
        }
    )
    return graph


@pytest.mark.parametrize("mutation_mode", ["replace", "in_place"])
@pytest.mark.parametrize("vectorized", [False, True])
def test_live_gamma_configuration_invalidates_cached_none(
    monkeypatch: pytest.MonkeyPatch,
    mutation_mode: str,
    vectorized: bool,
) -> None:
    graph = _graph()
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)

    integrators.update_epi_via_nodal_equation(
        graph,
        dt=0.25,
        method="euler",
    )
    assert graph.graph["_gamma_spec"]["type"] == "none"

    active = {
        "type": "kuramoto_linear",
        "beta": 0.5,
        "R0": 0.0,
    }
    if mutation_mode == "replace":
        graph.graph["GAMMA"] = active
    else:
        graph.graph["GAMMA"].update(active)

    integrators.update_epi_via_nodal_equation(
        graph,
        dt=0.25,
        method="euler",
    )

    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.525)
    assert graph.graph["_gamma_spec"]["type"] == "kuramoto_linear"


def test_repeated_requested_time_does_not_reuse_stale_global_phase_order():
    graph = nx.path_graph(2)
    graph.graph["GAMMA"] = {"type": "kuramoto_linear", "beta": 1.0}
    for node in graph:
        graph.nodes[node]["theta"] = 0.0
    assert eval_gamma(graph, 0, 0.0, strict=True) == 1.0

    graph.nodes[1]["theta"] = math.pi / 2
    # R*cos(theta_i-psi) equals the projection of the mean unit phasor
    # onto node 0's direction: (1 + cos(pi/2))/2.
    assert eval_gamma(graph, 0, 0.0, strict=True) == pytest.approx(0.5)
