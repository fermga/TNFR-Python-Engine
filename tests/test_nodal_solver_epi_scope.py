"""Scalar nodal solvers must not evolve a rich EPI magnitude as its state.

Unsupported initial form is rejected before solver-owned writes or field/Gamma
cache work. This is input admission, not rollback of arbitrary caller callbacks.
Uniform-real BEPI remains the same signed scalar chart on every tested route.
"""

import pickle

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import integrators
from tnfr.dynamics.canonical import integrate_canonical_nodal_equation
from tnfr.errors.contextual import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.types import ensure_bepi, serialize_bepi

ROUTES = (
    ("default", "euler", True),
    ("default", "rk4", True),
    ("default", "euler", False),
    ("default", "rk4", False),
    ("extended", "euler", True),
    ("extended", "euler", False),
    ("canonical", "euler", True),
    ("canonical", "rk4", True),
)


def _graph(route):
    graph = nx.path_graph(3)
    inject_defaults(graph)
    graph.graph.update(
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        use_extended_dynamics=route == "extended",
        _t=2.0,
    )
    for node in graph:
        graph.nodes[node].update(
            {ALIAS_EPI[0]: -0.5, ALIAS_VF[0]: 1.0, ALIAS_DNFR[0]: 0.25, "theta": 0.0}
        )
    return graph


def _rich_form(kind, serialized=False):
    if kind == "nonuniform":
        value = BEPIElement((0.5, -0.5), (0.25, -0.25), (0.0, 1.0))
    else:
        value = BEPIElement((0.5j, 0.5j), (0.5j, 0.5j), (0.0, 1.0))
    return serialize_bepi(value) if serialized else value


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _run(graph, route, method, dt=0.25):
    if route == "canonical":
        return integrate_canonical_nodal_equation(
            graph, dt=dt, method=method, max_steps=1, tolerance=0.0, use_gpu=False
        )
    if route == "extended":
        return integrators.update_epi_via_nodal_equation(graph, dt=dt, method=method)
    return integrators.DefaultIntegrator().integrate(
        graph, dt=dt, t=None, method=method, n_jobs=1
    )


@pytest.mark.parametrize("route,method,vectorized", ROUTES)
@pytest.mark.parametrize("kind", ("nonuniform", "complex"))
@pytest.mark.parametrize("serialized", (False, True))
def test_rich_epi_rejected_before_state_and_cache_writes(
    monkeypatch, route, method, vectorized, kind, serialized
):
    graph = _graph(route)
    # Last-node failure must not advance valid earlier nodes. Zero capacity
    # also cannot authorize replacement of this unsupported form by a number.
    graph.nodes[2][ALIAS_EPI[0]] = _rich_form(kind, serialized)
    graph.nodes[2][ALIAS_VF[0]] = 0.0
    if len(ALIAS_EPI) > 1:
        graph.nodes[2][ALIAS_EPI[1]] = -0.5  # No permissive alias fallback.
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported EPI reached forcing or cached field evaluation")

    monkeypatch.setattr(integrators, "_get_gamma_spec", unexpected)
    monkeypatch.setattr(integrators, "eval_gamma_vectorized", unexpected)
    monkeypatch.setattr("tnfr.physics.extended.compute_phase_current", unexpected)
    before = _snapshot(graph)
    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        _run(graph, route, method)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("route,method,vectorized", ROUTES)
@pytest.mark.parametrize("serialized", (False, True))
def test_negative_uniform_bepi_retains_signed_nodal_evolution(
    monkeypatch, route, method, vectorized, serialized
):
    graph = _graph(route)
    value = ensure_bepi(-0.5)
    value = serialize_bepi(value) if serialized else value
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = value
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    _run(graph, route, method)
    # Stored pressure and capacity give -1/2 + (1/4)*(1)*(1/4)=-7/16.
    assert all(graph.nodes[node][ALIAS_EPI[0]] == -7.0 / 16.0 for node in graph)


@pytest.mark.parametrize("route,method,vectorized", ROUTES)
def test_zero_step_does_not_represent_or_modify_rich_epi(
    monkeypatch, route, method, vectorized
):
    graph = _graph(route)
    graph.nodes[2][ALIAS_EPI[0]] = _rich_form("nonuniform")
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    before = _snapshot(graph)
    _run(graph, route, method, dt=0.0)
    assert _snapshot(graph) == before
