"""Analytic trajectories and boundary semantics of numerical integration."""

from __future__ import annotations

import copy
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.constants.aliases import ALIAS_THETA
from tnfr.alias import get_attr
from tnfr.dynamics import integrators
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.errors.contextual import NetworkConfigError


SYMPLECTIC_METHODS = ["velocity_verlet", "leapfrog", "yoshida_4th_order"]


def _mechanical_node(q=0.0, velocity=1.0):
    return {EPI_PRIMARY: np.array([q, velocity]), VF_PRIMARY: 1.0,
            DNFR_PRIMARY: np.array([0.0, -q])}


def _harmonic_force(node):
    return np.array([0.0, -node[EPI_PRIMARY][0]])


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_symplectic_free_particle_advances_exactly_one_timestep(method):
    node = _mechanical_node()
    getattr(TNFRSymplecticIntegrator, method)(node, 0.1, lambda _: np.zeros(2))
    assert node[EPI_PRIMARY] == pytest.approx([0.1, 1.0], abs=1e-14)


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_symplectic_zero_step_preserves_state_without_force_evaluation(method):
    node = _mechanical_node(1.0, 0.3)
    epi, pressure = node[EPI_PRIMARY], node[DNFR_PRIMARY]

    def unexpected_force(_):
        pytest.fail("A zero timestep must not evaluate forces")

    getattr(TNFRSymplecticIntegrator, method)(node, 0.0, unexpected_force)
    assert node[EPI_PRIMARY] is epi
    assert node[DNFR_PRIMARY] is pressure


@pytest.mark.parametrize("method,order", [("velocity_verlet", 2), ("yoshida_4th_order", 4)])
def test_symplectic_harmonic_oscillator_has_claimed_convergence_order(method, order):
    exact = np.array([math.cos(1.0), -math.sin(1.0)])
    errors = []
    for steps in [10, 20, 40]:
        node = _mechanical_node(1.0, 0.0)
        for _ in range(steps):
            getattr(TNFRSymplecticIntegrator, method)(node, 1.0 / steps, _harmonic_force)
        errors.append(np.linalg.norm(node[EPI_PRIMARY] - exact))
    measured = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert measured == pytest.approx([order, order], abs=0.04)


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_symplectic_steps_are_time_reversible(method):
    node = _mechanical_node(0.8, 0.3)
    expected = node[EPI_PRIMARY].copy()
    step = getattr(TNFRSymplecticIntegrator, method)
    step(node, 0.2, _harmonic_force)
    step(node, -0.2, _harmonic_force)
    assert node[EPI_PRIMARY] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
@pytest.mark.parametrize("dt", [math.nan, math.inf, -math.inf])
def test_symplectic_nonfinite_steps_are_rejected_before_mutation(method, dt):
    node = _mechanical_node(1.0, 0.0)
    epi, pressure = node[EPI_PRIMARY], node[DNFR_PRIMARY]
    with pytest.raises(ValueError, match="finite"):
        getattr(TNFRSymplecticIntegrator, method)(node, dt, _harmonic_force)
    assert node[EPI_PRIMARY] is epi
    assert node[DNFR_PRIMARY] is pressure


def _graph(*, extended=False):
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.graph.update(DT=0.1, DT_MIN=0.0, GAMMA={"type": "none"},
                       use_extended_dynamics=extended)
    graph.nodes[0].update({EPI_PRIMARY: 0.4, VF_PRIMARY: 1.0,
                           DNFR_PRIMARY: 0.0, "theta": 0.0})
    return graph


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("dt", [-0.1, math.nan, math.inf, -math.inf])
def test_explicit_and_default_invalid_nodal_timesteps_share_validation(default, dt):
    graph = _graph()
    if default:
        graph.graph["DT"] = dt
    original = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError):
        integrators.update_epi_via_nodal_equation(graph, dt=None if default else dt)
    assert dict(graph.nodes[0]) == dict(original.nodes[0])
    assert graph.graph == original.graph


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("vectorized", [False, True])
def test_zero_timestep_is_identity_even_with_clipping(monkeypatch, extended, vectorized):
    graph = _graph(extended=extended)
    graph.graph["CLIP_MODE"] = "soft"
    graph.nodes[0][EPI_PRIMARY] = -0.4
    original = copy.deepcopy(graph)
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    integrators.update_epi_via_nodal_equation(graph, dt=0.0)
    assert dict(graph.nodes[0]) == dict(original.nodes[0])
    assert graph.graph == original.graph


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("vf,pressure", [(0.0, 1.0), (1.0, 0.0)])
def test_zero_nodal_derivative_is_not_changed_by_soft_clipping(monkeypatch, vectorized, vf, pressure):
    graph = _graph()
    graph.graph["CLIP_MODE"] = "soft"
    graph.nodes[0].update({VF_PRIMARY: vf, DNFR_PRIMARY: pressure})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert graph.nodes[0][EPI_PRIMARY] == 0.4


def test_extended_default_timestep_and_configured_epi_bounds():
    graph = _graph(extended=True)
    graph.graph.update(DT=0.02, EPI_MIN=-2.0, EPI_MAX=2.0)
    graph.nodes[0].update({EPI_PRIMARY: -1.0, DNFR_PRIMARY: 0.5})
    integrators.update_epi_via_nodal_equation(graph)
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(-0.99)
    assert graph.graph["_t"] == pytest.approx(0.02)


def test_extended_euler_is_independent_of_node_insertion_order():
    results = []
    for order in [(0, 1, 2), (2, 1, 0)]:
        graph = nx.Graph()
        graph.add_nodes_from(order)
        graph.add_edges_from([(0, 1), (1, 2)])
        inject_defaults(graph)
        graph.graph.update(use_extended_dynamics=True, DT_MIN=0.0)
        for node in graph:
            graph.nodes[node].update({EPI_PRIMARY: 0.4, VF_PRIMARY: 1.0,
                DNFR_PRIMARY: 0.1 * (node + 1), "theta": 0.4 * node})
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
        results.append(np.array([[graph.nodes[i][key] for key in
                                 [EPI_PRIMARY, "theta", DNFR_PRIMARY]] for i in range(3)]))
    assert results[0] == pytest.approx(results[1], abs=1e-14)


def test_extended_does_not_silently_substitute_euler_for_rk4():
    graph = _graph(extended=True)
    original = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError, match="euler"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1, method="rk4")
    assert dict(graph.nodes[0]) == dict(original.nodes[0])


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph])
def test_flux_divergence_is_independent_of_graph_size_and_disconnected_padding(graph_type):
    graph = graph_type()
    graph.add_edge(0, 1, weight=4.0)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=2.0)
    expected = {node: integrators._compute_flux_divergence_centralized(graph, {0: 2., 1: 1.}, node)
                for node in graph}
    graph.add_nodes_from(range(2, 101))
    flux = {node: 1.0 for node in graph}
    flux[0] = 2.0
    actual = integrators.compute_flux_divergence_vectorized(graph, flux)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])
    assert all(actual[node] == 0.0 for node in range(2, 101))


def test_cpu_canonical_integrator_does_not_import_optional_gpu_system(monkeypatch):
    import builtins
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    original_import = builtins.__import__

    def import_without_gpu(name, *args, **kwargs):
        if "unified_gpu_system" in name:
            raise ImportError("optional GPU dependencies deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gpu)
    graph = _graph()
    graph.nodes[0][DNFR_PRIMARY] = 0.5
    result = integrate_canonical_nodal_equation(graph, dt=0.1, max_steps=1, use_gpu=False)
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.45)
    assert result["backend_used"] == "cpu"


def test_canonical_zero_step_is_not_replaced_by_default():
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    graph = _graph()
    graph.nodes[0][DNFR_PRIMARY] = 0.5
    original = copy.deepcopy(graph)
    result = integrate_canonical_nodal_equation(graph, dt=0.0, max_steps=1, use_gpu=False)
    assert dict(graph.nodes[0]) == dict(original.nodes[0])
    assert result["steps"] == 0
    assert result["parameters"]["dt"] == 0.0


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_canonical_constant_pressure_is_exact_and_preserves_zero_tolerance(method):
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    graph = _graph()
    graph.nodes[0][DNFR_PRIMARY] = 0.5
    result = integrate_canonical_nodal_equation(
        graph, dt=0.1, max_steps=5, tolerance=0.0, method=method, use_gpu=False
    )
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.65)
    assert result["parameters"]["tolerance"] == 0.0
    assert result["steps"] == 5


@pytest.mark.parametrize("parameter,value", [
    ("dt", math.nan), ("dt", math.inf), ("max_steps", 0), ("max_steps", 1.5),
    ("tolerance", math.nan), ("tolerance", -0.1),
])
def test_canonical_integrator_rejects_invalid_parameters_without_mutation(parameter, value):
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    graph = _graph()
    original = copy.deepcopy(graph)
    arguments = {"dt": 0.1, "max_steps": 1, "tolerance": 0.0, "use_gpu": False}
    arguments[parameter] = value
    with pytest.raises(ValueError):
        integrate_canonical_nodal_equation(graph, **arguments)
    assert dict(graph.nodes[0]) == dict(original.nodes[0])


@pytest.mark.parametrize("method,order", [("euler", 1), ("rk4", 4)])
@pytest.mark.parametrize("vectorized", [False, True])
def test_nodal_harmonic_forcing_has_claimed_quadrature_order(monkeypatch, method, order, vectorized):
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    beta, omega = 0.2, 1.7
    exact = 0.4 + beta * (1.0 - math.cos(omega)) / omega
    errors = []
    for steps in [10, 20, 40]:
        graph = _graph()
        graph.graph["GAMMA"] = {"type": "harmonic", "beta": beta, "omega": omega, "phi": 0.0}
        for _ in range(steps):
            integrators.update_epi_via_nodal_equation(graph, dt=1.0 / steps, method=method)
        errors.append(abs(graph.nodes[0][EPI_PRIMARY] - exact))
    assert np.log2(np.asarray(errors[:-1]) / errors[1:]) == pytest.approx([order, order], abs=0.05)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_array_clipping_matches_scalar_boundary_policy(mode):
    from tnfr.dynamics.structural_clip import structural_clip, structural_clip_array

    values = np.array([-3.0, -1.0, 0.1, 0.4, 0.9, 3.0])
    expected = [structural_clip(value, lo=-1.0, hi=1.0, mode=mode) for value in values]
    assert structural_clip_array(values, lo=-1.0, hi=1.0, mode=mode) == pytest.approx(expected, abs=1e-15)


def _phase_pair(phase_shift=0.0):
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(use_extended_dynamics=True, DT_MIN=0.0)
    for node in graph:
        graph.nodes[node].update({EPI_PRIMARY: 0.4, VF_PRIMARY: 0.0,
                                 DNFR_PRIMARY: 0.0, "phase": node * math.pi / 2 + phase_shift})
    return graph


def test_extended_phase_uses_canonical_current_and_respects_wrapped_aliases():
    results = []
    for shift in [0.0, 4.0 * math.pi]:
        graph = _phase_pair(shift)
        coupling = integrators._estimate_local_coupling_strength(graph, 0)
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
        phases = [get_attr(graph.nodes[node], ALIAS_THETA) for node in graph]
        # J_phi=(+1,-1) at phase separation pi/2; pressure and its flux are zero.
        expected = [0.135 * coupling * 0.1, math.pi / 2 - 0.135 * coupling * 0.1]
        assert phases == pytest.approx(expected, abs=2e-14)
        assert all(0 <= phase < 2 * math.pi for phase in phases)
        assert all(graph.nodes[node][EPI_PRIMARY] == 0.4 for node in graph)
        results.append(phases)
    assert results[0] == pytest.approx(results[1], abs=2e-14)


def test_extended_phase_pair_has_first_order_euler_convergence():
    # For J_phi=(sin(delta),-sin(delta)), delta'=-2*0.135*kappa*sin(delta).
    graph = _phase_pair()
    coupling = integrators._estimate_local_coupling_strength(graph, 0)
    duration = 5.0
    exact_delta = 2 * math.atan(math.exp(-2 * 0.135 * coupling * duration))
    errors = []
    for steps in [10, 20, 40]:
        graph = _phase_pair()
        for _ in range(steps):
            integrators.update_epi_via_nodal_equation(graph, dt=duration / steps)
        actual_delta = get_attr(graph.nodes[1], ALIAS_THETA) - get_attr(graph.nodes[0], ALIAS_THETA)
        errors.append(abs(actual_delta - exact_delta))
    assert np.log2(np.asarray(errors[:-1]) / errors[1:]) == pytest.approx([1.0, 1.0], abs=0.06)


@pytest.mark.parametrize("vectorized", [False, True])
def test_empty_graph_clock_advances_consistently(monkeypatch, vectorized):
    graph = nx.Graph()
    graph.graph.update(_t=2.0, DT_MIN=0.0)
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert graph.graph["_t"] == pytest.approx(2.1)


@pytest.mark.parametrize("value", [np.float32(0.1), np.float64(0.1), np.int64(1)])
@pytest.mark.parametrize("from_graph", [False, True])
def test_real_numpy_timestep_scalars_preserve_constant_derivative(value, from_graph):
    graph = _graph()
    graph.nodes[0][DNFR_PRIMARY] = 0.2
    if from_graph:
        graph.graph["DT"] = value
    integrators.update_epi_via_nodal_equation(graph, dt=None if from_graph else value)
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.4 + float(value) * 0.2)
    assert graph.graph["_t"] == pytest.approx(float(value))


@pytest.mark.parametrize("value", [np.float32(-0.1), np.float32(np.nan), np.float32(np.inf)])
@pytest.mark.parametrize("from_graph", [False, True])
def test_invalid_numpy_timesteps_fail_before_state_changes(value, from_graph):
    graph = _graph()
    if from_graph:
        graph.graph["DT"] = value
    original_node = dict(graph.nodes[0])
    with pytest.raises(NetworkConfigError):
        integrators.update_epi_via_nodal_equation(graph, dt=None if from_graph else value)
    assert dict(graph.nodes[0]) == original_node
    assert "_t" not in graph.graph


@pytest.mark.parametrize("value", [False, True])
def test_timestep_real_scalar_validation_preserves_existing_bool_policy(value):
    graph = _graph()
    step, count, _, _ = integrators.prepare_integration_params(graph, dt=value)
    assert step == float(value)
    assert count == 1
