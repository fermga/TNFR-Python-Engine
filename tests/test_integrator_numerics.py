"""Analytic trajectories and boundary semantics of numerical integration."""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import integrators
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.errors.contextual import FrequencyError, NetworkConfigError

SYMPLECTIC_METHODS = ["velocity_verlet", "leapfrog", "yoshida_4th_order"]


def _mechanical_node(q=0.0, velocity=1.0):
    return {
        EPI_PRIMARY: np.array([q, velocity]),
        VF_PRIMARY: 1.0,
        DNFR_PRIMARY: np.array([0.0, -q]),
    }


def _harmonic_force(node):
    return np.array([0.0, -node[EPI_PRIMARY][0]])


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_symplectic_free_particle_advances_exactly_one_timestep(method):
    node = _mechanical_node()
    getattr(TNFRSymplecticIntegrator, method)(node, 0.1, lambda _: np.zeros(2))
    assert node[EPI_PRIMARY] == pytest.approx([0.1, 1.0], abs=1e-14)


@pytest.mark.parametrize("method", ["velocity_verlet", "yoshida_4th_order"])
def test_symplectic_zero_step_preserves_state_without_force_evaluation(method):
    node = _mechanical_node(1.0, 0.3)
    epi, pressure = node[EPI_PRIMARY], node[DNFR_PRIMARY]

    def unexpected_force(_):
        pytest.fail("A zero timestep must not evaluate forces")

    getattr(TNFRSymplecticIntegrator, method)(node, 0.0, unexpected_force)
    assert node[EPI_PRIMARY] is epi
    assert node[DNFR_PRIMARY] is pressure


@pytest.mark.parametrize(
    "method,order", [("velocity_verlet", 2), ("yoshida_4th_order", 4)]
)
def test_symplectic_harmonic_oscillator_has_claimed_convergence_order(method, order):
    exact = np.array([math.cos(1.0), -math.sin(1.0)])
    errors = []
    for steps in [10, 20, 40]:
        node = _mechanical_node(1.0, 0.0)
        for _ in range(steps):
            getattr(TNFRSymplecticIntegrator, method)(
                node, 1.0 / steps, _harmonic_force
            )
        errors.append(np.linalg.norm(node[EPI_PRIMARY] - exact))
    measured = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert measured == pytest.approx([order, order], abs=0.04)


@pytest.mark.parametrize("method", ["velocity_verlet", "yoshida_4th_order"])
def test_symplectic_steps_are_time_reversible(method):
    node = _mechanical_node(0.8, 0.3)
    expected = node[EPI_PRIMARY].copy()
    step = getattr(TNFRSymplecticIntegrator, method)
    step(node, 0.2, _harmonic_force)
    step(node, -0.2, _harmonic_force)
    assert node[EPI_PRIMARY] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    "method,dt",
    [
        ("velocity_verlet", math.nan),
        ("velocity_verlet", math.inf),
        ("yoshida_4th_order", math.nan),
    ],
)
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
    graph.graph.update(
        DT=0.1, DT_MIN=0.0, GAMMA={"type": "none"}, use_extended_dynamics=extended
    )
    graph.nodes[0].update(
        {EPI_PRIMARY: 0.4, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0, "theta": 0.0}
    )
    return graph


@pytest.mark.parametrize(
    "default,dt", [(False, -0.1), (True, -0.1), (False, math.nan), (True, math.inf)]
)
def test_explicit_and_default_invalid_nodal_timesteps_share_validation(default, dt):
    graph = _graph()
    if default:
        graph.graph["DT"] = dt
    original = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError):
        integrators.update_epi_via_nodal_equation(graph, dt=None if default else dt)
    assert dict(graph.nodes[0]) == dict(original.nodes[0])
    assert graph.graph == original.graph


@pytest.mark.parametrize("method", ["", False, 0])
def test_invalid_explicit_method_does_not_silently_select_the_graph_default(method):
    graph = _graph()
    graph.graph["INTEGRATOR_METHOD"] = "euler"
    before = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError, match="method"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1, method=method)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


@pytest.mark.parametrize("extended", [False, True])
def test_zero_timestep_is_identity_even_with_clipping(extended):
    graph = _graph(extended=extended)
    graph.graph["CLIP_MODE"] = "soft"
    graph.nodes[0][EPI_PRIMARY] = -0.4
    original = copy.deepcopy(graph)
    integrators.update_epi_via_nodal_equation(graph, dt=0.0)
    assert dict(graph.nodes[0]) == dict(original.nodes[0])
    assert graph.graph == original.graph


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("vf,pressure", [(0.0, 1.0), (1.0, 0.0)])
def test_zero_nodal_derivative_is_not_changed_by_soft_clipping(
    monkeypatch, vectorized, vf, pressure
):
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
            graph.nodes[node].update(
                {
                    EPI_PRIMARY: 0.4,
                    VF_PRIMARY: 1.0,
                    DNFR_PRIMARY: 0.1 * (node + 1),
                    "theta": 0.4 * node,
                }
            )
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
        results.append(
            np.array(
                [
                    [
                        graph.nodes[i][key]
                        for key in [EPI_PRIMARY, "theta", DNFR_PRIMARY]
                    ]
                    for i in range(3)
                ]
            )
        )
    assert results[0] == pytest.approx(results[1], abs=1e-14)


def test_extended_does_not_silently_substitute_euler_for_rk4():
    graph = _graph(extended=True)
    original = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError, match="euler"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1, method="rk4")
    assert dict(graph.nodes[0]) == dict(original.nodes[0])


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph])
def test_flux_divergence_is_independent_of_graph_size_and_disconnected_padding(
    graph_type,
):
    graph = graph_type()
    graph.add_edge(0, 1, weight=4.0)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=2.0)
    expected = {
        node: integrators._compute_flux_divergence_centralized(
            graph, {0: 2.0, 1: 1.0}, node
        )
        for node in graph
    }
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
    result = integrate_canonical_nodal_equation(
        graph, dt=0.1, max_steps=1, use_gpu=False
    )
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.45)
    assert result["backend_used"] == "cpu"


def test_canonical_zero_step_is_not_replaced_by_default():
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    graph = _graph()
    graph.nodes[0][DNFR_PRIMARY] = 0.5
    original = copy.deepcopy(graph)
    result = integrate_canonical_nodal_equation(
        graph, dt=0.0, max_steps=1, use_gpu=False
    )
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


@pytest.mark.parametrize(
    "aliases,value,error",
    [
        (ALIAS_VF, -1.0, FrequencyError),
        (ALIAS_VF, math.nan, FrequencyError),
        (ALIAS_VF, math.inf, FrequencyError),
        (ALIAS_DNFR, math.nan, NetworkConfigError),
        (ALIAS_DNFR, math.inf, NetworkConfigError),
    ],
)
def test_canonical_backend_rejects_authoritative_invalid_coefficients_without_writes(
    aliases, value, error
):
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    graph = _graph()
    graph.add_node(1, **{EPI_PRIMARY: 0.2, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.25})
    graph.nodes[1][aliases[0]] = value
    graph.nodes[1][aliases[1]] = 0.5
    before = copy.deepcopy(graph)
    with pytest.raises(error):
        integrate_canonical_nodal_equation(graph, dt=0.1, max_steps=2, use_gpu=False)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_validated_canonical_derivative_rejects_overflowing_finite_factors():
    from tnfr.dynamics.canonical import compute_canonical_nodal_derivative

    with pytest.raises(NetworkConfigError, match="nodal product"):
        compute_canonical_nodal_derivative(1e308, 2.0)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_canonical_backend_late_overflow_preserves_all_graph_outputs(method):
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation
    from tnfr.errors import TNFRValueError

    graph = _graph()
    graph.add_node(1, **{EPI_PRIMARY: 0.0, VF_PRIMARY: 1.0, DNFR_PRIMARY: 8e307})
    before = copy.deepcopy(graph)
    with np.errstate(over="ignore"):
        with pytest.raises(TNFRValueError, match="finite"):
            integrate_canonical_nodal_equation(
                graph, dt=1.0, max_steps=3, method=method, tolerance=0.0, use_gpu=False
            )
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_canonical_backend_step_norm_does_not_square_into_overflow():
    from tnfr.dynamics.canonical import integrate_canonical_nodal_equation

    graph = _graph()
    graph.nodes[0].update({EPI_PRIMARY: 0.0, DNFR_PRIMARY: 1e200})
    graph.add_node(1, **{EPI_PRIMARY: 0.0, VF_PRIMARY: 1.0, DNFR_PRIMARY: -1e200})
    result = integrate_canonical_nodal_equation(
        graph, dt=0.5, max_steps=1, tolerance=0.0, use_gpu=False
    )
    assert graph.nodes[0][EPI_PRIMARY] == 5e199
    assert graph.nodes[1][EPI_PRIMARY] == -5e199
    assert result["final_error"] == math.hypot(5e199, -5e199)
    assert not result["converged"]


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("dt", math.nan),
        ("dt", math.inf),
        ("max_steps", 0),
        ("max_steps", 1.5),
        ("tolerance", math.nan),
        ("tolerance", -0.1),
    ],
)
def test_canonical_integrator_rejects_invalid_parameters_without_mutation(
    parameter, value
):
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
def test_nodal_harmonic_forcing_has_claimed_quadrature_order(
    monkeypatch, method, order, vectorized
):
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    beta, omega = 0.2, 1.7
    exact = 0.4 + beta * (1.0 - math.cos(omega)) / omega
    errors = []
    for steps in [10, 20, 40]:
        graph = _graph()
        graph.graph["GAMMA"] = {
            "type": "harmonic",
            "beta": beta,
            "omega": omega,
            "phi": 0.0,
        }
        for _ in range(steps):
            integrators.update_epi_via_nodal_equation(
                graph, dt=1.0 / steps, method=method
            )
        errors.append(abs(graph.nodes[0][EPI_PRIMARY] - exact))
    assert np.log2(np.asarray(errors[:-1]) / errors[1:]) == pytest.approx(
        [order, order], abs=0.05
    )


def _phase_pair(phase_shift=0.0):
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(use_extended_dynamics=True, DT_MIN=0.0)
    for node in graph:
        graph.nodes[node].update(
            {
                EPI_PRIMARY: 0.4,
                VF_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.0,
                "phase": node * math.pi / 2 + phase_shift,
            }
        )
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
        actual_delta = get_attr(graph.nodes[1], ALIAS_THETA) - get_attr(
            graph.nodes[0], ALIAS_THETA
        )
        errors.append(abs(actual_delta - exact_delta))
    assert np.log2(np.asarray(errors[:-1]) / errors[1:]) == pytest.approx(
        [1.0, 1.0], abs=0.06
    )


@pytest.mark.parametrize(
    "value,from_graph",
    [(np.float32(0.1), False), (np.float64(0.1), True), (np.int64(1), True)],
)
def test_real_numpy_timestep_scalars_preserve_constant_derivative(value, from_graph):
    graph = _graph()
    graph.nodes[0][DNFR_PRIMARY] = 0.2
    if from_graph:
        graph.graph["DT"] = value
    integrators.update_epi_via_nodal_equation(graph, dt=None if from_graph else value)
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.4 + float(value) * 0.2)
    assert graph.graph["_t"] == pytest.approx(float(value))


@pytest.mark.parametrize(
    "value,from_graph",
    [(np.float32(-0.1), True), (np.float32(np.nan), False), (np.float32(np.inf), True)],
)
def test_invalid_numpy_timesteps_fail_before_state_changes(value, from_graph):
    graph = _graph()
    if from_graph:
        graph.graph["DT"] = value
    original_node = dict(graph.nodes[0])
    with pytest.raises(NetworkConfigError):
        integrators.update_epi_via_nodal_equation(
            graph, dt=None if from_graph else value
        )
    assert dict(graph.nodes[0]) == original_node
    assert "_t" not in graph.graph


@pytest.mark.parametrize(
    "field,value,extended",
    [
        ("dt", False, False),
        ("dt", True, True),
        ("dt", Fraction(1, 2**2000), False),
        ("DT", "0.1", True),
        ("t", "1.5", False),
        ("_t", True, True),
        ("DT_MIN", True, False),
        ("DT_MIN", Fraction(1, 2**2000), True),
    ],
)
def test_solver_clock_uses_shared_raw_real_admission(field, value, extended):
    graph = _graph(extended=extended)
    kwargs = {"dt": 0.1}
    if field in ("dt", "t"):
        kwargs[field] = value
    else:
        graph.graph[field] = value
        if field == "DT":
            kwargs["dt"] = None
    before = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError):
        integrators.update_epi_via_nodal_equation(graph, **kwargs)
    assert graph.graph == before.graph
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))


@pytest.mark.parametrize("vectorized,previous", [(True, "0.0"), (False, True)])
def test_retained_rate_cannot_be_coerced_into_acceleration_history(
    vectorized, previous, monkeypatch
):
    graph = _graph()
    graph.nodes[0][ALIAS_DEPI[0]] = previous
    graph.nodes[0][ALIAS_DEPI[1]] = 0.0
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    before = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert graph.graph == before.graph
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))


@pytest.mark.parametrize("vectorized", [False, True])
def test_held_zero_sum_pressure_does_not_imply_binary64_mean_conservation(
    monkeypatch, vectorized
):
    # A prescribed integrator-input fixture, not a canonically generated pressure field.
    graph = nx.empty_graph(6)
    inject_defaults(graph)
    graph.graph.update(DT_MIN=1 / 16, GAMMA={"type": "none"}, CLIP_MODE="hard")
    pressure = 2.0**-50
    for node in graph:
        graph.nodes[node].update(
            {
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                DNFR_PRIMARY: pressure if node % 2 == 0 else -pressure,
            }
        )
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    integrators.update_epi_via_nodal_equation(graph, dt=0.25, method="euler")
    values = tuple(Fraction(graph.nodes[node][EPI_PRIMARY]) for node in graph)
    assert values == (Fraction(1, 2), Fraction(1, 2) - Fraction(1, 2**52)) * 3
    assert sum(values) / 6 - Fraction(1, 2) == -Fraction(1, 2**53)
    assert sum(graph.nodes[node][DNFR_PRIMARY] for node in graph) == 0
    assert graph.graph["_t"] == 0.25


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("clip_mode", ["hard", "soft"])
@pytest.mark.parametrize("pressure", [-(2.0**-51), 2.0**-50])
def test_held_rounding_stasis_preserves_epi_but_updates_rate_and_time(
    monkeypatch,
    vectorized,
    clip_mode,
    pressure,
):
    graph = _graph()
    graph.graph.update(DT_MIN=1 / 16, CLIP_MODE=clip_mode)
    graph.nodes[0].update({EPI_PRIMARY: 0.5, DNFR_PRIMARY: pressure})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    integrators.update_epi_via_nodal_equation(graph, dt=0.25, method="euler")
    assert graph.nodes[0][EPI_PRIMARY] == 0.5
    assert get_attr(graph.nodes[0], ALIAS_DEPI) == pressure
    assert get_attr(graph.nodes[0], ALIAS_D2EPI) == 0.0
    assert graph.graph["_t"] == 0.25


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("route", ["batch", "default"])
def test_euler_preserves_separate_multiply_and_add(monkeypatch, vectorized, route):
    graph = _graph()
    step, rate = 1.0 + 2.0**-52, 1.0 - 2.0**-52
    graph.nodes[0].update({EPI_PRIMARY: -1.0, DNFR_PRIMARY: rate})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    if route == "batch":
        result = integrators._apply_increments(
            graph, step, {0: (rate,)}, method="euler"
        )
        actual, derivative = result[0][:2]
    else:
        integrators.update_epi_via_nodal_equation(graph, dt=step, method="euler")
        actual = graph.nodes[0][EPI_PRIMARY]
        derivative = get_attr(graph.nodes[0], ALIAS_DEPI)
    # The exact product-plus-sum is representable; an FMA would retain it.
    assert Fraction(-1) + Fraction(step) * Fraction(rate) == -Fraction(1, 2**104)
    assert actual == 0.0
    assert derivative == rate


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("route", ["batch", "default"])
@pytest.mark.parametrize("step,rate", [(math.ulp(0.0), 1e307), (0.5, 1e308)])
def test_rk4_retains_finite_held_response_at_intermediate_range_limits(
    monkeypatch, vectorized, route, step, rate
):
    graph = _graph()
    graph.graph.update(EPI_MIN=-1e308, EPI_MAX=1e308)
    graph.nodes[0].update({EPI_PRIMARY: 0.0, DNFR_PRIMARY: rate, ALIAS_DEPI[0]: rate})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    if route == "batch":
        actual, derivative, acceleration = integrators._apply_increments(
            graph, step, {0: (rate,) * 4}, method="rk4"
        )[0]
    else:
        integrators.update_epi_via_nodal_equation(graph, dt=step, method="rk4")
        actual = graph.nodes[0][EPI_PRIMARY]
        derivative = get_attr(graph.nodes[0], ALIAS_DEPI)
        acceleration = get_attr(graph.nodes[0], ALIAS_D2EPI)
        assert graph.graph["_t"] == step
    # A constant supplied rate integrates exactly to h*rate; h/6 may be zero
    # or the unscaled 1:2:2:1 sum infinite although this answer is finite.
    assert actual == float(Fraction(step) * Fraction(rate))
    assert actual > 0.0
    assert derivative == rate
    assert acceleration == 0.0


@pytest.mark.parametrize("vectorized", [False, True])
def test_rk4_range_fallback_preserves_signed_stage_cancellation(
    monkeypatch, vectorized
):
    graph = _graph()
    graph.nodes[0].update({EPI_PRIMARY: 0.0, ALIAS_DEPI[0]: 1e308})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    stages = (1e308, -1e308, -1e308, 1e308)
    actual, derivative, acceleration = integrators._apply_increments(
        graph, 1.0, {0: stages}, method="rk4"
    )[0]
    exact = (
        sum(
            weight * Fraction(stage)
            for weight, stage in zip((1, 2, 2, 1), stages, strict=True)
        )
        / 6
    )
    assert actual == float(exact)
    assert derivative == stages[-1]
    assert acceleration == 0.0


@pytest.mark.parametrize("vectorized", [False, True])
def test_rk4_rejects_nonrepresentable_final_response_without_writes(
    monkeypatch, vectorized
):
    graph = _graph()
    graph.nodes[0].update({EPI_PRIMARY: 0.0, DNFR_PRIMARY: 1e308, ALIAS_DEPI[0]: 1e308})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    before = copy.deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(NetworkConfigError, match="finite"):
        integrators.update_epi_via_nodal_equation(graph, dt=4.0, method="rk4")
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


def _integration_route(monkeypatch, route):
    graph = _graph(extended=route == "extended")
    if route == "scalar":
        monkeypatch.setattr(integrators, "np", None)
    return graph


# Configuration admission is shared and precedes backend dispatch. Exercise
# each invalid policy once, then retain consumer/frozen-row wiring sentinels.
@pytest.mark.parametrize(
    "route,capacity,configuration",
    [
        ("vectorized", 0.0, {"CLIP_MODE": "misspelled"}),
        ("vectorized", 0.0, {"EPI_MIN": math.nan}),
        ("vectorized", 0.0, {"EPI_MAX": math.inf}),
        ("vectorized", 0.0, {"EPI_MIN": 2.0, "EPI_MAX": 1.0}),
        ("vectorized", 0.0, {"EPI_MIN": "-1.0"}),
        ("vectorized", 0.0, {"CLIP_SOFT_K": 0.0}),
        ("vectorized", 0.0, {"CLIP_SOFT_K": True}),
        ("vectorized", 0.0, {"CLIP_SOFT_K": math.inf}),
        ("scalar", 0.0, {"CLIP_MODE": "misspelled"}),
        ("extended", 0.0, {"CLIP_MODE": "misspelled"}),
        ("vectorized", 1.0, {"CLIP_MODE": "misspelled"}),
    ],
)
def test_active_clip_policy_rejects_before_any_solver_evaluation(
    monkeypatch, route, capacity, configuration
):
    graph = _integration_route(monkeypatch, route)
    graph.graph.update(configuration)
    graph.nodes[0].update({VF_PRIMARY: capacity, DNFR_PRIMARY: 0.2})
    before = copy.deepcopy(graph)

    # Invalid clipping must not reach Gamma, including when the nodal row is
    # frozen. The extended route's field-cache writes are checked below too.
    def unexpected_gamma(*args, **kwargs):
        pytest.fail("Invalid clipping policy reached Gamma evaluation")

    monkeypatch.setattr(integrators, "eval_gamma", unexpected_gamma)
    monkeypatch.setattr(integrators, "eval_gamma_vectorized", unexpected_gamma)
    with pytest.raises(NetworkConfigError, match="clipping policy"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


# _node_state owns all seven boundaries; the additional consumers need to
# demonstrate delegation and pre-write rejection, not repeat its whole matrix.
@pytest.mark.parametrize(
    "route,aliases,value,error",
    [
        ("vectorized", ALIAS_VF, -1.0, FrequencyError),
        ("vectorized", ALIAS_VF, math.nan, FrequencyError),
        ("vectorized", ALIAS_VF, math.inf, FrequencyError),
        ("vectorized", ALIAS_DNFR, math.nan, NetworkConfigError),
        ("vectorized", ALIAS_DNFR, math.inf, NetworkConfigError),
        ("vectorized", ALIAS_DEPI, math.nan, NetworkConfigError),
        ("vectorized", ALIAS_DEPI, math.inf, NetworkConfigError),
        ("scalar", ALIAS_VF, -1.0, FrequencyError),
        ("extended", ALIAS_VF, -1.0, FrequencyError),
    ],
)
def test_invalid_authoritative_nodal_input_rejects_before_caches_or_writes(
    monkeypatch, route, aliases, value, error
):
    graph = _integration_route(monkeypatch, route)
    # A later valid alias must not mask the invalid authoritative value.
    graph.nodes[0][aliases[0]] = value
    graph.nodes[0][aliases[1]] = 0.5
    before = copy.deepcopy(graph)
    with pytest.raises(error):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert dict(graph.nodes[0]) == dict(before.nodes[0])
    assert graph.graph == before.graph


@pytest.mark.parametrize(
    "route,t0,dt,dt_min",
    [
        ("scalar", 1e20, 0.1, 0.0),
        ("extended", 1e20, 0.1, 0.0),
        ("vectorized", 1e20, 0.1, 0.0),
        ("vectorized", 1e308, 1e308, 0.0),
        # The full span advances, and the first substep advances; the second does not.
        ("vectorized", float(2**53 - 1), 2.0, 1.0),
    ],
)
def test_entire_represented_clock_grid_is_validated_before_evolution(
    monkeypatch, route, t0, dt, dt_min
):
    graph = _integration_route(monkeypatch, route)
    graph.graph.update(_t=t0, DT_MIN=dt_min)
    graph.nodes[0][DNFR_PRIMARY] = 0.25
    before = copy.deepcopy(graph)
    with pytest.raises(NetworkConfigError, match="clock"):
        integrators.update_epi_via_nodal_equation(graph, dt=dt)
    assert dict(graph.nodes[0]) == dict(before.nodes[0])
    assert graph.graph == before.graph


@pytest.mark.parametrize("route", ["scalar", "vectorized", "extended"])
def test_empty_graph_uses_same_repeated_add_clock_as_populated_graph(
    monkeypatch, route
):
    graph = _integration_route(monkeypatch, route)
    graph.graph.update(_t=0.1, DT_MIN=0.1)
    empty = nx.Graph()
    empty.graph.update(copy.deepcopy(graph.graph))
    for candidate in (graph, empty):
        integrators.update_epi_via_nodal_equation(candidate, dt=1.0)
    step, count, _, _ = integrators.prepare_integration_params(graph, dt=1.0, t=0.1)
    expected = 0.1
    for _ in range(count):
        expected += step
    assert expected != 0.1 + step * count
    assert graph.graph["_t"] == empty.graph["_t"] == expected


@pytest.mark.parametrize("route", ["scalar", "vectorized", "extended"])
def test_late_node_nonfinite_output_does_not_partially_write(monkeypatch, route):
    graph = _integration_route(monkeypatch, route)
    graph.nodes[0].update({EPI_PRIMARY: 0.0, DNFR_PRIMARY: 0.25})
    graph.add_node(1, **{EPI_PRIMARY: 0.0, VF_PRIMARY: 1e308, DNFR_PRIMARY: 1.0})
    before = copy.deepcopy(dict(graph.nodes(data=True)))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(NetworkConfigError, match="finite"):
            integrators.update_epi_via_nodal_equation(graph, dt=2.0)
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


@pytest.mark.parametrize("route", ["scalar", "vectorized", "extended"])
def test_finite_epi_does_not_admit_infinite_acceleration_metadata(monkeypatch, route):
    graph = _integration_route(monkeypatch, route)
    graph.nodes[0].update({EPI_PRIMARY: 0.0, DNFR_PRIMARY: 1.0})
    before = copy.deepcopy(dict(graph.nodes(data=True)))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(NetworkConfigError, match="finite"):
            integrators.update_epi_via_nodal_equation(graph, dt=1e-320)
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


@pytest.mark.parametrize("route", ["scalar", "vectorized", "extended"])
def test_later_substep_failure_restores_all_solver_owned_outputs(monkeypatch, route):
    graph = _integration_route(monkeypatch, route)
    graph.graph.update(DT_MIN=0.1, GAMMA={"type": "harmonic"})
    graph.nodes[0][DNFR_PRIMARY] = 0.2
    before = copy.deepcopy(dict(graph.nodes(data=True)))
    calls = []
    if route == "extended":
        from tnfr.dynamics import canonical

        original = canonical.compute_extended_nodal_system

        def extended_rates(**kwargs):
            calls.append(kwargs)
            result = original(**kwargs)
            return (
                result if len(calls) == 1 else result._replace(dnfr_derivative=math.inf)
            )

        monkeypatch.setattr(canonical, "compute_extended_nodal_system", extended_rates)
    else:

        def gamma(*args, **kwargs):
            calls.append(args)
            return 0.0 if len(calls) == 1 else math.inf

        monkeypatch.setattr(integrators, "eval_gamma", gamma)
        monkeypatch.setattr(integrators, "eval_gamma_vectorized", gamma)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(NetworkConfigError, match="finite"):
            integrators.update_epi_via_nodal_equation(graph, dt=0.2)
    assert len(calls) == 2
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


@pytest.mark.parametrize("route", ["scalar", "vectorized", "extended"])
def test_clipped_motion_remains_distinct_from_stored_model_derivative(
    monkeypatch, route
):
    graph = _integration_route(monkeypatch, route)
    graph.nodes[0].update({EPI_PRIMARY: 0.9, DNFR_PRIMARY: 0.5})
    integrators.update_epi_via_nodal_equation(graph, dt=0.5)
    assert graph.nodes[0][EPI_PRIMARY] == 1.0
    assert get_attr(graph.nodes[0], ALIAS_DEPI) == 0.5
    assert get_attr(graph.nodes[0], ALIAS_D2EPI) == 1.0
    assert (graph.nodes[0][EPI_PRIMARY] - 0.9) / 0.5 != 0.5
