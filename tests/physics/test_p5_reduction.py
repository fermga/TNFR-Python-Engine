"""Exact orbit closure and independent REMESH obstruction on fixed P5.

The checks derive the microscopic generator from path adjacency, compare
the existing graph quotient, and retain microscopic energy discarded by
the observer. The REMESH examples are exact real recurrence observations.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.p5_reduction import (
    P5ReducedState,
    P5ReductionGeometry,
    P5RemeshReduction,
    observe_p5_remesh_reduction,
    p5_reduction_geometry,
    reduce_p5_state,
)
from tnfr.physics.structural_morphism import certify_epi_coarse_graining


F = Fraction


def test_numpy_integral_states_use_unbounded_rational_components():
    values = np.array([2**62, -2**62, 3, -2**62, 2**62], dtype=np.int64)
    observed = reduce_p5_state(values)
    expected = reduce_p5_state([int(value) for value in values])
    assert observed == expected
    assert all(type(value.numerator) is int for value in observed.epi)
    assert p5_reduction_geometry(np.int64(3)) == p5_reduction_geometry(3)


def _matmul(left, right):
    return tuple(
        tuple(
            sum((a * b for a, b in zip(row, column)), F(0))
            for column in zip(*right)
        )
        for row in left
    )


def _matvec(matrix, vector):
    return tuple(
        sum((a * b for a, b in zip(row, vector)), F(0))
        for row in matrix
    )


def _diagonal(values):
    return tuple(
        tuple(value if i == j else F(0) for j in range(len(values)))
        for i, value in enumerate(values)
    )


def _energy(values, metric):
    mean = sum(w * x for w, x in zip(metric, values)) / sum(metric)
    return sum(w * (x - mean) ** 2 for w, x in zip(metric, values)) / 2


def _path_generator(capacity):
    neighbors = ((1,), (0, 2), (1, 3), (2, 4), (3,))
    return tuple(
        tuple(
            capacity * (F(i == j) - F(j in row, len(row)))
            for j in range(5)
        )
        for i, row in enumerate(neighbors)
    )


def _determinant3(matrix):
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def test_physics_facade_exposes_the_fixed_model_observer():
    import tnfr.physics as physics

    names = (
        "P5ReducedState", "P5ReductionGeometry", "P5RemeshReduction",
        "reduce_p5_state", "p5_reduction_geometry",
        "observe_p5_remesh_reduction",
    )
    assert set(names) <= set(physics.__all__)
    for name in names:
        assert getattr(physics, name) is globals()[name]


@pytest.mark.parametrize("capacity", [F(1), F(3, 2), F(1, 7)])
def test_exact_orbit_generator_is_derived_from_path_adjacency(capacity):
    geometry = p5_reduction_geometry(capacity=capacity)
    assert type(geometry) is P5ReductionGeometry
    assert geometry.capacity == capacity
    assert geometry.micro_generator == _path_generator(capacity)
    assert geometry.micro_metric == tuple(
        F(degree) / capacity for degree in (1, 2, 2, 2, 1)
    )
    assert _matmul(geometry.orbit_projection, geometry.orbit_lift) == (
        (1, 0, 0), (0, 1, 0), (0, 0, 1)
    )
    assert _matmul(geometry.orbit_projection, geometry.micro_generator) == (
        _matmul(geometry.orbit_generator, geometry.orbit_projection)
    )
    assert _matmul(geometry.micro_generator, geometry.orbit_lift) == (
        _matmul(geometry.orbit_lift, geometry.orbit_generator)
    )
    assert geometry.orbit_metric == tuple(
        F(degree) / capacity for degree in (2, 4, 2)
    )


def test_orbit_quotient_matches_existing_graph_coarse_graining():
    graph = nx.path_graph(5)
    for node in graph:
        graph.nodes[node].update(EPI=node - 2.0, nu_f=1.5, theta=0.0)
    existing = certify_epi_coarse_graining(graph, ((0, 4), (1, 3), (2,)))
    exact = p5_reduction_geometry(capacity=F(3, 2))

    assert existing.nodal_closure_within_tolerance
    for name, observed in (
        ("orbit_projection", existing.projection),
        ("orbit_lift", existing.lift),
        ("orbit_generator", existing.macro_generator),
        ("orbit_metric", existing.macro_metric_weights),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(exact, name), dtype=float), observed,
            rtol=1e-15, atol=0,
        )
    np.testing.assert_array_equal(
        existing.macro_conductance, ((0, 2, 0), (2, 0, 2), (0, 2, 0))
    )
    np.testing.assert_array_equal(existing.macro_frequency, (1.5, 1.5, 1.5))


def test_nested_visible_observer_loses_exactly_the_required_coordinate():
    geometry = p5_reduction_geometry()
    merge = ((1, 0, 0), (0, F(2, 3), F(1, 3)))
    transform = ((1, 0, 0), (0, F(2, 3), F(1, 3)), (0, -1, 1))
    assert _matmul(merge, geometry.orbit_projection) == (
        geometry.visible_projection
    )
    assert _matmul(transform, geometry.orbit_projection) == (
        geometry.memory_projection
    )
    assert geometry.memory_projection[:2] == geometry.visible_projection
    assert _matmul(geometry.memory_projection, geometry.memory_lift) == (
        (1, 0, 0), (0, 1, 0), (0, 0, 1)
    )
    assert _matmul(geometry.memory_projection, geometry.micro_generator) == (
        _matmul(geometry.memory_generator, geometry.memory_projection)
    )
    assert geometry.memory_generator == (
        (F(1), F(-1), F(1, 3)),
        (F(-1, 3), F(1, 3), F(-1, 9)),
        (F(1, 2), F(-1, 2), F(5, 3)),
    )


@pytest.mark.parametrize("capacity", [F(1), F(7, 3)])
def test_linear_minimality_has_a_nonzero_observability_minor(capacity):
    geometry = p5_reduction_geometry(capacity=capacity)
    visible = geometry.visible_projection
    derivative_row = _matvec(tuple(zip(*geometry.micro_generator)), visible[0])
    minor = tuple(row[:3] for row in (*visible, derivative_row))
    assert _determinant3(minor) == capacity / 12
    assert geometry.minimal_linear_dimension == 3


def test_both_metrics_are_induced_from_the_same_fine_nodal_metric():
    geometry = p5_reduction_geometry(capacity=F(5, 2))
    fine_metric = _diagonal(geometry.micro_metric)
    for lift, metric in (
        (geometry.orbit_lift, geometry.orbit_metric),
        (geometry.memory_lift, geometry.memory_metric),
    ):
        assert _matmul(tuple(zip(*lift)), _matmul(fine_metric, lift)) == (
            _diagonal(metric)
        )
    assert geometry.memory_metric == (F(4, 5), F(12, 5), F(8, 15))
    assert _matvec(geometry.memory_projection, (1,) * 5) == (1, 1, 0)
    assert _matvec(geometry.memory_generator, (1, 1, 0)) == (0, 0, 0)
    assert _matvec(geometry.memory_generator, (1, 1, 1)) != (0, 0, 0)


@pytest.mark.parametrize(
    "epi",
    [(3, -1, 2, 7, -5), (2, 1, 4, 1, 2), (4, 2, 3, 4, 2)],
)
def test_state_reconstruction_and_centered_energy_split(epi):
    state = reduce_p5_state(epi)
    geometry = p5_reduction_geometry(capacity=F(3, 2))
    assert type(state) is P5ReducedState
    assert all(type(value) is F for value in state.epi)
    lift = _matvec(geometry.orbit_lift, state.orbit_epi)
    assert tuple(a + b for a, b in zip(lift, state.discarded_epi)) == state.epi
    assert _matvec(geometry.memory_lift, state.memory_epi) == lift
    assert _matvec(geometry.orbit_projection, state.discarded_epi) == (0, 0, 0)
    assert sum(w * x for w, x in zip(geometry.micro_metric, state.discarded_epi)) == 0
    assert _energy(state.epi, geometry.micro_metric) == (
        _energy(state.orbit_epi, geometry.orbit_metric)
        + _energy(state.discarded_epi, geometry.micro_metric)
    )
    a, b, u = state.memory_epi
    assert state.conserved_mean == (a + 3 * b) / 4
    assert _energy(state.orbit_epi, geometry.orbit_metric) == (
        3 * (a - b) ** 2 / (4 * geometry.capacity)
        + 2 * u ** 2 / (3 * geometry.capacity)
    )


@pytest.mark.parametrize(
    "epi,coefficients,maximum",
    [
        ((1, 0, 0, 0, 1), (F(2, 3), F(1, 3)), F(1)),
        ((0, F(3, 2), -3, F(3, 2), 0), (F(2), F(-2)), F(1, 2)),
        ((5, 5, 5, 5, 5), (F(0), F(0)), F(0)),
    ],
)
def test_contrast_extrema_include_the_interior_vertex(epi, coefficients, maximum):
    state = reduce_p5_state(epi)
    assert state.contrast_coefficients == coefficients
    assert state.max_reference_contrast == maximum


def test_two_equal_macro_states_have_different_nodal_rates():
    geometry = p5_reduction_geometry()
    first = reduce_p5_state((0, 1, 1, 1, 0))
    second = reduce_p5_state((0, 0, 3, 0, 0))
    assert first.memory_epi[:2] == second.memory_epi[:2] == (0, 1)
    rates = []
    for state in (first, second):
        derivative = tuple(-x for x in _matvec(geometry.micro_generator, state.epi))
        rates.append(_matvec(geometry.visible_projection, derivative))
    assert rates == [(F(1), F(-1, 3)), (F(0), F(0))]


@pytest.mark.parametrize(
    "alpha,local,global_delay,history",
    [
        (F(1, 2), 1, 2, ((3, -1, 2, 7, -5), (1, 2, 3, 4, 5), (9, 0, 1, 2, 3))),
        (F(1, 3), 1, 1, ((3, 0, -1, 8, 1), (4, 2, 6, 0, -4))),
        (F(1), 7, 1, ((1, 3, 2, 5, 7), (9, 0, 1, 2, 3))),
    ],
)
def test_uniform_remesh_commutes_and_splits_augmented_history_energy(
    alpha, local, global_delay, history
):
    result = observe_p5_remesh_reduction(
        history, alpha=alpha, tau_local=local, tau_global=global_delay,
        capacity=F(3, 2),
    )
    assert type(result) is P5RemeshReduction
    assert result.certificate.stability_certificate_certified
    assert result.commutation_residual == (0, 0, 0)
    assert result.augmented_energy_split_residual == (0, 0, 0)
    assert result.projected_next == result.orbit.exact_next_field
    for part in (result.fine, result.orbit, result.discarded):
        assert part.transition_observation_certified
        assert part.exact_energy_drop >= 0
    beta, gamma, delta = (1 - alpha) ** 2, alpha * (1 - alpha), alpha
    expected = tuple(
        beta * F(history[0][i])
        + (gamma * F(history[local][i]) if gamma else 0)
        + delta * F(history[global_delay][i])
        for i in range(5)
    )
    assert result.fine.exact_next_field == expected
    for name in (
        "exact_augmented_energy_before", "exact_augmented_energy_after",
        "exact_energy_drop",
    ):
        assert getattr(result.fine, name) == (
            getattr(result.orbit, name) + getattr(result.discarded, name)
        )


def test_actual_past_of_a_decay_mode_cannot_supply_its_forward_flow():
    # These are exact samples at times 0, -log(2), -2*log(2).
    mode = (1, 0, -1, 0, 1)
    history = tuple(tuple(4 + scale * v for v in mode) for scale in (1, 2, 4))
    result = observe_p5_remesh_reduction(history, alpha=F(1, 2))
    assert _matvec(result.geometry.micro_generator, mode) == mode
    remesh_next = tuple(F(4) + F(11, 4) * v for v in mode)
    true_forward = tuple(F(4) + F(1, 2) * v for v in mode)
    assert result.fine.exact_next_field == remesh_next
    assert remesh_next != true_forward
    assert result.fine.exact_history_energies[0] == 2
    assert result.fine.exact_next_energy == F(121, 8)
    assert result.fine.exact_augmented_energy_before == F(32, 3)
    assert result.fine.exact_augmented_energy_after == F(55, 6)
    assert result.fine.exact_energy_drop == F(3, 2)
    assert result.discarded.exact_augmented_energy_before == 0


def test_orbit_consensus_does_not_certify_fine_consensus():
    hidden = (4, 2, 3, 4, 2)
    state = reduce_p5_state(hidden)
    assert state.orbit_epi == (3, 3, 3)
    result = observe_p5_remesh_reduction((hidden,) * 3, alpha=F(1, 2))
    assert result.orbit.exact_next_energy == 0
    assert result.discarded.exact_next_energy == 3
    assert result.fine.exact_next_energy == 3
    assert result.fine.exact_energy_drop == 0
    assert _matvec(result.geometry.micro_generator, hidden) != (0,) * 5


def test_exact_fractions_and_large_integers_are_not_rounded_to_binary64():
    value = F(10**400, 7)
    state = reduce_p5_state((value, 0, 0, 0, value))
    assert state.epi[0] == value
    assert state.memory_epi[0] == value
    assert state.max_reference_contrast == value
    floating = reduce_p5_state((0.1, 0, 0, 0, 0.1))
    assert floating.epi[0] == F.from_float(0.1)
    geometry = p5_reduction_geometry(capacity=value)
    assert geometry.capacity == value


def test_results_detach_caller_state_and_remain_immutable():
    initial = [1, 2, 3, 4, 5]
    state = reduce_p5_state(initial)
    initial[0] = 99
    assert state.epi == (1, 2, 3, 4, 5)
    with pytest.raises(FrozenInstanceError):
        state.conserved_mean = F(0)
    with pytest.raises(TypeError):
        state.epi[0] = F(0)
    history = [[1, 2, 3, 4, 5], [3, 0, 1, 2, 4], [2, 5, 1, 3, 0]]
    result = observe_p5_remesh_reduction(history, alpha=F(1, 2))
    original = result.fine.exact_history
    history[0][0] = 99
    assert result.fine.exact_history == original
    with pytest.raises(FrozenInstanceError):
        result.projected_next = ()


@pytest.mark.parametrize(
    "epi",
    [(), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6), "12345", {i: i for i in range(5)},
     (True, 0, 0, 0, 0), (1j, 0, 0, 0, 0), ("1", 0, 0, 0, 0),
     (float("nan"), 0, 0, 0, 0), (float("inf"), 0, 0, 0, 0)],
)
def test_invalid_initial_coordinates_are_rejected(epi):
    with pytest.raises((TypeError, ValueError)):
        reduce_p5_state(epi)


@pytest.mark.parametrize("capacity", [0, -1, True, "1", 1j, float("inf")])
def test_capacity_requires_a_positive_finite_real(capacity):
    with pytest.raises((TypeError, ValueError)):
        p5_reduction_geometry(capacity=capacity)


@pytest.mark.parametrize("alpha", [0, -1, F(4, 3), True, "0.5", float("nan")])
def test_remesh_requires_the_declared_positive_coefficient_class(alpha):
    with pytest.raises((TypeError, ValueError)):
        observe_p5_remesh_reduction(((0,) * 5,) * 3, alpha=alpha)


@pytest.mark.parametrize("delay", [0, -1, F(3, 2), True, "1"])
def test_remesh_requires_positive_integer_delays(delay):
    with pytest.raises(ValueError):
        observe_p5_remesh_reduction(
            ((0,) * 5,) * 3, alpha=F(1, 2), tau_local=delay
        )


@pytest.mark.parametrize(
    "history", [(), ((0,) * 5,) * 2, ((0,) * 5,) * 4, ((0,) * 4,) * 3]
)
def test_remesh_requires_exactly_the_active_newest_first_history(history):
    with pytest.raises(ValueError):
        observe_p5_remesh_reduction(history, alpha=F(1, 2))
