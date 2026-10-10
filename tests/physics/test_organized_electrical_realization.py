"""Exact circuit-map obstruction and detached complete-row compatibility.

The formal positive energy map does not certify a passive electrical realization.
These static checks retain the full same-slot support and phase state; they run
neither a trajectory nor a frozen source producer or physical experiment.
"""

from fractions import Fraction as Q
from random import Random

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_product,
)
from tnfr.physics._organized_slot_registration import (
    _derive_organized_slot_registration,
    _observe_organized_slot_registration,
)


def _laplacian(size, edges):
    result = [[Q(0) for _ in range(size)] for _ in range(size)]
    for left, right, weight in edges:
        result[left][left] += weight
        result[right][right] += weight
        result[left][right] -= weight
        result[right][left] -= weight
    return tuple(tuple(row) for row in result)


def _projection(size):
    return tuple(
        tuple(Q(int(row == column)) - Q(1, size) for column in range(size))
        for row in range(size)
    )


def _pseudoinverse(laplacian):
    size = len(laplacian)
    shifted = tuple(tuple(value + Q(1, size) for value in row) for row in laplacian)
    inverse = exact_matrix_inverse(shifted)
    return tuple(tuple(value - Q(1, size) for value in row) for row in inverse)


def _action(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def cycle():
    laplacian = _laplacian(9, tuple((j, (j + 1) % 9, Q(1)) for j in range(9)))
    return laplacian, _pseudoinverse(laplacian)


@pytest.fixture(scope="module")
def joined():
    geometry = _derive_organized_slot_registration(slot_offsets=(0,) * 5)
    laplacian = _laplacian(54, geometry.weighted_edges)
    # The only 54-by-54 inverse is shared by every test in this module.
    return geometry, laplacian, _pseudoinverse(laplacian)


def test_cycle_required_capacitance_has_exact_forbidden_off_diagonal(cycle):
    laplacian, inverse = cycle
    assert exact_square_matrix_product(laplacian, inverse) == _projection(9)
    for row in range(9):
        assert sum(inverse[row]) == 0
        for column in range(9):
            distance = (column - row) % 9
            # Independent cycle resistance-distance formula, not inverse output.
            expected = Q(20, 27) - Q(distance * (9 - distance), 18)
            assert inverse[row][column] == expected
        assert inverse[row][(row + 1) % 9] == Q(8, 27) > 0


def test_actual_support_is_the_declared_cartesian_product(joined, cycle):
    geometry, laplacian, _ = joined
    cycle_laplacian, _ = cycle
    wheel_edges = tuple((c, (c + 1) % 5, Q(1)) for c in range(5)) + tuple(
        (c, 5, Q(1)) for c in range(5)
    )
    wheel = _laplacian(6, wheel_edges)
    assert len(geometry.weighted_edges) == 144
    assert geometry.support_degrees == (5,) * 45 + (7,) * 9
    for row in range(54):
        child, slot = divmod(row, 9)
        for column in range(54):
            other_child, other_slot = divmod(column, 9)
            expected = (
                cycle_laplacian[slot][other_slot] if child == other_child else Q(0)
            ) + (wheel[child][other_child] if slot == other_slot else Q(0))
            assert laplacian[row][column] == expected


def test_actual_neutral_inverse_is_symmetric_and_has_no_common_mode(joined):
    _, laplacian, inverse = joined
    assert exact_square_matrix_product(laplacian, inverse) == _projection(54)
    assert exact_square_matrix_product(inverse, laplacian) == _projection(54)
    assert all(sum(row) == 0 for row in inverse)
    assert all(inverse[a][b] == inverse[b][a] for a in range(54) for b in range(54))


def test_actual_support_inherits_positive_adjacent_slot_capacitance(joined, cycle):
    _, _, inverse = joined
    _, cycle_inverse = cycle
    for row in range(54):
        slot = row % 9
        for other_slot in range(9):
            # Sum over all six children, leaving the physical slot unchanged.
            values = tuple(inverse[row][9 * child + other_slot] for child in range(6))
            assert sum(values) == cycle_inverse[slot][other_slot]
            if other_slot == (slot + 1) % 9:
                assert sum(values) == Q(8, 27)
                assert max(values) > 0
                assert all(row != 9 * child + other_slot for child in range(6))


@pytest.mark.parametrize("size", (3, 9))
def test_complete_graph_is_an_admissible_passive_capacitance_control(size):
    laplacian = _laplacian(
        size,
        tuple((a, b, Q(1)) for a in range(size) for b in range(a + 1, size)),
    )
    inverse = _pseudoinverse(laplacian)
    projection = _projection(size)
    assert inverse == tuple(tuple(value / size for value in row) for row in projection)
    # A capacitor 1/n^2 between every pair realizes this neutral matrix.
    realized = _laplacian(
        size,
        tuple((a, b, Q(1, size**2)) for a in range(size) for b in range(a + 1, size)),
    )
    assert inverse == realized
    assert all(inverse[a][b] < 0 for a in range(size) for b in range(size) if a != b)


def test_eliminating_grounded_common_voltage_preserves_passive_sign():
    # Nonuniform ground capacitors plus positive pair capacitors: the exact
    # neutral Schur reduction cannot supply the forbidden positive offdiagonal.
    ground = (Q(1), Q(3), Q(2))
    edge_matrix = _laplacian(3, ((0, 1, Q(2)), (1, 2, Q(5))))
    capacitance = tuple(
        tuple(value + (ground[a] if a == b else 0) for b, value in enumerate(row))
        for a, row in enumerate(edge_matrix)
    )
    effective = tuple(
        tuple(
            value - ground[a] * ground[b] / sum(ground) for b, value in enumerate(row)
        )
        for a, row in enumerate(capacitance)
    )
    voltage = (Q(2, 7), Q(-5, 11), Q(13, 17))
    shift = -sum(a * b for a, b in zip(ground, voltage)) / sum(ground)
    charges = _action(capacitance, tuple(value + shift for value in voltage))
    assert sum(charges) == 0
    assert _action(effective, voltage) == charges
    assert all(sum(row) == 0 for row in effective)
    assert all(effective[a][b] <= 0 for a in range(3) for b in range(3) if a != b)


@pytest.fixture(scope="module")
def collective_wheel(joined):
    geometry, _, _ = joined
    weights = {}
    for left, right, weight in geometry.weighted_edges:
        child, slot = divmod(left, 9)
        other_child, other_slot = divmod(right, 9)
        if child == other_child:
            continue
        assert slot == other_slot
        pair = tuple(sorted((child, other_child)))
        weights[pair] = weights.get(pair, Q(0)) + weight / 9
    edges = tuple((left, right, weight) for (left, right), weight in weights.items())
    laplacian = _laplacian(6, edges)
    return edges, laplacian, _pseudoinverse(laplacian)


def test_collective_wheel_has_an_exact_passive_capacitance_realization(
    collective_wheel,
):
    edges, laplacian, inverse = collective_wheel
    assert len(edges) == 10
    assert all(weight == 1 for _, _, weight in edges)
    assert exact_square_matrix_product(laplacian, inverse) == _projection(6)
    for row in range(6):
        for column in range(6):
            if row == column:
                expected = Q(5, 36) if row == 5 else Q(103, 396)
            elif row == 5 or column == 5:
                expected = Q(-1, 36)
            elif (row - column) % 5 in (1, 4):
                expected = Q(-5, 396)
            else:
                expected = Q(-41, 396)
            assert inverse[row][column] == expected
    # Every positive pair component is derived from the exact required matrix,
    # including rim pairs absent from the original ten Josephson connections.
    capacitors = tuple(
        (a, b, -inverse[a][b]) for a in range(6) for b in range(a + 1, 6)
    )
    assert all(value > 0 for _, _, value in capacitors)
    assert _laplacian(6, capacitors) == inverse


@pytest.fixture(scope="module")
def collective_carrier(collective_wheel):
    edges, laplacian, _ = collective_wheel
    form = (Q(2, 7), Q(-5, 11), Q(1, 3), Q(4, 9), Q(-2, 5))
    form += (-sum(form),)
    offsets = (Q(1, 17), Q(-2, 13), Q(3, 19), Q(-5, 23), Q(2, 11), Q(-1, 7))
    state = _observe_organized_slot_registration(
        slot_offsets=(0,) * 5,
        relative_phase_origin=0,
        form=tuple(value for value in form for _ in range(9)),
        phase_offsets=tuple(value for value in offsets for _ in range(9)),
    ).interaction
    mp = mpmath.mp.clone()
    mp.dps = 180
    gamma = 1 / (1023 * mp.pi)
    gradient = _action(laplacian, form)
    current = [mp.mpf(0) for _ in range(6)]
    form_storage, phase_storage = Q(0), mp.mpf(0)
    for left, right, weight in edges:
        angle = _mp(mp, offsets[right] - offsets[left])
        transfer = _mp(mp, weight) * mp.sin(angle)
        current[left] += transfer
        current[right] -= transfer
        form_storage += weight * (form[left] - form[right]) ** 2 / 2
        phase_storage += _mp(mp, weight) * (1 - mp.cos(angle))
    return mp, state, gamma, gradient, current, form_storage, phase_storage


def test_exact_helical_carrier_closes_both_complete_collective_rows(collective_carrier):
    mp, state, gamma, gradient, current, _, _ = collective_carrier
    assert sum(state.form) == 0
    for child in range(6):
        form_rate = (-_mp(mp, gradient[child]) + gamma * current[child]) / 4
        phase_rate = gamma * _mp(mp, gradient[child]) / 4
        for slot in range(9):
            node = 9 * child + slot
            assert state.form_gradient[node] == gradient[child]
            _contains(mp, state.phase_current_bounds[node], current[child])
            _contains(mp, state.form_tau_rate_bounds[node], form_rate)
            _contains(mp, state.phase_tau_rate_bounds[node], phase_rate)
        _contains(mp, state.child_form_tau_rate_bounds[child], form_rate)
        _contains(mp, state.child_phase_tau_rate_bounds[child], phase_rate)


def test_exact_helical_carrier_retains_storage_constant_and_loss_scale(
    collective_carrier,
):
    mp, state, _, gradient, _, form_storage, phase_storage = collective_carrier
    internal_phase_storage = 54 * (1 - mp.cos(2 * mp.pi / 9))
    assert state.form_storage == 9 * form_storage
    _contains(
        mp, state.phase_storage_bounds, 9 * phase_storage + internal_phase_storage
    )
    _contains(
        mp,
        state.storage_bounds,
        9 * (_mp(mp, form_storage) + phase_storage) + internal_phase_storage,
    )
    coarse_loss = -sum(value**2 for value in gradient) / 4
    assert state.storage_tau_rate == 9 * coarse_loss
    _contains(mp, state.storage_rate_from_rows_bounds, 9 * _mp(mp, coarse_loss))


@pytest.fixture(scope="module")
def complete_state():
    generator = Random(731029)
    form = tuple(Q(generator.randrange(-37, 38), 101) for _ in range(54))
    offsets = tuple(Q(generator.randrange(-29, 30), 257) for _ in range(54))
    beta = Q(2, 5)
    state = _observe_organized_slot_registration(
        slot_offsets=(0,) * 5,
        relative_phase_origin=beta,
        form=form,
        phase_offsets=offsets,
    ).interaction
    context = mpmath.mp.clone()
    context.dps = 180
    return context, state, beta


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def test_full_state_formal_circuit_rows_and_charge_relation(joined, complete_state):
    geometry, laplacian, inverse = joined
    mp, state, beta = complete_state
    mean = sum(state.form) / 54
    charge = tuple(value - mean for value in state.form)
    gradient = _action(laplacian, state.form)
    assert gradient == state.form_gradient
    assert _action(inverse, gradient) == charge
    gamma = 1 / (1023 * mp.pi)
    phase = tuple(
        2 * mp.pi * (node % 9 - 4) / 9
        + _mp(mp, offset)
        + (_mp(mp, beta) if node >= 45 else 0)
        for node, offset in enumerate(state.phase_offsets)
    )
    voltage = tuple(gamma * _mp(mp, value) / 4 for value in gradient)
    incoming = [mp.mpf(0) for _ in range(54)]
    for left, right, weight in geometry.weighted_edges:
        current = gamma * _mp(mp, weight) * mp.sin(phase[right] - phase[left]) / 4
        incoming[left] += current
        incoming[right] -= current
    # Units are normalized a0=b=kappa=1. G=P/gamma is realized by positive
    # complete-pair resistors; C=(4/gamma)L+ remains only a formal matrix.
    conductance = tuple(
        tuple(_mp(mp, value) / gamma for value in row) for row in _projection(54)
    )
    dissipative_current = _action(conductance, voltage)
    for node in range(54):
        _contains(mp, state.phase_tau_rate_bounds[node], voltage[node])
        _contains(
            mp,
            state.form_tau_rate_bounds[node],
            incoming[node] - dissipative_current[node],
        )
    assert mp.almosteq(sum(incoming), 0, abs_eps=mp.mpf("1e-170"))
    assert mp.almosteq(sum(dissipative_current), 0, abs_eps=mp.mpf("1e-170"))


def test_formal_storage_and_joule_balance_match_complete_rows(joined, complete_state):
    geometry, _, inverse = joined
    mp, state, beta = complete_state
    gamma = 1 / (1023 * mp.pi)
    voltage = tuple(gamma * _mp(mp, value) / 4 for value in state.form_gradient)
    capacitance = tuple(
        tuple(4 * _mp(mp, value) / gamma for value in row) for row in inverse
    )
    charge = _action(capacitance, voltage)
    electric_energy = sum(v * q for v, q in zip(voltage, charge)) / 2
    josephson_energy = mp.mpf(0)
    for left, right, weight in geometry.weighted_edges:
        difference = (
            2 * mp.pi * (right % 9 - left % 9) / 9
            + _mp(mp, state.phase_offsets[right] - state.phase_offsets[left])
            + _mp(mp, beta) * (int(right >= 45) - int(left >= 45))
        )
        josephson_energy += gamma * _mp(mp, weight) * (1 - mp.cos(difference)) / 4
    _contains(
        mp, state.storage_bounds, 4 * (electric_energy + josephson_energy) / gamma
    )
    assert mp.almosteq(electric_energy, gamma * _mp(mp, state.form_storage) / 4)
    joule_loss = sum(value**2 for value in voltage) / gamma
    assert mp.almosteq(joule_loss, -gamma * _mp(mp, state.storage_tau_rate) / 4)
    _contains(mp, state.storage_rate_from_rows_bounds, -4 * joule_loss / gamma)
    # Positive energy is compatible with this obstruction: a passive pair
    # capacitance realization also needs every offdiagonal to be nonpositive.
    assert electric_energy > 0 and josephson_energy > 0
    assert any(inverse[a][b] > 0 for a in range(54) for b in range(54) if a != b)


def _child_means(values):
    return tuple(sum(values[9 * child : 9 * child + 9]) / 9 for child in range(6))


def test_off_carrier_mean_rows_retain_the_bounded_covariance_defect(
    collective_wheel, complete_state
):
    edges, laplacian, _ = collective_wheel
    mp, state, beta = complete_state
    gamma = 1 / (1023 * mp.pi)
    means = _child_means(state.form)
    offset_means = _child_means(state.phase_offsets)
    phases = offset_means[:5] + (offset_means[5] + beta,)
    departures = tuple(
        value - offset_means[node // 9]
        for node, value in enumerate(state.phase_offsets)
    )
    assert _child_means(departures) == (Q(0),) * 6
    gradient = _action(laplacian, means)
    coarse_current = [mp.mpf(0) for _ in range(6)]
    remainder = [mp.mpf(0) for _ in range(6)]
    defects = []
    for left, right, weight in edges:
        angle = _mp(mp, phases[right] - phases[left])
        edge_departures = tuple(
            departures[9 * right + slot] - departures[9 * left + slot]
            for slot in range(9)
        )
        assert sum(edge_departures) == 0
        actual_current = sum(mp.sin(angle + _mp(mp, z)) for z in edge_departures) / 9
        coarse = mp.sin(angle)
        defect = actual_current - coarse
        defects.append(defect)
        quadratic_bound = sum(z**2 for z in edge_departures) / 18
        assert abs(defect) <= _mp(mp, quadratic_bound)
        coarse_current[left] += _mp(mp, weight) * coarse
        coarse_current[right] -= _mp(mp, weight) * coarse
        remainder[left] += gamma * _mp(mp, weight) * defect / 4
        remainder[right] -= gamma * _mp(mp, weight) * defect / 4
    assert max(abs(value) for value in defects) > mp.mpf("1e-6")
    assert mp.almosteq(sum(remainder), 0, abs_eps=mp.mpf("1e-170"))
    mean_form_rate, mean_phase_rate = [], []
    for child in range(6):
        dx = (-_mp(mp, gradient[child]) + gamma * coarse_current[child]) / 4
        dx += remainder[child]
        phase_rate = gamma * _mp(mp, gradient[child]) / 4
        mean_form_rate.append(dx)
        mean_phase_rate.append(phase_rate)
        _contains(mp, state.child_form_tau_rate_bounds[child], dx)
        _contains(mp, state.child_phase_tau_rate_bounds[child], phase_rate)
    actual_mean_storage_rate = sum(
        _mp(mp, g) * dx - current * phase_rate
        for g, dx, current, phase_rate in zip(
            gradient, mean_form_rate, coarse_current, mean_phase_rate
        )
    )
    expected_mean_storage_rate = -sum(_mp(mp, value) ** 2 for value in gradient) / 4
    expected_mean_storage_rate += sum(
        _mp(mp, g) * r for g, r in zip(gradient, remainder)
    )
    assert mp.almosteq(actual_mean_storage_rate, expected_mean_storage_rate)


def test_off_carrier_signed_storage_and_orthogonal_loss_remainders(
    joined, collective_wheel, complete_state
):
    geometry, laplacian, _ = joined
    wheel_edges, wheel_laplacian, _ = collective_wheel
    mp, generic, _ = complete_state
    # A gentle centered donor deformation against nearly opposing origins
    # makes the actual phase remainder negative, despite positive total storage.
    beta = Q(3)
    donor_departure = tuple(Q(value, 100) for value in (0, 2, 3, 3, 1, -1, -3, -3, -2))
    z = (Q(0),) * 45 + donor_departure
    assert _child_means(z) == (Q(0),) * 6
    state = _observe_organized_slot_registration(
        slot_offsets=(0,) * 5,
        relative_phase_origin=beta,
        form=tuple(value / 1000 for value in generic.form),
        phase_offsets=z,
    ).interaction
    means = _child_means(state.form)
    u = tuple(value - means[node // 9] for node, value in enumerate(state.form))
    assert _child_means(u) == (Q(0),) * 6
    coarse_gradient = _action(wheel_laplacian, means)
    hidden_gradient = _action(laplacian, u)
    assert state.form_gradient == tuple(
        coarse_gradient[node // 9] + value for node, value in enumerate(hidden_gradient)
    )
    coarse_form_storage = sum(
        weight * (means[left] - means[right]) ** 2 / 2
        for left, right, weight in wheel_edges
    )
    hidden_form_storage = sum(a * b for a, b in zip(u, hidden_gradient)) / 2
    assert state.form_storage == 9 * coarse_form_storage + hidden_form_storage
    fine_phase_storage = mp.mpf(0)
    for left, right, weight in geometry.weighted_edges:
        angle = (
            2 * mp.pi * (right % 9 - left % 9) / 9
            + _mp(mp, z[right] - z[left])
            + _mp(mp, beta) * (int(right >= 45) - int(left >= 45))
        )
        fine_phase_storage += _mp(mp, weight) * (1 - mp.cos(angle))
    _contains(mp, state.phase_storage_bounds, fine_phase_storage)
    internal_storage = 54 * (1 - mp.cos(2 * mp.pi / 9))
    coarse_phase_storage = 5 * (1 - mp.cos(_mp(mp, beta)))
    phase_remainder = fine_phase_storage - internal_storage - 9 * coarse_phase_storage
    phase_budget = sum(a * b for a, b in zip(z, _action(laplacian, z))) / 2
    assert -_mp(mp, phase_budget) <= phase_remainder < 0
    total_remainder = _mp(mp, hidden_form_storage) + phase_remainder
    assert total_remainder < 0
    _contains(
        mp,
        state.storage_bounds,
        internal_storage
        + 9 * (_mp(mp, coarse_form_storage) + coarse_phase_storage)
        + total_remainder,
    )
    coarse_loss = 9 * sum(value**2 for value in coarse_gradient) / 4
    hidden_loss = sum(value**2 for value in hidden_gradient) / 4
    assert coarse_loss > 0 and hidden_loss > 0
    assert state.storage_tau_rate == -coarse_loss - hidden_loss
