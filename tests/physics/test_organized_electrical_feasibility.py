"""Independent algebra for the electrical protocol's apparatus requirements.

These controls use synthetic rational scales, matrix identities and polynomial
extrema. They evaluate no reserved response, instrument or physical source.
"""

import hashlib
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration


@pytest.fixture(scope="module", autouse=True)
def preserve_frozen_evidence_and_producers():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
            "docs/assets/organized_robust_current/attempt.json",
            "docs/assets/organized_robust_current/result.json",
            "benchmarks/organized_robust_current.py",
            "src/tnfr/physics/_organized_robust_current.py",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_sine_regeneration():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _laplacian_matrix(size, edges, weight=Q(1)):
    matrix = [[Q(0)] * size for _ in range(size)]
    for left, right in edges:
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(map(tuple, matrix))


def _action(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


@pytest.fixture(scope="module")
def c5_matrices():
    cycle = _laplacian_matrix(5, ((c, (c + 1) % 5) for c in range(5)))
    inverse = _laplacian_matrix(5, ((c, (c + 2) % 5) for c in range(5)), Q(1, 5))
    return cycle, inverse


def test_voltage_error_bound_is_sharp_and_not_only_common_mode(c5_matrices):
    cycle, inverse = c5_matrices
    # Build L+ from the physical pentagram; independently verify L L+=P.
    for column in range(5):
        values = tuple(inverse[row][column] for row in range(5))
        assert _action(cycle, values) == tuple(
            Q(int(row == column)) - Q(1, 5) for row in range(5)
        )
    assert _action(inverse, (Q(1),) * 5) == (Q(0),) * 5
    norm = max(sum(map(abs, row)) for row in inverse)
    witness = (Q(1), Q(0), Q(-1), Q(-1), Q(0))
    assert max(map(abs, witness)) == 1
    assert max(map(abs, _action(inverse, witness))) == norm == Q(4, 5)
    # A valid correlated differential error reaches the bound. Gauge removal
    # therefore cannot justify dropping it from a uniform raw-error budget.
    voltage_scale, voltage_error = Q(7, 13), Q(3, 17)
    converted = _action(
        inverse, tuple(voltage_error * value / voltage_scale for value in witness)
    )
    assert max(map(abs, converted)) == 4 * voltage_error / (5 * voltage_scale)


@pytest.mark.parametrize("clock_ratio", (Q(13, 12), Q(9, 10)))
def test_raw_noise_aperture_clock_tradeoff_keeps_actual_clock(clock_ratio):
    a00, gamma, raw_error, radius = Q(3, 7), Q(1, 37), Q(2, 19), Q(5, 23)
    invariant = 5 * a00 * gamma * raw_error * radius / (16 * clock_ratio)
    for c0, g0 in ((Q(5, 11), Q(7, 17)), (Q(13, 19), Q(2, 3))):
        b0 = 4 * g0 / c0
        v0 = a00 * gamma * g0 / c0
        # These are independently obtained from nominal voltage conversion
        # and ds=b0*D*dt, not two definitions of the same physical clock.
        voltage_allowance = raw_error * v0 / Q(4, 5)
        laboratory_aperture = radius / (b0 * clock_ratio)
        assert voltage_allowance * laboratory_aperture == invariant
        assert voltage_allowance * laboratory_aperture != invariant * clock_ratio

    sensor_error = Q(11, 31)
    b_min = 16 * sensor_error / (5 * a00 * gamma * raw_error)
    boundary_aperture = invariant / sensor_error
    for factor, admitted in ((Q(1, 2), True), (Q(1), True), (Q(2), False)):
        required_aperture = factor * boundary_aperture
        b_max = radius / (clock_ratio * required_aperture)
        assert (b_min <= b_max) is admitted
        assert (sensor_error * required_aperture <= invariant) is admitted


@pytest.mark.parametrize("cap,clock", ((Q(7, 6), Q(13, 12)), (Q(9, 8), Q(4, 5))))
def test_integrated_delivery_charge_cancels_clock_but_not_capacitance(cap, clock):
    a00, gamma, g0, c0 = Q(3, 7), Q(1, 37), Q(5, 11), Q(2, 13)
    b0, kappa0 = 4 * g0 / c0, a00 * gamma * g0 / 9
    # Discontinuous, neutral, nonconstant physical current errors in lab time.
    segments = (
        (Q(2, 17), (Q(3), Q(-3), Q(0), Q(0), Q(0))),
        (Q(5, 19), (Q(0), Q(2), Q(-1), Q(-1), Q(0))),
    )
    physical_charge = sum(
        duration * max(map(abs, current)) for duration, current in segments
    )
    structural_integral = Q(0)
    for duration, current in segments:
        assert sum(current) == 0
        d = tuple(value / (9 * b0 * kappa0 * cap * clock) for value in current)
        ds = b0 * clock * duration
        structural_integral += ds * max(map(abs, d))
    assert physical_charge == 9 * kappa0 * cap * structural_integral
    assert physical_charge == a00 * gamma * g0 * cap * structural_integral
    assert physical_charge != 9 * kappa0 * structural_integral

    plateau, tracking = Q(3, 29), Q(2, 101)
    current_scale = a00 * gamma**2 * g0**2 / c0
    allowed_charge = a00 * gamma * g0 * cap * tracking
    lab_duration = plateau / (b0 * clock)
    assert (allowed_charge / lab_duration) / current_scale == (
        4 * cap * clock * tracking / (gamma * plateau)
    )


@pytest.mark.parametrize("mu", (Q(1), Q(2)))
def test_physical_work_and_heat_use_actual_capacitance_prefactor(c5_matrices, mu):
    cycle, inverse = c5_matrices
    a00, gamma, g0, c0 = Q(3, 7), Q(1, 37), Q(5, 11), Q(2, 13)
    cap, conductance, phase, clock = Q(7, 6), Q(11, 10), Q(9, 8), Q(13, 12)
    b0, kappa0 = 4 * g0 / c0, a00 * gamma * g0 / 9
    v0, j0 = a00 * gamma * g0 / c0, a00 * gamma**2 * g0**2 / c0
    conversion = kappa0 * v0
    lam, rho = conductance / (cap * clock), 1 / (cap * clock)
    eta, beta = 1 / (phase * clock), phase * conductance**2 / (cap**2 * clock)
    means = tuple(Q(c - 2, 17) for c in range(5))
    hidden = tuple(Q((c + 1) * (j - 4), 101) for c in range(5) for j in range(9))
    state = tuple(means[c] + hidden[9 * c + j] for c in range(5) for j in range(9))
    fine_edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9)
    ) + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    fine_matrix = _laplacian_matrix(45, fine_edges)
    fine_lap = _action(fine_matrix, state)
    hidden_lap = _action(fine_matrix, hidden)
    mean_lap = _action(cycle, means)
    voltage = tuple(v0 * value for value in mean_lap)
    forcing = tuple(Q(2 - c, 19) for c in range(5))
    delivery = tuple(Q(c - 2, 103) for c in range(5))
    applied = tuple(rho * f + d for f, d in zip(forcing, delivery))
    current = tuple(9 * b0 * kappa0 * f for f in forcing)
    current_error = tuple(9 * b0 * kappa0 * cap * clock * d for d in delivery)
    actual_current = tuple(a + b for a, b in zip(current, current_error))
    fine_applied = tuple(value for value in applied for _ in range(9))
    structural_power = _dot(fine_lap, fine_applied)
    physical_work_density = _dot(voltage, actual_current) / (b0 * clock)
    assert physical_work_density == cap * conversion * structural_power != 0
    assert physical_work_density != conversion * structural_power
    physical_heat_density = g0 * conductance * _dot(voltage, voltage) / (b0 * clock)
    structural_collective_loss = (
        lam * (_dot(fine_lap, fine_lap) - _dot(hidden_lap, hidden_lap)) / 4
    )
    assert physical_heat_density == cap * conversion * structural_collective_loss
    assert physical_heat_density < cap * conversion * lam * _dot(fine_lap, fine_lap) / 4

    actual_charge = tuple(c0 * cap * value for value in _action(inverse, voltage))
    assert _dot(voltage, actual_charge) / 2 == (
        9 * cap * conversion * _dot(means, mean_lap) / 2
    )
    actual_a, actual_j = a00 * phase, mu * j0 * phase * conductance**2 / cap
    assert actual_a * actual_j == 9 * cap * conversion * mu * beta / eta


def test_ring_junction_capacitance_cannot_be_scalar_calibration(c5_matrices):
    cycle, inverse = c5_matrices
    nominal_capacitance, junction_capacitance = Q(7, 13), Q(1, 10**12)
    changed = tuple(
        tuple(
            nominal_capacitance * a + junction_capacitance * b
            for a, b in zip(row, edge_row)
        )
        for row, edge_row in zip(inverse, cycle)
    )
    # A positive ring branch introduces adjacent off-diagonal entries where
    # every scalar multiple of the pentagram reference is exactly zero.
    assert inverse[0][1] == 0
    assert changed[0][1] == -junction_capacitance != 0
    # This is also a neutral-subspace obstruction: all matrices annihilate
    # constants, so equal action there would imply equality everywhere.
    assert _action(changed, (Q(1),) * 5) == (Q(0),) * 5
    first = (Q(1), Q(-1), Q(0), Q(0), Q(0))
    second = (Q(1), Q(0), Q(-1), Q(0), Q(0))
    inferred_first = _action(changed, first)[0] / _action(inverse, first)[0]
    inferred_second = _action(changed, second)[0] / _action(inverse, second)[0]
    assert inferred_first != inferred_second


def _quadratic_extrema(coefficients):
    constant, linear, quadratic = coefficients
    points = [Q(0), Q(1)]
    if quadratic:
        vertex = -linear / (2 * quadratic)
        if 0 <= vertex <= 1:
            points.append(vertex)
    return tuple(constant + linear * r + quadratic * r**2 for r in points)


def test_cubic_command_peaks_and_plateau_leakage_ratio(c5_matrices):
    cycle, inverse = c5_matrices
    column = tuple(row[0] for row in inverse)
    contrast = _action(cycle, column)
    # Differentiate s=3r²-2r³, then form 4v*s''+k*s' exactly.
    input_polynomials = tuple(
        (24 * v, -48 * v + 6 * k, -6 * k) for v, k in zip(column, contrast)
    )
    input_peak = max(
        abs(value) for p in input_polynomials for value in _quadratic_extrema(p)
    )
    cubic_derivative_peak = max(_quadratic_extrema((Q(0), Q(6), Q(-6))))
    a00, g0, b0, gamma = Q(3, 7), Q(5, 11), Q(2, 13), Q(1, 37)
    largest_increment = max(map(abs, (Q(1, 10), Q(-1, 5), Q(1, 10))))
    command_current_peak = a00 * g0 * b0 * largest_increment * input_peak
    command_voltage_peak = (
        a00 * b0 * largest_increment * max(map(abs, contrast)) * cubic_derivative_peak
    )
    assert command_current_peak == Q(48, 25) * a00 * g0 * b0
    assert command_voltage_peak == Q(6, 25) * a00 * b0

    cap, clock, plateau, tracking = Q(7, 6), Q(13, 12), Q(3, 29), Q(2, 101)
    permitted_mean_leakage = (
        a00 * gamma * g0 * cap * tracking / (plateau / (b0 * clock))
    )
    assert command_current_peak / permitted_mean_leakage == (
        48 * plateau / (25 * gamma * cap * clock * tracking)
    )
