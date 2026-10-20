"""Independent algebra for uncertain five-child current discrimination.

These static controls do not evaluate the prospective uncertainty corner,
generate a trajectory, or establish a physical preparation or instrument.
"""

import hashlib
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration


@pytest.fixture(scope="module", autouse=True)
def preserve_frozen_observation_evidence():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_sine_regeneration():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _laplacian(values, edges):
    result = [0] * len(values)
    for left, right in edges:
        difference = values[left] - values[right]
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def _sine_current(context, phases, edges):
    result = [context.mpf(0)] * len(phases)
    for left, right in edges:
        current = context.sin(phases[right] - phases[left])
        result[left] += current
        result[right] -= current
    return tuple(result)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


@pytest.mark.parametrize("mu", (1, 2))
def test_both_physical_rows_use_one_nominal_voltage_state_and_input(mp, mu):
    # Deliberately nonidentity independent calibration and clock ratios.
    cap, conductance, phase_scale, clock = (Q(7, 6), Q(11, 10), Q(9, 8), Q(13, 12))
    c0, g0, a0 = Q(5, 7), Q(2, 3), Q(3, 5)
    gamma = 1 / (1023 * mp.pi)
    b0 = 4 * g0 / c0
    v0 = _mp(mp, a0 * g0 / c0) * gamma
    j0 = _mp(mp, a0 * g0**2 / c0) * gamma**2
    edges = tuple((c, (c + 1) % 5) for c in range(5))
    state = tuple(Q(c - 2, 7) for c in range(5))
    phases = tuple(_mp(mp, Q((c - 2) ** 2, 13)) for c in range(5))
    forcing = tuple(Q(2 - c, 19) for c in range(5))
    lap = _laplacian(state, edges)
    sine = _sine_current(mp, phases, edges)
    voltage = tuple(v0 * _mp(mp, value) for value in lap)
    # The same external current is used for both amplitude alternatives.
    supplied = tuple(_mp(mp, c0 * b0 * value) * v0 for value in forcing)
    actual_j = mu * j0 * _mp(mp, phase_scale * conductance**2 / cap)
    physical_charge_rate = tuple(
        -_mp(mp, g0 * conductance) * value + actual_j * current + supplied[c]
        for c, (value, current) in enumerate(zip(voltage, sine))
    )
    diffusion = conductance / (cap * clock)
    beta = phase_scale * conductance**2 / (cap**2 * clock)
    rho, eta = 1 / (cap * clock), 1 / (phase_scale * clock)
    expected_form = tuple(
        -_mp(mp, diffusion * value / 4)
        + mu * gamma * _mp(mp, beta / 4) * sine[c]
        + _mp(mp, rho * forcing[c])
        for c, value in enumerate(lap)
    )
    converted_form = tuple(
        value / (_mp(mp, cap * c0 * b0 * clock) * v0) for value in physical_charge_rate
    )
    converted_phase = tuple(
        value / _mp(mp, a0 * phase_scale * b0 * clock) for value in voltage
    )
    expected_phase = tuple(gamma * _mp(mp, eta * value / 4) for value in lap)
    tolerance = mp.mpf("1e-75")
    assert max(abs(a - b) for a, b in zip(converted_form, expected_form)) < tolerance
    assert max(abs(a - b) for a, b in zip(converted_phase, expected_phase)) < tolerance
    # Nominal charge and recorded time alter the measured current scale.
    apparent_current = actual_j / _mp(mp, cap * clock)
    assert abs(apparent_current - mu * j0 * _mp(mp, beta)) < tolerance
    assert abs(apparent_current - actual_j) > mp.mpf("1e-10")


@pytest.mark.parametrize("mu", (1, 2))
def test_own_law_storage_cancels_full_nonlinear_fine_current(mp, mu):
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9)
    ) + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    form = tuple(_mp(mp, Q((i % 7) - 3, 31)) for i in range(45))
    phase = tuple(_mp(mp, Q((i % 11) - 5, 17)) for i in range(45))
    lap, sine = _laplacian(form, edges), _sine_current(mp, phase, edges)
    lam, beta, eta, rho = map(mp.mpf, ("1.2", "0.8", "1.1", "0.9"))
    gamma = 1 / (1023 * mp.pi)
    external = tuple(_mp(mp, Q((i // 9) - 2, 23)) for i in range(45))
    tracking = tuple(_mp(mp, Q(2 - i // 9, 10**4)) for i in range(45))
    applied = tuple(rho * f + d for f, d in zip(external, tracking))
    form_rate = tuple(
        -lam * g / 4 + mu * gamma * beta * s / 4 + f
        for g, s, f in zip(lap, sine, applied)
    )
    phase_rate = tuple(gamma * eta * g / 4 for g in lap)
    potential_rate = sum(
        mp.sin(phase[right] - phase[left]) * (phase_rate[right] - phase_rate[left])
        for left, right in edges
    )
    storage_rate = _dot(lap, form_rate) + mu * beta / eta * potential_rate
    balance = -lam * _dot(lap, lap) / 4 + _dot(lap, applied)
    assert abs(storage_rate - balance) < mp.mpf("1e-75")
    wrong_storage_rate = _dot(lap, form_rate) + potential_rate
    assert abs(wrong_storage_rate - balance) > mp.mpf("1e-8")


def _polynomial_integral(coefficients, left, right):
    return sum(
        coefficient * (right ** (power + 1) - left ** (power + 1)) / (power + 1)
        for power, coefficient in enumerate(coefficients)
    )


def test_complete_cubic_word_gradient_integral_and_physical_input_cancel_gamma():
    gamma, amplitude = Q(1, 37), Q(2, 15)
    edges = tuple((c, (c + 1) % 5) for c in range(5))
    inverse_column = (Q(2, 5), Q(0), Q(-1, 5), Q(-1, 5), Q(0))
    contrast = _laplacian(inverse_column, edges)
    assert contrast == (Q(4, 5),) + (Q(-1, 5),) * 4
    # s'=6r-6r² is nonnegative, so the norm integral is exact.
    assert _polynomial_integral((Q(0), Q(6), Q(-6)), Q(0), Q(1)) == 1
    amplitudes = (Q(1, 10), Q(-1, 5), Q(1, 10)) * 5
    integral = sum(4 * abs(a) * max(map(abs, contrast)) / gamma for a in amplitudes)
    assert integral == Q(32, 5) / gamma
    a0, g0, b0, position = Q(3, 7), Q(5, 9), Q(2, 11), Q(2, 5)
    first, second = 6 * position * (1 - position), 6 - 12 * position
    kappa0 = a0 * gamma * g0 / 9
    for v, k in zip(inverse_column, contrast):
        structural_input = amplitude * (4 * v * second + k * first) / gamma
        physical_input = 9 * b0 * kappa0 * structural_input
        assert physical_input == a0 * g0 * b0 * amplitude * (4 * v * second + k * first)


@pytest.mark.parametrize(
    "pulses",
    (
        ((Q(0), Q(1, 7), Q(3)),),
        ((Q(4, 5), Q(1), Q(-2)),),
        ((Q(1, 9), Q(2, 9), Q(7)), (Q(3, 5), Q(7, 10), Q(-11))),
    ),
)
def test_l1_tracking_trapezoid_controls_allow_discontinuous_pulses(pulses):
    horizon, lam = Q(1), Q(7, 5)
    primitive_endpoint = sum(height * (right - left) for left, right, height in pulses)
    primitive_integral = sum(
        height * ((right - left) ** 2 / 2 + (horizon - right) * (right - left))
        for left, right, height in pulses
    )
    variation = sum(abs(height) * (right - left) for left, right, height in pulses)
    primitive_error = horizon * primitive_endpoint / 2 - primitive_integral
    direct_kernel = sum(
        height * _polynomial_integral((-horizon / 2, Q(1)), left, right)
        for left, right, height in pulses
    )
    assert primitive_error == direct_kernel
    assert abs(primitive_error) <= horizon * variation / 2
    # A scalar extremal component w''=-2*lambda*d controls the full-graph
    # Laplacian norm. Its curvature need not be pointwise bounded.
    endpoint = (
        -2
        * lam
        * sum(
            height * _polynomial_integral((horizon, Q(-1)), left, right)
            for left, right, height in pulses
        )
    )
    integral = -lam * sum(
        height * _polynomial_integral((horizon**2, -2 * horizon, Q(1)), left, right)
        for left, right, height in pulses
    )
    peano_error = horizon * endpoint / 2 - integral
    independent_kernel = -lam * sum(
        height * _polynomial_integral((Q(0), horizon, Q(-1)), left, right)
        for left, right, height in pulses
    )
    assert peano_error == independent_kernel
    assert abs(peano_error) <= lam * horizon**2 * variation / 4
    total = primitive_endpoint + primitive_error + peano_error
    assert abs(total) <= variation * (1 + horizon / 2 + lam * horizon**2 / 4)


def test_inward_positive_aperture_bound_does_not_cover_a_crossed_switch():
    radius, plateau_rate, ramp_rate = Q(1, 100), Q(2), Q(100)
    weights = (Q(1, 6), Q(1, 3), Q(1, 2))
    inward_offsets = (-radius, -radius / 2, Q(0))
    average = sum(w * plateau_rate * t for w, t in zip(weights, inward_offsets))
    assert sum(weights) == 1 and abs(average) <= plateau_rate * radius
    # State remains continuous, but a sample beyond the switch consumes the
    # large ramp derivative. Merely citing its timestamp radius is insufficient.
    crossed_sample = ramp_rate * radius / 2
    assert crossed_sample > plateau_rate * radius
    signed_average = 3 * (-plateau_rate * radius) - 2 * Q(0)
    assert abs(signed_average) > plateau_rate * radius
