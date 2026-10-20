"""Static controls for conditional finite compression of constituent feedback.

These checks use supplied states and exact comparison inequalities. They do not
evaluate trajectories, restore formed sources or observe reserved responses.
"""

import hashlib
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics._organized_electrical_source import _transport_hidden_deviations

_CYCLE = tuple((c, (c + 1) % 5) for c in range(5))
_EDGES = tuple(
    (9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9)
) + tuple((9 * c + j, 9 * d + j) for c, d in _CYCLE for j in range(9))


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
            "docs/assets/organized_finite_window_design/method-sources-v1.json",
            "docs/assets/organized_recovery_observation/method-sources-v1.json",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_organized_response_evaluation():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


@pytest.fixture(scope="module")
def context():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _means(values):
    return tuple(sum(values[9 * c : 9 * c + 9]) / 9 for c in range(5))


def _hidden(values):
    means = _means(values)
    return tuple(value - means[i // 9] for i, value in enumerate(values))


def _norm_squared(values):
    return sum(value**2 for value in values)


def _interfaces(values):
    return tuple(
        tuple(values[9 * d + j] - values[9 * c + j] for j in range(9))
        for c, d in _CYCLE
    )


def _variances(values):
    return tuple(_norm_squared(edge) / 9 for edge in _interfaces(values))


def _source(context, phases, edges, weights=None):
    weights = (Q(1),) * len(edges) if weights is None else weights
    result = [context.mpf(0)] * len(phases)
    for (c, d), weight in zip(edges, weights):
        current = _mp(context, weight) * context.sin(phases[d] - phases[c])
        result[c] += current
        result[d] -= current
    return tuple(result)


def _close(context, left, right):
    assert abs(left - right) < context.mpf("1e-80")


def test_declared_budget_uses_shared_transport_and_improves_intrinsic_envelope():
    a, b, g, horizon, mu = Q(31, 10**9), Q(601, 10**10), Q(1, 3000), Q(16), Q(1, 10**17)
    transport = _transport_hidden_deviations(
        initial_form=a,
        initial_phase=b,
        gamma_upper=g,
        laplacian_upper=Q(8),
        duration=horizon,
    )
    pi = pi_interval()
    assert (1 / (1023 * pi)).hi < g
    assert Q(1500015, 100000) < horizon
    coupling = 2 * g * horizon
    form, phase = transport.form_upper, transport.phase_upper
    # Independently invert the two integral comparison inequalities.
    inverse = exact_matrix_inverse(((Q(1), -coupling), (-coupling, Q(1))))
    assert (form, phase) == tuple(row[0] * a + row[1] * b for row in inverse)
    assert all(entry >= 0 for row in inverse for entry in row)
    integral_drift = 2 * g * horizon * form
    heat_drift = g * a + 2 * g**2 * horizon * phase
    assert 0 < heat_drift < integral_drift
    drift = min(integral_drift, heat_drift)
    rho = g * drift * (phase + b) / 9 + 2 * g * phase**3 / 27 + g * mu / 4
    old_rho = g * phase**2 / 9
    inverse = exact_matrix_inverse(((Q(1), -g * horizon), (-g * horizon, Q(1))))
    error, phase_error = tuple(row[0] * horizon * rho for row in inverse)
    old_error = inverse[0][0] * horizon * old_rho
    assert error == horizon * rho / (1 - (g * horizon) ** 2)
    assert phase_error == g * horizon * error
    assert 0 < error < old_error / 100
    assert form < Q(32, 10**9)
    assert phase < Q(61, 10**9)
    assert drift < Q(11, 10**12)
    assert rho < Q(9, 10**22)
    assert error < Q(15, 10**21)
    assert phase_error < Q(8, 10**23)
    assert 2 * phase < pi.lo / 2
    mean_phase_transport = 2 * g * Q(1, 100) + 2 * g**2 * horizon
    assert pi.lo / 2 - Q(3, 2) - 2 * mean_phase_transport - 2 * phase_error > Q(7, 100)
    # A bound on initialization is a separate input, not absorbed into rho.
    initial_error = Q(1, 10**12)
    assert inverse[0][0] * initial_error > error


@pytest.mark.parametrize("scale", (Q(1, 17), Q(1, 10**9)))
def test_interface_variance_drift_follows_edge_cauchy_bounds(scale):
    before = _hidden(tuple(scale * ((3 * i) % 11 - 5) for i in range(45)))
    after = _hidden(tuple(scale * ((7 * i) % 13 - 6) for i in range(45)))
    delta = tuple(a - b for a, b in zip(after, before))
    total = tuple(a + b for a, b in zip(after, before))
    variance_drift = sum(
        abs(a - b) for a, b in zip(_variances(after), _variances(before))
    )
    edge_delta = sum(_norm_squared(edge) for edge in _interfaces(delta))
    edge_total = sum(_norm_squared(edge) for edge in _interfaces(total))
    assert variance_drift**2 <= edge_delta * edge_total / 81
    assert edge_delta <= 4 * _norm_squared(delta)
    assert edge_total <= 4 * _norm_squared(total)
    assert variance_drift**2 <= Q(16, 81) * _norm_squared(delta) * _norm_squared(total)


@pytest.mark.parametrize("drift_scale", (Q(0), Q(1, 10**12)))
def test_weighted_projection_and_static_uncertainty_remainder(context, drift_scale):
    phases = tuple(context.mpf(c - 2) / 7 for c in range(5))
    z0 = _hidden(tuple(Q((3 * i) % 7 - 3, 10**8) for i in range(45)))
    delta = _hidden(tuple(drift_scale * ((5 * i) % 11 - 5) for i in range(45)))
    z = tuple(a + b for a, b in zip(z0, delta))
    variance, mu = _variances(z0), Q(1, 10**17)
    mhat = tuple(value + mu * (i + 1) / 15 for i, value in enumerate(variance))
    assert sum(abs(a - b) for a, b in zip(mhat, variance)) == mu
    assert all(0 <= value <= 2 for value in mhat)
    weights = tuple(1 - value / 2 for value in mhat)
    assert all(0 <= value <= 1 for value in weights)
    gamma = 1 / (1023 * context.pi)
    carrier = _source(context, phases, _CYCLE)
    weighted = _source(context, phases, _CYCLE, weights)
    exact_quadratic = _source(
        context, phases, _CYCLE, tuple(1 - m / 2 for m in variance)
    )
    correction = [context.mpf(0)] * 5
    for (c, d), moment in zip(_CYCLE, mhat):
        value = gamma * context.sin(phases[d] - phases[c]) * _mp(context, moment) / 8
        correction[c] -= value
        correction[d] += value
    for c in range(5):
        _close(context, gamma * weighted[c] / 4, gamma * carrier[c] / 4 + correction[c])
    uncertainty = sum(
        abs(gamma * (a - b) / 4) for a, b in zip(weighted, exact_quadratic)
    )
    assert uncertainty <= gamma * _mp(context, mu) / 4
    theta = tuple(
        2 * context.pi * (j - 4) / 9 + phases[c] + _mp(context, z[9 * c + j])
        for c in range(5)
        for j in range(9)
    )
    projected = _means(_source(context, theta, _EDGES))
    remainder = sum(abs(gamma * (a - b) / 4) for a, b in zip(projected, weighted))
    norm = context.sqrt(_mp(context, _norm_squared(z)))
    initial_norm = context.sqrt(_mp(context, _norm_squared(z0)))
    drift = context.sqrt(_mp(context, _norm_squared(delta)))
    assert remainder <= (
        gamma * drift * (norm + initial_norm) / 9
        + 2 * gamma * norm**3 / 27
        + gamma * _mp(context, mu) / 4
    )


@pytest.mark.parametrize("moments", ((Q(0),) * 5, (Q(0), Q(1, 2), Q(1), Q(3, 2), Q(2))))
def test_compressed_storage_retains_actual_signed_input_work(context, moments):
    phases = tuple(context.mpf(c - 2) / 13 for c in range(5))
    form = tuple(context.mpf(value) / 19 for value in (2, -1, 3, -5, 1))
    forcing = tuple(context.mpf(value) / 17 for value in (1, -2, 3, -1, -1))
    weights = tuple(1 - value / 2 for value in moments)
    gamma = 1 / (1023 * context.pi)
    lap_form = [context.mpf(0)] * 5
    for c, d in _CYCLE:
        current = form[c] - form[d]
        lap_form[c] += current
        lap_form[d] -= current
    source = _source(context, phases, _CYCLE, weights)
    form_prime = tuple(
        (-q + gamma * s) / 4 + f for q, s, f in zip(lap_form, source, forcing)
    )
    phase_prime = tuple(gamma * q / 4 for q in lap_form)
    # Differentiate the edge-defined weighted storage, independently of its
    # compact cancellation formula. Boundary weight zero remains admitted.
    storage_rate = sum(
        (form[c] - form[d]) * (form_prime[c] - form_prime[d])
        + _mp(context, weight)
        * context.sin(phases[d] - phases[c])
        * (phase_prime[d] - phase_prime[c])
        for (c, d), weight in zip(_CYCLE, weights)
    )
    work = sum(q * f for q, f in zip(lap_form, forcing))
    _close(context, storage_rate, -sum(q**2 for q in lap_form) / 4 + work)
    assert abs(work) > context.mpf("1e-6")
    discrepancy = tuple(context.mpf(value) / 10**20 for value in (2, -1, 0, 1, -2))
    power_difference = 9 * sum(
        (discrepancy[c] - discrepancy[d]) * (forcing[c] - forcing[d]) for c, d in _CYCLE
    )
    allowance = (
        36
        * context.sqrt(sum(value**2 for value in discrepancy))
        * sum(abs(value) for value in forcing)
    )
    actual_power, predicted_power = 9 * work + power_difference, 9 * work
    assert abs(actual_power - predicted_power) <= allowance
    assert abs(abs(actual_power) - abs(predicted_power)) <= allowance
    assert abs(max(actual_power, 0) - max(predicted_power, 0)) <= allowance


def test_shared_norm_certificates_do_not_supply_initial_interface_statistics(context):
    epsilon, mu = Q(1, 2**26), Q(1, 10**17)
    zero = (Q(0),) * 45
    perturbed = (epsilon, -epsilon) + zero[2:]
    b = Q(601, 10**10)
    assert _means(zero) == _means(perturbed) == (Q(0),) * 5
    assert _norm_squared(zero) <= b**2
    assert _norm_squared(perturbed) <= b**2
    assert _variances(zero) == (Q(0),) * 5
    assert _variances(perturbed) == (
        2 * epsilon**2 / 9,
        Q(0),
        Q(0),
        Q(0),
        2 * epsilon**2 / 9,
    )
    # The two L1 uncertainty balls are disjoint: one statistic estimate cannot
    # serve both admitted initial points at the declared precision.
    assert (
        sum(abs(a - b) for a, b in zip(_variances(zero), _variances(perturbed)))
        > 2 * mu
    )
    assert max(abs(value) for value in perturbed) < Q(1, 10**6)
    phases = tuple(2 * context.pi * (c - 2) / 5 for c in range(5))
    weighted = []
    for z in (zero, perturbed):
        weights = tuple(1 - m / 2 for m in _variances(z))
        weighted.append(_source(context, phases, _CYCLE, weights))
    assert weighted[0][1] != weighted[1][1]
    pi = pi_interval()
    assert (2 * pi / 5).hi < Q(7, 5)
    assert (2 * pi / 9 + 2 * epsilon).hi < pi.lo
    assert sin(I(epsilon)).lo > 0
    # Both points satisfy the larger inference-class collars. Neither this
    # ambiguity nor the common norm certificate proves acquired-image access.
