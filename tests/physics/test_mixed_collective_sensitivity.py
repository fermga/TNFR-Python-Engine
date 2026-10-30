"""Signed collective sensitivity separates an envelope from accuracy claims.

Actual and predicted phases evolve independently. Static counterexamples
concern proof-method transfer, not acquired predictor failures.
"""

import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_collective_sensitivity as owner
from tnfr.physics import _mixed_effective_interaction as effective_owner
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**9)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
MATURE = Q(1, 2**28)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
QLO, QHI = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
G = Q(1, 3000)
SWITCH_AGE = Q(3375000)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    old = tuple(
        sorted(
            tuple(sorted((9 * p + j, 9 * p + (j + 1) % 9)))
            for p in (0, 1)
            for j in range(9)
        )
    )
    edges = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return old, tuple(sorted(edges - set(old)))


def _nominal_weights(mp):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if j < 9 else 1) * _mp(mp, DELTA) / 2
        for j, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _weights(mp, side):
    values = tuple(
        Q(mp.nstr(value, 210))
        + (1 if value < _mp(mp, Q(7, 8)) else -1)
        * WEIGHT_ERROR
        * Q(1 + (j + side) % 3, 6)
        for j, value in enumerate(_nominal_weights(mp))
    )
    assert all(Q(3, 4) <= value <= 1 for value in values)
    return values


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, weight) for (a, b), weight in zip(new, weights, strict=True)
    )


def _h_apply(mp, values, weights):
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            current = _mp(mp, weight) * (values[a] - values[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _sine(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            a, b = a + 18 * side, b + 18 * side
            current = mp.sin(theta[b] - theta[a])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _exchange(values, slots=18):
    first = tuple(values[j] - values[18 + j] if j < slots else 0 for j in range(18))
    return first + tuple(-value for value in first)


def _field(mp, forms, phases, weights):
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine, gx = _h_apply(mp, x, weights), _sine(mp, phases), _exchange(x)
    gamma = 1 / (1023 * mp.pi)
    cross = tuple(
        mp.sin(_mp(mp, phases[j + 18]) - _mp(mp, phases[j])) for j in range(18)
    )
    cross = cross + tuple(-value for value in cross)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    splus = tuple(a + 9 * b for a, b in zip(sine, cross, strict=True))
    return tuple((-a + gamma * b) / 9 for a, b in zip(lx, splus, strict=True)), tuple(
        gamma * a / 9 for a in lx
    )


def _state(forms, phases):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + I(value)
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )


def _new_field(weights):
    return phase_owner._mixed_phase_sensitive_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        registered_observation_error_bound=NOISE,
        aligned_observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_collective_sensitivity(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("collective sensitivity controls cannot replay or advance a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        # Permit the existing row builder for static full-state comparisons;
        # every integrator, acquisition producer and operator remains blocked.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220

    def forbidden(*args, **kwargs):
        pytest.fail(
            "static collective sensitivity proof cannot evaluate an exponential or root"
        )

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def _copy_center(values):
    return _center(values[:18]) + _center(values[18:])


def _profile(s, z):
    qd, qr = (s + z) / 2, (s - z) / 2
    return (-qd / 2,) * 9 + (qd / 2,) * 9 + (-qr / 2,) * 9 + (qr / 2,) * 9


def _offsets(y):
    return tuple(y[j + 18] - y[j] for j in range(18))


def _moment(mp, y, psi):
    return sum(mp.sin(psi + value) for value in _offsets(y)) - 18 * mp.sin(psi)


def _probe(mp, multiplier, modulus, scale):
    return tuple(
        _mp(mp, v)
        for v in _copy_center(
            tuple(
                Q((j * multiplier) % modulus - modulus // 2, scale) for j in range(36)
            )
        )
    )


def test_complete_and_effective_rows_give_signed_collective_errors(mp):
    raw_x = _copy_center(tuple(Q((j * 3) % 11 - 5, 10**8) for j in range(36)))
    common = (-Q(1, 10**8),) * 18 + (Q(1, 10**8),) * 18
    forms = tuple(a + b for a, b in zip(raw_x, common, strict=True))
    centered = _copy_center(tuple(Q((j * 5) % 7 - 3, 10**5) for j in range(36)))
    psi = mp.pi / 2 + _mp(mp, Q(7, 10000))
    predicted_psi = mp.pi / 2 - _mp(mp, Q(3, 10000))
    phases = tuple(_mp(mp, v) + (psi if j >= 18 else 0) for j, v in enumerate(centered))
    weights = (_weights(mp, 0), _weights(mp, 1))
    full = _new_field(weights)
    state = _state(forms, centered)
    state = state[:54] + tuple(
        v + pi_interval() / 2 + I(Q(7, 10000)) for v in state[54:]
    )
    actual = full(state)
    expected = _field(mp, forms, phases, weights)
    for interval, value in zip(actual, expected[0] + expected[1], strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    gamma = 1 / (1023 * mp.pi)
    aggregate = -sum(forms[:18])
    predicted_aggregate = aggregate + Q(1, 10**9)
    reduced = effective_owner._mixed_effective_interaction_field()(
        (I(predicted_aggregate), pi_interval() / 2 - I(Q(3, 10000)))
    )
    predicted_rate = (
        -2 * _mp(mp, predicted_aggregate) - 18 * gamma * mp.sin(predicted_psi),
        2 * gamma * _mp(mp, predicted_aggregate) / 9,
    )
    for interval, value in zip(reduced, predicted_rate, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    qdot = -sum(expected[0][:18])
    psidot = (sum(expected[1][18:]) - sum(expected[1][:18])) / 18
    residual = _moment(mp, tuple(_mp(mp, v) for v in centered), psi)
    assert abs(
        qdot + 2 * _mp(mp, aggregate) + 18 * gamma * mp.sin(psi) + gamma * residual
    ) < mp.mpf("1e-190")
    assert abs(psidot - 2 * gamma * _mp(mp, aggregate) / 9) < mp.mpf("1e-190")
    a, b = _mp(mp, aggregate - predicted_aggregate) / 9, psi - predicted_psi
    secant = (mp.sin(psi) - mp.sin(predicted_psi)) / b
    assert abs(
        (qdot - predicted_rate[0]) / 9
        - (-2 * a - 2 * gamma * secant * b - gamma * residual / 9)
    ) < mp.mpf("1e-190")
    assert abs(psidot - predicted_rate[1] - 2 * gamma * a) < mp.mpf("1e-190")
    # These are constructed full-row examples, not acquired source endpoints.
    # An actual phase is never inserted into the independent reduced field.


def test_quadratic_metric_and_lift_keep_the_moment_forcing_distinct(mp):
    gamma = 1 / (1023 * mp.pi)
    a, b, residual = (mp.mpf(v) for v in ("0.000001", "0.000002", "-0.00000003"))
    secant = mp.mpf("0.4")
    da, db = -2 * a - 2 * gamma * secant * b - gamma * residual / 9, 2 * gamma * a
    derivative = 2 * a * da + 2 * b * db + 2 * gamma * (da * b + a * db)
    homogeneous = -4 * (
        (1 - gamma**2) * a**2 + gamma * secant * a * b + gamma**2 * secant * b**2
    )
    forcing = -2 * gamma * (a + gamma * b) * residual / 9
    assert abs(derivative - homogeneous - forcing) < mp.mpf("1e-190")
    assert forcing != 0
    e, h = b, b + 2 * gamma * a
    de, dh = db, db + 2 * gamma * da
    assert abs(de - (h - e)) < mp.mpf("1e-190")
    assert abs(
        dh - ((1 - 4 * gamma**2 * secant) * e - h - 2 * gamma**2 * residual / 9)
    ) < mp.mpf("1e-190")
    # A negative collective secant can occur arbitrarily close to pi/2.
    # Positive centered internal stiffness does not change this calculation.
    small = _mp(mp, Q(1, 10**12))
    negative = (mp.sin(mp.pi / 2 + small) - 1) / small
    counter_derivative = -4 * gamma**2 * negative * small**2
    assert negative < 0 and counter_derivative > 0
    # This is a method-transfer counterexample, not an attained instability.


def test_individual_lift_rectangles_and_central_entry_preserve_both_phases(mp, report):
    gamma = 1 / (1023 * mp.pi)
    excess = _mp(mp, report.initial_lift_excess_upper_bound)
    r0 = _mp(mp, report.phase_moment_upper_bound)
    low, high = -mp.pi / 2, mp.pi / 2 + excess
    for upper in (high, mp.pi / 2):
        for u in (low, mp.mpf(0), upper):
            assert 1 - 4 * gamma**2 * mp.cos(u) > 0
            for residual in (-r0, r0):
                assert (
                    u - upper - 4 * gamma**2 * mp.sin(u) - 2 * gamma**2 * residual / 9
                    <= 0
                )
                assert (
                    u - low - 4 * gamma**2 * mp.sin(u) - 2 * gamma**2 * residual / 9
                    >= 0
                )
        for v in (low, upper):
            assert v - upper <= 0 and v - low >= 0
    delta = 6 * ETA + 2 * G * report.initial_aggregate_upper_bound / 9
    assert report.initial_lift_excess_upper_bound == delta
    assert report.lift_monotonicity_margin == 1 - 4 * G**2 > 0
    assert (
        report.first_unit_phase_displacement_upper_bound
        == 6 * ETA + 2 * G * report.aggregate_absolute_upper_bound / 9
        < Q(1, 4)
    )
    assert report.central_entry_margin == Q(1, 3300) ** 2 / 4 - delta > 0
    assert report.phase_moment_upper_bound <= Q(9, 2)
    # Exact cooperative comparison at t=1 has a formal d=1-exp(-2) in[0,1].
    # Its worst allowed value already proves both upper entry bounds.
    for decay_difference in (Q(0), Q(1, 2), Q(1)):
        first_drop = Q(1, 3300) ** 2 * (Q(1, 2) - decay_difference / 4)
        second_drop = Q(1, 3300) ** 2 * (Q(1, 2) + decay_difference / 4)
        assert first_drop >= Q(1, 3300) ** 2 / 4 > delta
        assert second_drop >= Q(1, 3300) ** 2 / 2 > delta
    assert report.individual_rectangles_certified and report.central_entry_certified


def test_metzler_gain_prices_only_the_initial_nonacute_unit(report):
    delta = report.initial_lift_excess_upper_bound
    for secant in (-delta, Q(0), Q(1, 2), Q(1)):
        matrix = ((-Q(1), Q(1)), (1 - 4 * G**2 * secant, -Q(1)))
        assert matrix[0][1] >= 0 and matrix[1][0] > 0
        logarithmic_norm = max(sum(row) for row in matrix)
        assert logarithmic_norm == max(Q(0), -4 * G**2 * secant)
        assert logarithmic_norm <= 4 * G**2 * delta
        if secant >= 0:
            assert logarithmic_norm == 0
    loss = 4 * G**2 * delta
    assert report.pre_entry_row_sum_upper_bound == loss
    assert report.sensitivity_gain_denominator == 1 - loss > 0
    assert report.sensitivity_gain_upper_bound == 1 / (1 - loss)
    initial = 6 * ETA + 2 * G * report.inherited_output_error_upper_bound / 9
    assert report.initial_lift_error_upper_bound == initial
    assert report.analytical_entry_time == 1
    assert report.signed_sensitivity_certified
    # The post-entry row-sum estimate is nonexpansive, not a positive
    # uniform contraction rate. It consumes no derivative of the secant.


def test_internal_moment_decomposition_retains_its_signed_quadratic_part(mp):
    y = _probe(mp, 5, 7, 10**5)
    zeta = _offsets(y)
    assert abs(sum(zeta)) < mp.mpf("1e-190")
    even = sum(1 - mp.cos(value) for value in zeta)
    odd = sum(mp.sin(value) - value for value in zeta)
    assert even >= 0 and even <= sum(value**2 for value in zeta) / 2
    assert abs(odd) <= sum(abs(value) ** 3 for value in zeta) / 6
    for psi in (mp.mpf(0), mp.pi / 3, mp.pi / 2):
        assert abs(
            _moment(mp, y, psi) + even * mp.sin(psi) - odd * mp.cos(psi)
        ) < mp.mpf("1e-190")
    z = _mp(mp, QHI / 2)
    reference = _profile(_mp(mp, QHI), z)
    offsets = _offsets(reference)
    assert all(abs(value - z / 2) < mp.mpf("1e-190") for value in offsets[:9])
    assert all(abs(value + z / 2) < mp.mpf("1e-190") for value in offsets[9:])
    odd_ref = sum(mp.sin(value) - value for value in offsets)
    assert abs(odd_ref) < mp.mpf("1e-190")
    for psi in (mp.pi / 3, mp.pi / 2):
        expected = -18 * mp.sin(psi) * (1 - mp.cos(z / 2))
        assert abs(_moment(mp, reference, psi) - expected) < mp.mpf("1e-190")
        assert expected < 0
    # Symmetric paired offsets retain signed quadratic information even
    # when the common-linear and cubic-odd contributions cancel exactly.


def test_residual_difference_cancels_the_common_linear_phase_error(mp):
    z = _mp(mp, QHI / 2)
    reference = _profile(_mp(mp, QHI), z)
    error = _probe(mp, 3, 11, 10**7)
    actual = tuple(a + b for a, b in zip(reference, error, strict=True))
    error_norm = mp.sqrt(_dot(error, error))
    for psi in (mp.mpf(0), mp.pi / 3, mp.pi / 2):
        defect = _moment(mp, actual, psi) - _moment(mp, reference, psi)
        allowance = 3 * z * error_norm + error_norm**2
        assert abs(defect) <= allowance
        assert abs(_moment(mp, actual, psi)) <= mp.mpf(9) * z**2 / 4 + allowance
    dz = _offsets(error)
    assert abs(sum(dz)) < mp.mpf("1e-190")
    assert mp.sqrt(_dot(dz, dz)) <= mp.sqrt(2) * error_norm
    # Same psi is used only to estimate R. Replacing actual psi by the
    # independently predicted phase would introduce an additional term.
    mismatched = _moment(mp, actual, mp.pi / 2) - _moment(mp, actual, mp.mpf(0))
    assert abs(mismatched) > 0


def test_reference_cone_yields_the_l2_budget_without_a_terminal_reading(mp, report):
    gamma = 1 / (1023 * mp.pi)
    k = _mp(mp, Q(13, 27)) - _mp(mp, DELTA) / (18 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    s, z = _mp(mp, QHI / 2), _mp(mp, QHI / 3)
    w, psi = gamma * z / 2, mp.pi / 3
    restoring = coefficient * mp.cos(s / 2) / 9 + mp.cos(psi)
    zdot = -4 * gamma * (k + 1) * w
    wdot = -2 * (k + 1) * w + 2 * gamma * restoring * mp.sin(z / 2)
    derivative_wz = wdot * z + w * zdot
    identity = derivative_wz + 2 * (k + 1) * w * z + 4 * gamma * (k + 1) * w**2
    assert abs(identity - 2 * gamma * restoring * z * mp.sin(z / 2)) < mp.mpf("1e-190")
    bmin = Q(2, 9) * (1 - QHI**2 / 8) - Q(1, 1000)
    sinc = 1 - QHI**2 / 24
    assert report.reference_restoring_lower_bound == bmin > 0
    assert report.reference_sinc_lower_bound == sinc > 0
    assert report.reference_phase_squared_integral_upper_bound == QHI**2 * (
        1 / (4 * Q(1, 3300) ** 2) + Q(1, 2)
    ) / (bmin * sinc)
    # Finite-terminal bounds retain w(T)z(T), then price it using w<=gamma*z.
    # The remaining endpoint coefficient is nonpositive; no endpoint is read.
    for z_terminal in (Q(0), QHI / 2, QHI):
        initial_term = (1 / (4 * G) + G / 2) * QHI**2
        terminal_term = (G / 2 - 1 / (4 * G)) * z_terminal**2
        assert terminal_term <= 0
        assert initial_term + terminal_term <= initial_term
    assert report.reference_l2_certified


def test_metric_convolution_prices_initial_error_and_forcing_once(report):
    prior = report.source
    root_budget = (
        prior.initial_metric_root_upper_bound + prior.forcing_integral_upper_bound
    )
    assert root_budget == report.metric_root_upper_bound
    rate = report.decay_rate
    expected = root_budget**2 / (4 * (1 - 2 * G) * rate)
    assert report.comparison_phase_squared_integral_upper_bound == expected
    # A formal pair of translated exponential kernels has L2 inner product
    # between0 and1/lambda. Nonnegative impulse masses give the same Young
    # bound without evaluating a kernel, response or exponential.
    first, second = (
        prior.initial_metric_root_upper_bound,
        prior.forcing_integral_upper_bound,
    )
    for overlap in (Q(0), Q(1, 2), Q(1)):
        square_integral = (first**2 + second**2 + 2 * overlap * first * second) / rate
        assert square_integral <= root_budget**2 / rate
    epsilon = Q(1, 100)
    iz = report.reference_phase_squared_integral_upper_bound
    ie = report.comparison_phase_squared_integral_upper_bound
    integral = (Q(9, 4) + 3 * epsilon / 2) * iz + (1 + 3 / (2 * epsilon)) * ie
    assert report.fixed_young_epsilon == epsilon
    assert report.moment_integral_upper_bound == integral
    # The algebraic Young remainder is a square; the direct residual
    # contributes only one additional integral of ||ey|| squared.
    z, error = Q(1, 1000), Q(1, 100000)
    young_gap = 3 * epsilon * z**2 / 2 + 3 * error**2 / (2 * epsilon) - 3 * z * error
    assert young_gap == Q(3, 2) * (epsilon * z - error) ** 2 / epsilon >= 0
    assert (
        report.old_moment_integral_upper_bound == report.phase_moment_upper_bound / rate
    )
    assert (
        report.moment_integral_improvement_margin
        == report.old_moment_integral_upper_bound - integral
    )
    assert report.error_l2_certified and report.residual_integral_certified


def test_uniform_accuracy_targets_and_observation_allowances_remain_distinct(report):
    phase = report.sensitivity_gain_upper_bound * (
        6 * ETA
        + 2 * G * report.inherited_output_error_upper_bound / 9
        + 2 * G**2 * report.moment_integral_upper_bound / 9
    )
    driven_form = 9 * G * phase + G * report.phase_moment_upper_bound / 2
    form = max(report.inherited_output_error_upper_bound, driven_form)
    assert report.uniform_phase_error_upper_bound == phase
    assert report.uniform_form_error_upper_bound == form
    assert report.intrinsic_form_tolerance == report.intrinsic_phase_tolerance == NOISE
    assert report.form_accuracy_margin == NOISE - form
    assert report.phase_accuracy_margin == NOISE - phase
    assert report.form_accuracy_certified is (form < NOISE)
    assert report.phase_accuracy_certified is (phase < NOISE)
    assert report.registered_form_record_error_upper_bound == form + NOISE
    assert report.aligned_form_record_error_upper_bound == form + NOISE
    assert report.uniform_envelope_certified and report.certified
    # A damped row stays within max(initial,forcing/decay), not their sum.
    for fraction in (Q(0), Q(1, 3), Q(1)):
        convex_bound = (
            fraction * report.inherited_output_error_upper_bound
            + (1 - fraction) * driven_form
        )
        assert convex_bound <= form
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        outside = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not outside.observation_allowances_admitted and not outside.certified
        assert outside.uniform_envelope_certified
        assert (
            outside.uniform_form_error_upper_bound == form
            and outside.uniform_phase_error_upper_bound == phase
        )
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert (
        exact.registered_form_record_error_upper_bound
        == exact.aligned_form_record_error_upper_bound
        == form
    )


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_original_sources_and_all_consumed_bounds_are_readmitted(monkeypatch, report):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "registration_covariance_certified", False),
        ((), "source_comparison_upper_bound", 200 * MATURE),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "centered_entry_norm_upper_bound", Q(0)),
        ((), "registered_observation_error_bound", NOISE),
        ((), "source_correlated_initialization_retained", False),
        ((), "whole_history_covariance_retained", False),
        ((), "original_attachment_state_retained", False),
        ((), "state_reset_performed", True),
        ((), "phase_clamped", True),
        ((), "common_storage_reference", "translated reference"),
        ((), "switch_time", Q(5385016)),
        ((), "clock", "changed actual clock"),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        ((), "prior_domain_certified", False),
        ((), "common_metric_certified", False),
        ((), "metric_lower_bound", Q(1)),
        ((), "shared_phase_scale", Q(1)),
        ((), "auxiliary_capacity", Q(1)),
        ((), "scaled_norm_squared_upper_bound", Q(0)),
        ((), "phase_sector_bound", I(0)),
        ((), "phase_barrier_margin", Q(0)),
        ((), "initial_full_storage_upper_bound", Q(0)),
        ((), "source_cone_certified", False),
        ((), "reference_cone_certified", False),
        ((), "reference_integrals_certified", False),
        ((), "reference_velocity_cone_margins", (Q(1), Q(1))),
        ((), "reference_relative_phase", "predicted psi instead of actual psi"),
        ((), "reference_sum_phase_equation", "s_prime=0"),
        ((), "reference_difference_form_equation", "r_prime=-r"),
        ((), "reference_is_autonomous_predictor", True),
        ((), "phase_readout_input_required", True),
        ((), "reference_velocity_reset", True),
        ((), "incremental_stiffness_certified", False),
        ((), "forced_comparison_certified", False),
        ((), "initial_metric_root_upper_bound", Q(0)),
        ((), "forcing_integral_upper_bound", Q(0)),
        ((), "metric_root_upper_bound", Q(0)),
        ((), "forcing_metric_rate", "one reciprocal forcing row omitted"),
        ((), "incremental_stiffness_replaced_by_radial_stiffness", True),
        ((), "mean_error_forcing_discarded", True),
        ((), "source_error_paid_once", False),
        (("source",), "joint_bootstrap_certified", False),
        (("source",), "weighted_gap_certified", False),
        (("source",), "matching_cosine_lower_bound", Q(0)),
        (("source",) * 2, "clock", "changed sustained clock"),
        (("source",) * 3, "effective_phase_equation", "psihat_prime=0"),
        (("source",) * 3, "pressure_selected_from_preparation_label", True),
        (("source",) * 4, "complete_precontact_history_translated", False),
        (("source",) * 4, "registration_turns", Q(0)),
        (("source",) * 5, "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 6, "clock", "changed renewed clock"),
        (("source",) * 7, "signed_kernel_certified", False),
        (("source",) * 7, "total_output_error_upper_bound", Q(0)),
        (("source",) * 8, "cut_form_error_upper_bound", Q(0)),
        (("source",) * 9, "uniform_comparison_upper_bound", 200 * MATURE),
        (("source",) * 9, "bregman_denominator", Q(1)),
        (("source",) * 9, "nominal_cone_certified", False),
        (("source",) * 9, "inherited_form_velocity_retained", False),
        (("source",) * 10, "actual_dissipation_certified", False),
        (("source",) * 10, "selected_mu", Q(2)),
        (("source",) * 11, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 11, "prefix_cut_duration", Q(10001)),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_marked_identity", lambda **kw: changed
        )
        rejected = _assess()
        assert not rejected.certified and not rejected.uniform_envelope_certified
        assert not rejected.all_postattachment_times_covered
        assert not rejected.form_accuracy_certified
        assert not rejected.phase_accuracy_certified
    path = ("source",) * 11 + ("handoff", "prefix", "reference")
    reference = source
    for name in path:
        reference = getattr(reference, name)
    for name, value in (
        ("phase_reset_error_bound", 4 * ETA),
        ("new_weight_range", (Q(1, 2), Q(1))),
    ):
        original = replace(reference.sources[0], **{name: value})
        changed = _replace_at(
            source,
            path,
            sources=(original, reference.sources[1]),
            donor=replace(reference.donor, source=original),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_marked_identity", lambda **kw: changed
        )
        assert not _assess().source_association_certified
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("recipient_isolated_deadline", Q(10**10 - 1)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("inherited_donor_velocity_retained", False),
        ("clock", "changed original clock"),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_marked_identity", lambda **kw: changed
        )
        assert not _assess().uniform_envelope_certified
    monkeypatch.setattr(owner, "_assess_mixed_marked_identity", lambda **kw: source)
    geometry = owner._derive_registered_phase_probe_geometry()
    for changes in (
        dict(bare_gap_lower_bound=Q(3)),
        dict(weighted_gap_lower_bound=Q(2)),
        dict(spectral_gap_certified=False),
    ):
        altered = replace(geometry, successor=replace(geometry.successor, **changes))
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().prior_domain_certified
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: geometry
    )
    valid = (source.metric_lower_bound, source.metric_upper_bound, source.decay_rate)
    for position in range(3):
        corrupted = tuple(
            Q(0) if j == position else value for j, value in enumerate(valid)
        )
        monkeypatch.setattr(
            owner, "_sine_common_quadratic_coefficients", lambda **kw: corrupted
        )
        rejected = _assess()
        assert not rejected.incremental_comparison_certified
        assert not rejected.uniform_envelope_certified
        assert rejected.decay_rate == report.decay_rate > 0


def test_old_identity_and_forecast_verdicts_do_not_supply_the_envelope(
    monkeypatch, report
):
    source = report.source
    changed = replace(
        source,
        certified=False,
        strict_marked_identity_certified=False,
        cycle_windings_certified=False,
        all_postattachment_times_covered=False,
        original_finite_identity_contract_extended=False,
        uniform_collective_prediction_certified=True,
        marked_gap_upper_bound=Q(0),
        registration_margin=Q(0),
        scaled_error_upper_bound=Q(0),
        observation_allowances_admitted=False,
        status="unused_identity_verdict",
        unresolved_conditions=("unused",),
        source=replace(
            source.source,
            certified=False,
            transverse_decay_certified=False,
            integrable_moment_certified=False,
            actual_asymptotic_attraction_certified=False,
            phase_moment_integral_upper_bound=Q(0),
            source=replace(
                source.source.source,
                certified=False,
                form_prediction_certified=False,
                phase_prediction_certified=False,
                phase_error_upper_bound=Q(0),
                uniform_form_error_upper_bound=Q(0),
            ),
        ),
    )
    monkeypatch.setattr(owner, "_assess_mixed_marked_identity", lambda **kw: changed)
    rebuilt = _assess()
    assert rebuilt.certified and rebuilt.uniform_envelope_certified
    assert (
        rebuilt.uniform_form_error_upper_bound == report.uniform_form_error_upper_bound
    )
    assert (
        rebuilt.uniform_phase_error_upper_bound
        == report.uniform_phase_error_upper_bound
    )
    assert rebuilt.form_accuracy_certified == report.form_accuracy_certified
    assert rebuilt.phase_accuracy_certified == report.phase_accuracy_certified
    assert (
        rebuilt.registered_storage_jump_bounds == report.registered_storage_jump_bounds
    )
    assert source == report.source


def test_raw_fixed_width_and_immutable_evidence_preserve_scope(
    monkeypatch, report, record_property
):
    def fixed(value):
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    assert (
        _assess(
            form_error_bounds=(fixed(ERROR) for _ in range(2)),
            phase_error_bounds=(fixed(ERROR) for _ in range(2)),
            registered_observation_error_bound=fixed(NOISE),
            aligned_observation_error_bound=fixed(NOISE),
        )
        == report
    )
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified
    assert report.status == "conditional_uniform_collective_envelope"
    assert report.initial_registrations_turns == (Q(0), Q(1, 4))
    assert report.switch_age == SWITCH_AGE and report.switch_time == 5385000
    assert report.original_clock_start_time * Q(1023, 1024) == report.switch_time
    assert report.previous_certified_duration == 4096
    assert report.analytical_entry_time == 1
    assert report.fixed_young_epsilon == Q(1, 100)
    assert report.full_state_dimension == 72 and report.reduced_state_dimension == 2
    assert report.physical_capacities == (Q(1),) * 36 and report.internal_divisor == 9
    assert report.reduced_state_coordinates == ("Q", "psi")
    assert report.collective_error_coordinates == ("(Q-Qhat)/9", "psi-psihat")
    assert report.sensitivity_coordinates == ("b", "b+2*gamma*a")
    assert report.sensitivity_forcing == "(0,-2*gamma**2*R/9)"
    assert report.newly_observed_aggregates == report.newly_observed_phases == 0
    assert report.original_uniform_prediction_tolerance_certified is (
        report.form_accuracy_certified and report.phase_accuracy_certified
    )
    n0 = 4 * QHI + 200 * MATURE
    work = 972 * ETA + Q(9, 2) * n0**3
    assert report.switching_work_certified
    assert report.phase_switching_work_allowance == work
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (n0 + 18 * ETA) ** 2)
    assert report.registered_storage_jump_bounds == (162 - work, 162 + work + 9 * n0**2)
    for name in (
        "all_postattachment_times_covered",
        "same_complete_and_effective_laws",
        "same_law_for_both_preparations",
        "original_source_error_paid_once",
        "source_correlated_initialization_retained",
        "whole_history_covariance_retained",
        "original_attachment_work_retained_once",
    ):
        assert getattr(report, name) is True
    for name in (
        "actual_phase_supplied_to_predictor",
        "analytical_reference_is_autonomous_predictor",
        "previous_finite_guarantees_replaced",
        "centered_stiffness_used_as_collective_stiffness",
        "uniform_positive_collective_cosine_assumed",
        "stiffness_derivative_consumed",
        "original_target_relaxed",
        "actual_prediction_failure_claimed",
        "actual_collective_instability_claimed",
        "exact_actual_two_coordinate_closure",
        "new_field_installed",
        "new_memory_state_installed",
        "state_reset_performed",
        "phase_clamped",
        "original_source_reacquired",
        "original_source_replayed",
        "source_trajectory_evaluated",
        "reduced_trajectory_evaluated",
        "new_endpoint_or_phase_record_claimed",
        "runtime_event_executed",
        "exponentials_evaluated",
        "physical_property_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    assert not hasattr(report, "post_duration") and not hasattr(
        report, "endpoint_form_error_upper_bound"
    )
    with pytest.raises(FrozenInstanceError):
        report.uniform_phase_error_upper_bound = Q(0)
    for name in (
        "source",
        "horizon",
        "actual_phase",
        "initial_phase",
        "phase_observation",
        "initial_reading",
        "threshold",
        "baseline",
        "port_strength",
        "young_epsilon",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
    # Properties come only from this module's one accepted report fixture.
    for name in (
        "switch_time",
        "analytical_entry_time",
        "previous_certified_duration",
        "source_comparison_upper_bound",
        "inherited_output_error_upper_bound",
        "scaled_norm_squared_upper_bound",
        "decay_rate",
        "metric_root_upper_bound",
        "initial_aggregate_upper_bound",
        "aggregate_absolute_upper_bound",
        "phase_moment_upper_bound",
        "old_moment_integral_upper_bound",
        "initial_lift_excess_upper_bound",
        "first_unit_phase_displacement_upper_bound",
        "central_entry_margin",
        "pre_entry_row_sum_upper_bound",
        "sensitivity_gain_denominator",
        "sensitivity_gain_upper_bound",
        "reference_restoring_lower_bound",
        "reference_sinc_lower_bound",
        "reference_phase_squared_integral_upper_bound",
        "comparison_phase_squared_integral_upper_bound",
        "moment_integral_upper_bound",
        "moment_integral_improvement_margin",
        "initial_lift_error_upper_bound",
        "uniform_phase_error_upper_bound",
        "uniform_form_error_upper_bound",
        "form_accuracy_margin",
        "phase_accuracy_margin",
        "registered_form_record_error_upper_bound",
        "aligned_form_record_error_upper_bound",
        "phase_switching_work_allowance",
    ):
        record_property(name, str(getattr(report, name)))
    for name in ("aligned_storage_jump_bounds", "registered_storage_jump_bounds"):
        record_property(
            name, json.dumps([str(value) for value in getattr(report, name)])
        )
    for name in (
        "uniform_envelope_certified",
        "form_accuracy_certified",
        "phase_accuracy_certified",
        "original_uniform_prediction_tolerance_certified",
    ):
        record_property(name, str(getattr(report, name)))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached retained source work")

    monkeypatch.setattr(owner, "_assess_mixed_marked_identity", forbidden)
    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
        -Q(1, 10**100),
    )
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in invalid:
            for side in (0, 1):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(
                        **{name: tuple(bad if j == side else Q(0) for j in range(2))}
                    )
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})
