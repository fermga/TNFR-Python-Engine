"""Original signed source correlations retain every marked phase relation.

A same-phase driven centered reference is an analytical comparison device,
not a measured phase input or an autonomous collective predictor.
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
from tnfr.physics import _mixed_marked_identity as owner
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
    return owner._assess_mixed_marked_identity(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("marked-identity controls cannot replay or advance a source")

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
            "static marked-identity proof cannot evaluate an exponential or root"
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


def _nominal_weight_intervals():
    turns = _turns()
    pi = pi_interval()
    values = []
    for a, b in _support()[1]:
        wrapped_turn = (turns[b] - turns[a] + Q(1, 2)) % 1 - Q(1, 2)
        origin = (-DELTA / 2 if b < 9 else DELTA / 2) - (
            -DELTA / 2 if a < 9 else DELTA / 2
        )
        gap = 2 * pi * wrapped_turn + origin
        # These support angles are at most4pi/9; the original offsets do
        # not cross the principal branch. The pi-dependent weight is not
        # silently identified with its rational midpoint below.
        values.append(1 - abs(gap) / (2 * pi))
    return tuple(values)


def _profile(s, z, total_velocity, contrast_velocity):
    qd, qr = (s + z) / 2, (s - z) / 2
    vd, vr = (total_velocity + contrast_velocity) / 2, (
        total_velocity - contrast_velocity
    ) / 2
    forms = (-vd,) * 9 + (vd,) * 9 + (-vr,) * 9 + (vr,) * 9
    phases = (-qd / 2,) * 9 + (qd / 2,) * 9 + (-qr / 2,) * 9 + (qr / 2,) * 9
    return forms, phases


def _project_rows(forms, phases):
    vd = (sum(forms[9:18]) - sum(forms[:9])) / 18
    vr = (sum(forms[27:36]) - sum(forms[18:27])) / 18
    qd = (sum(phases[9:18]) - sum(phases[:9])) / 9
    qr = (sum(phases[27:36]) - sum(phases[18:27])) / 9
    return qd + qr, qd - qr, vd + vr, vd - vr


def _nominal_rows(mp, s, z, total_velocity, contrast_velocity, psi):
    gamma = 1 / (1023 * mp.pi)
    k = _mp(mp, Q(13, 27)) - _mp(mp, DELTA) / (18 * mp.pi)
    c = 1 + 2 * mp.cos(mp.pi / 9)
    return (
        4 * gamma * k * total_velocity,
        4 * gamma * (k + 1) * contrast_velocity,
        -2 * k * total_velocity - 2 * gamma * c * mp.sin(s / 2) * mp.cos(z / 2) / 9,
        -2 * (k + 1) * contrast_velocity
        - 2 * gamma * (c * mp.cos(s / 2) / 9 + mp.cos(psi)) * mp.sin(z / 2),
    )


def _average_cos(mp, origin, displacement):
    return (
        mp.cos(origin)
        if displacement == 0
        else (mp.sin(origin + displacement) - mp.sin(origin)) / displacement
    )


def _incremental_apply(mp, vector, first_phase, phase_difference, psi):
    out = [mp.mpf(0)] * 36
    theta = tuple(2 * mp.pi * _mp(mp, turn) for turn in _turns())
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            first, second = 18 * side + a, 18 * side + b
            origin = theta[b] - theta[a] + first_phase[second] - first_phase[first]
            displacement = phase_difference[second] - phase_difference[first]
            current = _average_cos(mp, origin, displacement) * (
                vector[first] - vector[second]
            )
            out[first] += current
            out[second] -= current
    for j in range(18):
        origin = psi + first_phase[j + 18] - first_phase[j]
        displacement = phase_difference[j + 18] - phase_difference[j]
        current = (
            9 * _average_cos(mp, origin, displacement) * (vector[j] - vector[j + 18])
        )
        out[j] += current
        out[j + 18] -= current
    return _copy_center(tuple(out))


def _gradient(mp, centered, psi):
    phases = tuple(v + (psi if j >= 18 else 0) for j, v in enumerate(centered))
    cross = tuple(mp.sin(phases[j + 18] - phases[j]) for j in range(18))
    cross += tuple(-v for v in cross)
    return _copy_center(
        tuple(-a - 9 * b for a, b in zip(_sine(mp, phases), cross, strict=True))
    )


def _probe(mp, multiplier, modulus, scale):
    return tuple(
        _mp(mp, v)
        for v in _copy_center(
            tuple(
                Q((j * multiplier) % modulus - modulus // 2, scale) for j in range(36)
            )
        )
    )


def test_nominal_centered_rows_match_production_with_priced_weight_rounding(mp):
    enclosure = _nominal_weight_intervals()
    midpoint = tuple(value.midpoint for value in enclosure)
    radius = max(value.radius for value in enclosure)
    assert all(Q(3, 4) <= value <= 1 for value in midpoint)
    assert radius > 0
    nominal = _nominal_weights(mp)
    for interval, expected in zip(enclosure, nominal, strict=True):
        assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
    forms, centered = _profile(QHI / 2, QHI / 3, -G * QHI / 8, -G * QHI / 12)
    x = tuple(_mp(mp, v) for v in forms)
    weights = (midpoint, midpoint)
    for offset in (Q(0), Q(1, 4)):
        psi = 2 * mp.pi * _mp(mp, offset) + mp.mpf("0.0007")
        phases = tuple(
            _mp(mp, v) + (psi if j >= 18 else 0) for j, v in enumerate(centered)
        )
        state = _state(forms, centered)
        state = state[:54] + tuple(
            v + 2 * pi_interval() * offset + I(Q(7, 10000)) for v in state[54:]
        )
        actual = _new_field(weights)(state)
        oracle = _field(mp, forms, phases, weights)
        for interval, value in zip(actual, oracle[0] + oracle[1], strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        exact_rows = _field(mp, forms, phases, (nominal, nominal))
        projected = _project_rows(*exact_rows)
        expected = _nominal_rows(
            mp,
            *(_mp(mp, v) for v in (QHI / 2, QHI / 3, -G * QHI / 8, -G * QHI / 12)),
            psi,
        )
        assert max(
            abs(a - b) for a, b in zip(projected, expected, strict=True)
        ) < mp.mpf("1e-190")
        # The official scalar-weight API uses a rational member. Its
        # difference from the analytic nominal law is explicitly priced.
        delta_x = tuple(a - b for a, b in zip(oracle[0], exact_rows[0], strict=True))
        delta_y = tuple(a - b for a, b in zip(oracle[1], exact_rows[1], strict=True))
        transport_allowance = _mp(mp, 14 * radius) * mp.sqrt(_dot(x, x)) / 9
        assert mp.sqrt(_dot(delta_x, delta_x)) <= transport_allowance
        assert mp.sqrt(_dot(delta_y, delta_y)) <= _mp(mp, G) * transport_allowance


def test_signed_cone_faces_retain_the_moving_phase_restoring_sign(mp, report):
    gamma = 1 / (1023 * mp.pi)
    k = _mp(mp, Q(13, 27)) - _mp(mp, DELTA) / (18 * mp.pi)
    c = 1 + 2 * mp.cos(mp.pi / 9)
    s, z = _mp(mp, QHI / 2), _mp(mp, QHI / 3)
    for psi in (mp.mpf(0), mp.pi / 2 + _mp(mp, Q(1, 1000))):
        upper = _nominal_rows(mp, s, z, 0, 0, psi)
        assert upper[2] < 0 and upper[3] < 0
        lower = _nominal_rows(mp, s, z, -gamma * s, -gamma * z, psi)
        assert lower[2] + gamma * lower[0] > 0
        assert lower[3] + gamma * lower[1] > 0
        assert c * mp.cos(s / 2) / 9 + mp.cos(psi) > _mp(mp, Q(1, 9) - Q(1, 1000))
        assert lower[0] < 0 and lower[1] < 0
    zero = _nominal_rows(mp, 0, 0, 0, 0, mp.pi / 2)
    assert zero == (0, 0, 0, 0)
    assert 4 * k > 1
    # Old correlated V>=-gamma*s/(4k),r>=-gamma*z lies in the new cone.
    assert -gamma * s < -gamma * s / (4 * k) < 0
    assert report.reference_restoring_lower_margin == Q(1, 9) - Q(1, 1000) > 0
    assert all(value > 0 for value in report.reference_velocity_cone_margins)
    assert report.reference_cone_certified


def test_correlated_profiles_bound_marked_gaps_and_reference_integrals(report):
    k = report.coupling_bounds.lo
    assert (
        report.reference_integral_coefficient_margin
        == 1 - (1 / k + 1 / (k + 1)) / 4
        > 0
    )
    assert 3 * report.reference_integral_coefficient_margin == 3 - Q(3, 4) * (
        1 / k + 1 / (k + 1)
    )
    for s, z in ((QHI, QHI), (QHI, Q(0)), (Q(0), QHI)):
        x, y = _profile(s, z, -G * s, -G * z)
        assert sum(x[:18]) == sum(x[18:]) == sum(y[:18]) == sum(y[18:]) == 0
        assert _dot(x, x) == 9 * G**2 * (s**2 + z**2) < (5 * G * QHI) ** 2
        assert _dot(y, y) == Q(9, 4) * (s**2 + z**2) < (Q(9, 4) * QHI) ** 2
        assert 0 <= (s + z) / 2 <= QHI and abs((s - z) / 2) <= QHI / 2
    assert report.reference_form_upper_bound == 5 * G * QHI
    assert report.reference_phase_upper_bound == Q(9, 4) * QHI < Q(1, 1000)
    assert report.reference_form_integral_upper_bound == 3 * QHI / Q(1, 3300)
    assert report.reference_phase_variation_upper_bound == 3 * QHI
    # For nonincreasing s,z, total variation bounds the profile path length:
    # (3/2)*integral(sqrt(sdot**2+zdot**2)) <=(3/2)*(s0+z0).
    assert Q(3, 2) * (QHI + QHI) == 3 * QHI
    assert report.reference_integrals_certified


def test_incremental_hessian_uses_the_same_phase_on_the_connecting_segment(mp, report):
    first = _probe(mp, 3, 7, 10**5)
    difference = _probe(mp, 5, 11, 10**6)
    second = tuple(a + b for a, b in zip(first, difference, strict=True))
    vector = _probe(mp, 7, 13, 10**5)
    psi = mp.pi / 2 + mp.mpf("0.0007")
    for y in (first, second):
        assert _dot(y, y) < _mp(mp, Q(1, 1000) ** 2)
    gradient_difference = tuple(
        a - b
        for a, b in zip(
            _gradient(mp, second, psi), _gradient(mp, first, psi), strict=True
        )
    )
    applied = _incremental_apply(mp, difference, first, difference, psi)
    assert max(
        abs(a - b) for a, b in zip(applied, gradient_difference, strict=True)
    ) < mp.mpf("1e-190")
    kv = _incremental_apply(mp, vector, first, difference, psi)
    weights = (_weights(mp, 0), _weights(mp, 1))
    lv = tuple(
        a + 9 * b
        for a, b in zip(_h_apply(mp, vector, weights), _exchange(vector), strict=True)
    )
    assert _dot(vector, kv) >= _mp(mp, Q(71, 200)) * _dot(vector, vector)
    assert _dot(vector, lv) / 500 <= _dot(vector, kv) / 4 <= _dot(vector, lv)
    # A radial Hessian from zero is a different segment and cannot replace
    # the actual-to-reference secant in the error equation.
    wrong = _incremental_apply(mp, difference, (mp.mpf(0),) * 36, second, psi)
    assert mp.sqrt(
        _dot(
            tuple(a - b for a, b in zip(applied, wrong, strict=True)),
            tuple(a - b for a, b in zip(applied, wrong, strict=True)),
        )
    ) > mp.mpf("1e-15")
    assert report.incremental_stiffness_lower_bound == Q(71, 200)
    assert report.incremental_stiffness_certified


def test_actual_error_rows_cancel_the_correlated_phase_forcing(mp, report):
    gamma = 1 / (1023 * mp.pi)
    nominal = _nominal_weights(mp)
    actual_weights = (_weights(mp, 0), _weights(mp, 1))
    xbar, ybar = _profile(
        *(_mp(mp, v) for v in (QHI / 2, QHI / 3, -G * QHI / 8, -G * QHI / 12))
    )
    ex, ey = _probe(mp, 3, 11, 10**8), _probe(mp, 5, 7, 10**8)
    actual_x = tuple(a + b for a, b in zip(xbar, ex, strict=True))
    actual_y = tuple(a + b for a, b in zip(ybar, ey, strict=True))
    psi = mp.pi / 2 + mp.mpf("0.0007")

    def phases(y):
        return tuple(v + (psi if j >= 18 else 0) for j, v in enumerate(y))

    actual = tuple(
        map(_copy_center, _field(mp, actual_x, phases(actual_y), actual_weights))
    )
    reference = tuple(
        map(_copy_center, _field(mp, xbar, phases(ybar), (nominal, nominal)))
    )
    de_x, de_y = (
        tuple(a - b for a, b in zip(left, right, strict=True))
        for left, right in zip(actual, reference, strict=True)
    )
    delta = tuple(
        a - b
        for a, b in zip(
            _h_apply(mp, xbar, actual_weights),
            _h_apply(mp, xbar, (nominal, nominal)),
            strict=True,
        )
    )
    le = tuple(
        a + 9 * b
        for a, b in zip(_h_apply(mp, ex, actual_weights), _exchange(ex), strict=True)
    )
    ke = _incremental_apply(mp, ey, ybar, ey, psi)
    expected_x = tuple(
        -(a + gamma * b + d) / 9 for a, b, d in zip(le, ke, delta, strict=True)
    )
    expected_y = tuple(gamma * (a + d) / 9 for a, d in zip(le, delta, strict=True))
    assert max(abs(a - b) for a, b in zip(de_x, expected_x, strict=True)) < mp.mpf(
        "1e-190"
    )
    assert max(abs(a - b) for a, b in zip(de_y, expected_y, strict=True)) < mp.mpf(
        "1e-190"
    )
    assert _dot(delta, delta) > 0
    forcing_x, forcing_y = tuple(-d / 9 for d in delta), tuple(
        gamma * d / 9 for d in delta
    )
    forcing = (
        2 * _dot(ex, forcing_x)
        + 8 * _dot(ey, forcing_y)
        + 8 * gamma * (_dot(forcing_x, ey) + _dot(ex, forcing_y))
    )
    exact = -2 * (1 - 4 * gamma**2) * _dot(ex, delta) / 9
    assert abs(forcing - exact) < mp.mpf("1e-190")
    omitted_phase = 2 * _dot(ex, forcing_x) + 8 * gamma * _dot(forcing_x, ey)
    assert abs(
        omitted_phase
        - forcing
        + 8 * _dot(ey, forcing_y)
        + 8 * gamma * _dot(ex, forcing_y)
    ) < mp.mpf("1e-190")
    phase_only = 8 * _dot(delta, forcing_y) + 8 * gamma * _dot(forcing_x, delta)
    assert abs(phase_only) < mp.mpf("1e-190")
    assert 8 * gamma * _dot(forcing_x, delta) < 0
    j = _dot(ex, ex) + 4 * _dot(ey, ey) + 8 * gamma * _dot(ex, ey)
    square = tuple(2 * b + 2 * gamma * a for a, b in zip(ex, ey, strict=True))
    assert abs(j - _dot(square, square) - (1 - 4 * gamma**2) * _dot(ex, ex)) < mp.mpf(
        "1e-190"
    )
    assert abs(forcing) / (2 * mp.sqrt(j)) <= mp.sqrt(_dot(delta, delta)) / 9
    zero_error_forcing_square = (
        _dot(forcing_x, forcing_x)
        + 4 * _dot(forcing_y, forcing_y)
        + 8 * gamma * _dot(forcing_x, forcing_y)
    )
    zero_error_identity = (1 - 4 * gamma**2) * _dot(delta, delta) / 81
    assert abs(zero_error_forcing_square - zero_error_identity) < mp.mpf("1e-190")
    assert 0 < zero_error_forcing_square <= _dot(delta, delta) / 81
    # This norm is the upper right derivative at zero error; it requires
    # neither division by sqrt(J=0) nor an observed nonzero error.
    assert report.weight_matrix_deviation_upper_bound == 14 * WEIGHT_ERROR
    assert report.forced_comparison_certified


def test_original_joint_error_initializes_the_scaled_metric_once(report):
    eb = (
        3
        * (24 * MATURE + 6 * 14 * WEIGHT_ERROR * 10 * QHI / Q(1, 3300))
        / (1 - 135 * QHI)
    )
    assert report.source_comparison_upper_bound == eb
    assert report.initial_scaled_error_upper_bound == 2 * eb
    ex, ey = 3 * eb / 5, 4 * eb / 5
    assert ex**2 + ey**2 == eb**2
    assert ex**2 + 4 * ey**2 <= 4 * eb**2
    assert ex**2 + 4 * ey**2 + 8 * G * ex * ey <= 4 * (1 + 2 * G) * eb**2
    assert 2 * eb > Q(6, 5) * eb
    initial = 2 * (1 + G) * eb
    forcing = 14 * WEIGHT_ERROR * QHI / (3 * Q(1, 3300))
    assert (1 + G) ** 2 >= 1 + 2 * G
    assert report.initial_metric_root_upper_bound == initial
    assert (
        report.forcing_integral_upper_bound
        == forcing
        == report.weight_matrix_deviation_upper_bound
        * report.reference_form_integral_upper_bound
        / 9
    )
    assert report.metric_root_upper_bound == initial + forcing
    assert report.scaled_error_upper_bound == (initial + forcing) / (1 - 2 * G)
    assert report.phase_error_upper_bound == report.scaled_error_upper_bound / 2
    assert (
        report.marked_edge_error_upper_bound == 3 * report.scaled_error_upper_bound / 4
    )
    # Scaling y by2 pays an old joint norm once; resetting to a new source
    # or dividing accumulated forcing by the slow rate is unnecessary.
    assert 1 - 2 * G < 1 and (1 - 2 * G) ** 2 <= 1 - 2 * G
    assert (
        report.metric_root_upper_bound
        < initial + report.weight_matrix_deviation_upper_bound / report.decay_rate
    )


def test_all_original_marked_pairs_keep_their_original_branch_collar(report):
    geometry = owner._derive_registered_phase_probe_geometry().successor
    expected_pairs = tuple(
        (a + 18 * side, b + 18 * side)
        for side in (0, 1)
        for a, b in geometry.marked_pairs
    )
    assert report.marked_pairs == expected_pairs and len(expected_pairs) == 18
    for s, z in ((QHI, QHI), (QHI, Q(0)), (Q(0), QHI)):
        _, centered = _profile(s, z, Q(0), Q(0))
        for side in (0, 1):
            for index, (a, b) in enumerate(geometry.marked_pairs):
                assert _turns()[b] - _turns()[a] == geometry.marked_branch_turns[index]
                expected = (s + z) / 2 if side == 0 else (s - z) / 2
                assert centered[b + 18 * side] - centered[a + 18 * side] == expected
    assert Q(2) < Q(3, 2) ** 2
    assert report.nominal_marked_gap_upper_bound == QHI
    assert report.marked_gap_upper_bound == QHI + report.marked_edge_error_upper_bound
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    assert report.strict_marked_identity_certified and report.cycle_windings_certified
    # Earlier Delta/3 witnesses belonged to the isotropic inference class;
    # they do not obey this tighter source-correlated reference tube.
    assert report.marked_gap_upper_bound < DELTA / 4 < DELTA / 3


def test_actual_phase_driver_is_a_proof_device_not_a_forecast(mp, report):
    state = tuple(_mp(mp, v) for v in (QHI / 2, QHI / 3, -G * QHI / 8, -G * QHI / 12))
    zero = _nominal_rows(mp, *state, mp.mpf(0))
    quarter = _nominal_rows(mp, *state, mp.pi / 2)
    assert zero[:3] == quarter[:3]
    expected = -2 * mp.sin(state[1] / 2) / (1023 * mp.pi)
    assert abs(zero[3] - quarter[3] - expected) < mp.mpf("1e-190")
    assert abs(zero[3] - quarter[3]) > mp.mpf("1e-9")
    # Identical four centered coordinates with different actual psi have
    # different rdot. These are algebraic examples, not attained sources.
    assert report.reference_cone_certified and report.prior_domain_certified


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_original_source_domain_and_cone_premises_are_readmitted(monkeypatch, report):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "registration_covariance_certified", False),
        ((), "source_comparison_upper_bound", 200 * MATURE),
        ((), "source_correlated_initialization_retained", False),
        ((), "whole_history_covariance_retained", False),
        ((), "original_attachment_state_retained", False),
        ((), "state_reset_performed", True),
        ((), "phase_clamped", True),
        ((), "common_storage_reference", "changed reference"),
        ((), "switch_time", Q(5385016)),
        ((), "clock", "wrong signed clock"),
        ((), "registered_observation_error_bound", NOISE),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        ((), "joint_bootstrap_certified", False),
        ((), "sharp_source_bound_certified", False),
        ((), "common_metric_certified", False),
        ((), "metric_lower_bound", Q(1)),
        ((), "shared_phase_scale", Q(1)),
        ((), "auxiliary_capacity", Q(1)),
        ((), "scaled_norm_squared_upper_bound", Q(0)),
        ((), "phase_sector_bound", I(0)),
        ((), "phase_barrier_margin", Q(0)),
        ((), "initial_full_storage_upper_bound", Q(0)),
        ((), "matching_cosine_lower_bound", Q(0)),
        ((), "stiffness_derivative_consumed", True),
        ((), "independently_fixed_copy_gauges_assumed", True),
        (("source",), "source_association_certified", False),
        (("source",) * 2, "source_correlated_initialization_retained", False),
        (("source",) * 3, "complete_precontact_history_translated", False),
        (("source",) * 3, "registration_turns", Q(0)),
        (("source",) * 4, "initial_relative_phase_mean_assumed_zero", True),
        (("source",) * 4, "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 5, "clock", "wrong renewed clock"),
        (("source",) * 6, "signed_kernel_certified", False),
        (("source",) * 7, "cut_form_error_upper_bound", Q(0)),
        (("source",) * 8, "uniform_comparison_upper_bound", 200 * MATURE),
        (("source",) * 8, "bregman_denominator", Q(1)),
        (("source",) * 8, "nominal_cone_certified", False),
        (("source",) * 8, "nominal_cone_margins", (Q(1),) * 3),
        (("source",) * 8, "nominal_contrast_form_cone_factor", Q(2)),
        (("source",) * 8, "initial_bregman_root_upper_bound", Q(0)),
        (("source",) * 8, "inherited_form_velocity_retained", False),
        (("source",) * 9, "actual_dissipation_certified", False),
        (("source",) * 10, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 10, "prefix_cut_duration", Q(10001)),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_signed_stability", lambda **kwargs: changed
        )
        rejected = _assess()
        assert not rejected.certified and not rejected.strict_marked_identity_certified
        assert not rejected.all_postattachment_times_covered
        assert not rejected.original_finite_identity_contract_extended
    path = ("source",) * 10 + ("handoff", "prefix", "reference")
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
            owner, "_assess_mixed_signed_stability", lambda **kwargs: changed
        )
        assert not _assess().certified
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("recipient_isolated_deadline", Q(10**10 - 1)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("inherited_donor_velocity_retained", False),
        ("clock", "wrong original clock"),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_signed_stability", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_signed_stability", lambda **kwargs: source
    )
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
    valid = (report.metric_lower_bound, report.metric_upper_bound, report.decay_rate)
    for position in range(3):
        corrupted = tuple(
            Q(0) if j == position else value for j, value in enumerate(valid)
        )
        monkeypatch.setattr(
            owner, "_sine_common_quadratic_coefficients", lambda **kwargs: corrupted
        )
        rejected = _assess()
        assert (
            not rejected.common_metric_certified
            and not rejected.strict_marked_identity_certified
        )
        assert rejected.decay_rate == report.decay_rate > 0


def test_old_identity_and_prediction_verdicts_do_not_supply_the_result(
    monkeypatch, report
):
    source = report.source
    changed = replace(
        source,
        certified=False,
        transverse_decay_certified=False,
        cycle_windings_certified=False,
        integrable_moment_certified=False,
        switching_work_certified=False,
        strict_identity_nonimplication_certified=False,
        collective_comparison_limitation_certified=False,
        observation_allowances_admitted=False,
        actual_asymptotic_attraction_certified=False,
        all_postattachment_times_covered=False,
        strict_marked_identity_certified=True,
        uniform_collective_prediction_certified=True,
        phase_moment_integral_upper_bound=Q(0),
        unresolved_exchange_integral_upper_bound=Q(0),
        marked_witness_phase=(Q(0),) * 36,
        marked_witness_gap=Q(0),
        status="unused_isotropic_verdict",
        unresolved_conditions=("unused",),
        source=replace(
            source.source,
            centered_identity_certified=False,
            form_prediction_certified=False,
            phase_prediction_certified=False,
        ),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_signed_stability", lambda **kwargs: changed
    )
    rebuilt = _assess()
    assert rebuilt.certified and rebuilt.strict_marked_identity_certified
    assert rebuilt.scaled_error_upper_bound == report.scaled_error_upper_bound
    assert rebuilt.registration_margin == report.registration_margin
    assert (
        rebuilt.registered_storage_jump_bounds == report.registered_storage_jump_bounds
    )
    assert not rebuilt.uniform_collective_prediction_certified
    assert source == report.source


def test_raw_source_and_reading_values_are_admitted_before_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached original source work")

    monkeypatch.setattr(owner, "_assess_mixed_signed_stability", forbidden)
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


def test_fixed_width_evidence_and_immutable_scope_keep_the_limits(
    report, record_property
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
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        outside = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not outside.observation_allowances_admitted and not outside.certified
        assert (
            outside.strict_marked_identity_certified
            and outside.all_postattachment_times_covered
        )
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert exact.certified and exact.registration_margin == report.registration_margin
    assert report.certified and report.status == "conditional_all_time_marked_identity"
    assert report.reference_coordinates == ("s", "z", "V", "r")
    assert (
        report.reference_relative_phase == "same actual continuous psi(t), analytically"
    )
    assert (
        report.reference_initialization
        == "original nominal centered source at A age3375000"
    )
    assert report.initial_registrations_turns == (Q(0), Q(1, 4))
    assert report.switch_age == SWITCH_AGE and report.switch_time == 5385000
    assert report.original_clock_start_time * Q(1023, 1024) == report.switch_time
    assert report.previous_certified_duration == 4096
    assert report.marked_pair_count == 18 and report.original_cycle_count == 4
    assert report.registration_gap_limit == DELTA / 4
    assert report.physical_capacities == (Q(1),) * 36 and report.internal_divisor == 9
    assert report.full_state_dimension == 72
    assert report.newly_observed_aggregates == report.newly_observed_phases == 0
    assert report.numerical_response_allowance == 0
    n0 = 4 * QHI + 200 * MATURE
    work = 972 * ETA + Q(9, 2) * n0**3
    assert report.phase_switching_work_allowance == work
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (n0 + 18 * ETA) ** 2)
    assert report.registered_storage_jump_bounds == (162 - work, 162 + work + 9 * n0**2)
    for name in (
        "all_postattachment_times_covered",
        "original_finite_identity_contract_extended",
        "source_error_paid_once",
        "source_correlated_initialization_retained",
        "whole_history_covariance_retained",
        "original_attachment_state_retained",
        "original_attachment_work_retained_once",
        "same_complete_and_effective_laws",
        "original_marked_branches_retained",
        "actual_phase_used_as_analytical_coefficient",
    ):
        assert getattr(report, name) is True
    for name in (
        "original_finite_prediction_contract_extended",
        "uniform_collective_prediction_certified",
        "reference_is_autonomous_predictor",
        "reference_is_complete_source_realization",
        "phase_readout_input_required",
        "new_field_installed",
        "independently_fixed_copy_gauges_assumed",
        "copy_means_assumed_conserved",
        "incremental_stiffness_replaced_by_radial_stiffness",
        "stiffness_derivative_consumed",
        "unequal_internal_blocks_assumed_commuting",
        "mean_error_forcing_discarded",
        "reference_velocity_reset",
        "prior_finite_endpoint_used_as_initial_state",
        "direct_entry_energy_adds_attachment_twice",
        "prior_isotropic_witness_invalidated",
        "phase_clamped",
        "state_reset_performed",
        "original_source_reacquired",
        "original_source_replayed",
        "original_source_trajectory_evaluated",
        "reference_trajectory_evaluated",
        "reduced_trajectory_evaluated",
        "new_endpoint_or_phase_record_claimed",
        "record_allowances_used_for_identity",
        "exact_actual_two_coordinate_closure",
        "intrinsic_pair_identity_claimed",
        "autonomous_pair_formation_claimed",
        "minimal_intrinsic_macrostate_claimed",
        "runtime_event_executed",
        "operator_occurrence_derived",
        "exponentials_evaluated",
        "spectral_roots_evaluated",
        "physical_property_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    assert not hasattr(report, "post_duration") and not hasattr(
        report, "endpoint_form_error_upper_bound"
    )
    with pytest.raises(FrozenInstanceError):
        report.marked_gap_upper_bound = Q(0)
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
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
    for name in (
        "switch_age",
        "switch_time",
        "previous_certified_duration",
        "source_comparison_upper_bound",
        "centered_entry_norm_upper_bound",
        "inherited_output_error_upper_bound",
        "initial_scaled_norm_squared_upper_bound",
        "scaled_norm_squared_upper_bound",
        "scaled_radius_margin",
        "initial_full_storage_upper_bound",
        "phase_boundary_storage_lower_bound",
        "phase_barrier_margin",
        "normalized_metric_lower_bound",
        "normalized_metric_upper_bound",
        "metric_lower_bound",
        "metric_upper_bound",
        "shared_metric_decay_rate",
        "decay_rate",
        "reference_restoring_lower_margin",
        "reference_form_upper_bound",
        "reference_phase_upper_bound",
        "reference_form_integral_upper_bound",
        "reference_phase_variation_upper_bound",
        "reference_integral_coefficient_margin",
        "incremental_stiffness_lower_bound",
        "weight_matrix_deviation_upper_bound",
        "initial_scaled_error_upper_bound",
        "initial_metric_root_upper_bound",
        "forcing_integral_upper_bound",
        "metric_root_upper_bound",
        "scaled_error_upper_bound",
        "phase_error_upper_bound",
        "marked_edge_error_upper_bound",
        "nominal_marked_gap_upper_bound",
        "marked_gap_upper_bound",
        "registration_margin",
        "phase_switching_work_allowance",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "reference_velocity_cone_margins",
        "aligned_storage_jump_bounds",
        "registered_storage_jump_bounds",
    ):
        record_property(name, json.dumps([str(v) for v in getattr(report, name)]))
