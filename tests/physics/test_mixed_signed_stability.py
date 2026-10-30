"""Signed transverse decay has explicit source and identity boundaries.

Independent full-state and matrix oracles retain moving relative phase.
The marked-gap witness and positive error-comparison mode are larger-class
limitations, not acquired failures or numerical response evaluations.
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
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
from tnfr.physics import _mixed_signed_stability as owner
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
    return owner._assess_mixed_signed_stability(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("signed controls cannot replay or advance a source")

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
        pytest.fail("static signed proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def _copy_center(values):
    return _center(values[:18]) + _center(values[18:])


def _average_cos(mp, origin, displacement):
    if displacement == 0:
        return mp.cos(origin)
    return (mp.sin(origin + displacement) - mp.sin(origin)) / displacement


def _stiffness_apply(mp, vector, centered_phase, psi):
    out = [mp.mpf(0)] * 36
    theta = tuple(2 * mp.pi * _mp(mp, turn) for turn in _turns())
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            first, second = 18 * side + a, 18 * side + b
            coefficient = _average_cos(
                mp, theta[b] - theta[a], centered_phase[second] - centered_phase[first]
            )
            current = coefficient * (vector[first] - vector[second])
            out[first] += current
            out[second] -= current
    for j in range(18):
        coefficient = 9 * _average_cos(
            mp, psi, centered_phase[j + 18] - centered_phase[j]
        )
        current = coefficient * (vector[j] - vector[j + 18])
        out[j] += current
        out[j + 18] -= current
    return _copy_center(tuple(out))


def _probe_vectors(mp):
    x = tuple(
        _mp(mp, v)
        for v in _copy_center(tuple(Q((j * 3) % 11 - 5, 10**5) for j in range(36)))
    )
    y = tuple(
        _mp(mp, v)
        for v in _copy_center(tuple(Q((j * 5) % 7 - 3, 10**5) for j in range(36)))
    )
    return x, y


def test_actual_centered_restoring_operator_has_signed_sector_bounds(mp, report):
    x, y = _probe_vectors(mp)
    weights = (_weights(mp, 0), _weights(mp, 1))
    psi = mp.pi / 2 + mp.mpf("0.0007")
    assert _dot(x, x) + 4 * _dot(y, y) < _mp(mp, Q(1, 500) ** 2)
    ky = _stiffness_apply(mp, y, y, psi)
    phases = tuple(value + (psi if j >= 18 else 0) for j, value in enumerate(y))
    cross = tuple(mp.sin(phases[j + 18] - phases[j]) for j in range(18))
    cross = cross + tuple(-v for v in cross)
    gradient = _copy_center(
        tuple(-a - 9 * b for a, b in zip(_sine(mp, phases), cross, strict=True))
    )
    assert max(abs(a - b) for a, b in zip(ky, gradient, strict=True)) < mp.mpf("1e-190")
    coefficients = tuple(_average_cos(mp, psi, y[j + 18] - y[j]) for j in range(18))
    assert min(coefficients) < 0 and min(coefficients) > -_mp(mp, Q(1, 400))
    assert cos(4 * pi_interval() / 9 + I(Q(1, 500))).lo > Q(1, 10)
    kx = _stiffness_apply(mp, x, y, psi)
    lx = tuple(
        a + 9 * b for a, b in zip(_h_apply(mp, x, weights), _exchange(x), strict=True)
    )
    assert _dot(x, kx) >= _mp(mp, Q(71, 200)) * _dot(x, x)
    assert _dot(x, lx) / 500 <= _dot(x, kx) / 4 <= _dot(x, lx)
    assert abs(_dot(x, ky) - _dot(y, kx)) < mp.mpf("1e-190")
    # The radial average is essential; endpoint Hessian weights need not
    # reproduce the nonlinear gradient, even at fixed instantaneous psi.
    assert abs(
        _average_cos(mp, psi, mp.mpf("0.0001")) - mp.cos(psi + mp.mpf("0.0001"))
    ) > mp.mpf("1e-6")
    assert report.phase_stiffness_lower_bound == Q(71, 200)
    assert report.phase_stiffness_margin == Q(71, 200) - Q(1, 3) > 0
    assert report.metric_stiffness_margin == Q(1, 12) - Q(36, 500) > 0
    assert report.phase_stiffness_certified


def test_shared_metric_scaling_matches_the_actual_complete_rows(mp):
    x, y = _probe_vectors(mp)
    weights = (_weights(mp, 0), _weights(mp, 1))
    gamma = 1 / (1023 * mp.pi)
    psi = mp.pi / 2 + mp.mpf("0.0007")
    common = (-mp.mpf("0.0001"),) * 18 + (mp.mpf("0.0001"),) * 18
    forms = tuple(a + b for a, b in zip(x, common, strict=True))
    phases = tuple(v + (psi if j >= 18 else 0) for j, v in enumerate(y))
    dx, dy = _field(mp, forms, phases, weights)
    raw_x = _copy_center(tuple(Q((j * 3) % 11 - 5, 10**5) for j in range(36)))
    raw_y = _copy_center(tuple(Q((j * 5) % 7 - 3, 10**5) for j in range(36)))
    raw_common = (-Q(1, 10000),) * 18 + (Q(1, 10000),) * 18
    raw_forms = tuple(a + b for a, b in zip(raw_x, raw_common, strict=True))
    state = _state(raw_forms, raw_y)
    state = state[:54] + tuple(
        value + pi_interval() / 2 + I(Q(7, 10000)) for value in state[54:]
    )
    actual_rows = _new_field(weights)(state)
    assert len(actual_rows) == 72
    for interval, expected in zip(actual_rows, dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
    dx, dy = _copy_center(dx), _copy_center(dy)
    lx = tuple(
        a + 9 * b for a, b in zip(_h_apply(mp, x, weights), _exchange(x), strict=True)
    )
    ky = _stiffness_apply(mp, y, y, psi)
    mapped_x = tuple(-(a + gamma * b) / 9 for a, b in zip(lx, ky, strict=True))
    mapped_twice_y = tuple(2 * gamma * value / 9 for value in lx)
    assert max(abs(a - b) for a, b in zip(dx, mapped_x, strict=True)) < mp.mpf("1e-190")
    assert max(
        abs(2 * a - b) for a, b in zip(dy, mapped_twice_y, strict=True)
    ) < mp.mpf("1e-190")
    # H=Lplus,K=Kplus/4,a=x,b=2y,gamma'=2gamma,V=4I/9.
    k_b = tuple(value / 2 for value in ky)
    shared_x = tuple(
        -(_mp(mp, Q(4, 9)) * h + 2 * gamma * _mp(mp, Q(4, 9)) * k) / 4
        for h, k in zip(lx, k_b, strict=True)
    )
    assert max(abs(a - b) for a, b in zip(dx, shared_x, strict=True)) < mp.mpf("1e-190")
    complete_phase = _field(mp, forms, phases, weights)[1]
    assert abs(sum(complete_phase[18:]) - sum(complete_phase[:18])) > mp.mpf("1e-7")


def test_common_quadratic_identity_requires_no_stiffness_derivative(mp, report):
    x, y = _probe_vectors(mp)
    weights = (_weights(mp, 0), _weights(mp, 1))
    gamma = 1 / (1023 * mp.pi)
    for psi in (mp.mpf(0), mp.pi / 2 + mp.mpf("0.0007")):
        phases = tuple(v + (psi if j >= 18 else 0) for j, v in enumerate(y))
        dx, dy = map(_copy_center, _field(mp, x, phases, weights))
        lx = tuple(
            a + 9 * b
            for a, b in zip(_h_apply(mp, x, weights), _exchange(x), strict=True)
        )
        ky = _stiffness_apply(mp, y, y, psi)
        derivative = (
            2 * _dot(x, dx) + 8 * _dot(y, dy) + 8 * gamma * (_dot(dx, y) + _dot(x, dy))
        )
        identity = (
            -mp.mpf(2)
            * (
                (1 - 4 * gamma**2) * _dot(x, lx)
                + gamma * _dot(x, ky)
                + 4 * gamma**2 * _dot(y, ky)
            )
            / 9
        )
        assert abs(derivative - identity) < mp.mpf("1e-190")
        metric = _dot(x, x) + 4 * _dot(y, y) + 8 * gamma * _dot(x, y)
        assert derivative <= -_mp(mp, report.decay_rate) * metric
    lower, upper, rate = owner._sine_common_quadratic_coefficients(
        gamma_lower=Q(2, 3300),
        gamma_upper=2 * G,
        cosine_lower=Q(1, 500),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    assert (
        report.metric_lower_bound,
        report.metric_upper_bound,
        report.decay_rate,
    ) == (lower, upper, rate)
    assert lower == Q(9, 4) * (1 - 2 * G) and upper == Q(9, 4) * (1 + 2 * G)
    assert rate == Q(1, 3300) ** 2 / (375 * (1 + 2 * G)) > 0
    assert report.common_metric_certified


def test_original_A_source_gives_separate_tight_centered_bounds(report):
    qhi = QHI
    for s, z in ((qhi, qhi), (qhi, Q(0)), (Q(0), qhi)):
        total_velocity, contrast_velocity = -Q(9, 16) * G * s, -G * z
        vd, vr = (total_velocity + contrast_velocity) / 2, (
            total_velocity - contrast_velocity
        ) / 2
        qd, qr = (s + z) / 2, (s - z) / 2
        x = (-vd,) * 9 + (vd,) * 9 + (-vr,) * 9 + (vr,) * 9
        y = (-qd / 2,) * 9 + (qd / 2,) * 9 + (-qr / 2,) * 9 + (qr / 2,) * 9
        assert (
            _dot(x, x)
            == 9 * (total_velocity**2 + contrast_velocity**2)
            < (4 * G * qhi) ** 2
        )
        assert (
            _dot(y, y)
            == Q(9, 4) * (s**2 + z**2)
            <= Q(9, 2) * qhi**2
            < (Q(9, 4) * qhi) ** 2
        )
    # Cone corners are algebraic bounds, not independently acquired states.
    eb = (
        3
        * (24 * MATURE + 6 * 14 * WEIGHT_ERROR * 10 * qhi / Q(1, 3300))
        / (1 - 135 * qhi)
    )
    assert report.source_comparison_upper_bound == eb < 200 * MATURE
    assert report.nominal_centered_form_upper_bound == 4 * G * qhi
    assert report.nominal_centered_phase_upper_bound == Q(9, 4) * qhi
    assert report.initial_centered_form_upper_bound == 4 * G * qhi + eb
    assert report.initial_centered_phase_upper_bound == Q(9, 4) * qhi + eb
    assert report.initial_relative_phase_error_upper_bound == 6 * ETA
    assert report.sharp_source_bound_certified


def test_source_metric_tube_closes_the_broad_centered_chart(report):
    x0, y0 = (
        report.initial_centered_form_upper_bound,
        report.initial_centered_phase_upper_bound,
    )
    z0_square = x0**2 + 4 * y0**2
    zbar_square = (1 + 2 * G) * z0_square / (1 - 2 * G)
    assert report.initial_scaled_norm_squared_upper_bound == z0_square
    assert report.scaled_norm_squared_upper_bound == zbar_square
    assert report.normalized_metric_lower_bound == 1 - 2 * G
    assert report.normalized_metric_upper_bound == 1 + 2 * G
    assert report.scaled_radius_margin == Q(1, 500) ** 2 - zbar_square > 0
    assert z0_square - 4 * x0 * y0 == (x0 - 2 * y0) ** 2 >= 0
    assert z0_square + 8 * G * x0 * y0 <= (1 + 2 * G) * z0_square
    assert report.joint_bootstrap_certified and report.transverse_decay_certified
    assert report.cycle_windings_certified


def test_full_storage_barrier_confines_relative_phase_without_clamping(mp, report):
    n0 = 4 * QHI + 200 * MATURE
    b0 = report.slow_amplitude_bounds[1] + report.inherited_output_error_upper_bound
    initial = 162 + b0**2 + 972 * ETA + 18 * n0**2
    boundary = 162 + Q(81, 1000) - 18 * Q(1, 500) ** 2
    assert report.centered_entry_norm_upper_bound == n0
    assert report.entry_aggregate_upper_bound == b0
    assert report.initial_full_storage_upper_bound == initial
    assert report.phase_boundary_storage_lower_bound == boundary
    assert report.phase_barrier_margin == boundary - initial > 0
    delta = _mp(mp, Q(1, 1000))
    assert mp.sin(delta) >= delta / 2
    assert 162 * (1 - mp.cos(mp.pi / 2 + delta)) - 18 * _mp(mp, Q(1, 500)) ** 2 >= _mp(
        mp, boundary
    )
    assert report.initial_relative_phase_error_upper_bound < Q(1, 1000)
    work = 972 * ETA + Q(9, 2) * n0**3
    assert report.phase_switching_work_allowance == work
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (n0 + 18 * ETA) ** 2)
    assert report.registered_storage_jump_bounds == (162 - work, 162 + work + 9 * n0**2)
    # Entry energy is direct postattachment energy. Adding the already paid
    # quarter-turn jump again would be a different, double-charged ledger.
    assert initial < boundary < initial + report.registered_storage_jump_bounds[0]
    assert report.switching_work_certified


def test_transverse_decay_prices_integrated_moment_and_hidden_exchange(report):
    amplitude, rate = report.scaled_norm_squared_upper_bound / 4, report.decay_rate
    assert report.phase_moment_amplitude_upper_bound == amplitude
    assert report.phase_moment_integral_upper_bound == amplitude / rate
    assert (
        report.aggregate_absolute_upper_bound
        == report.entry_aggregate_upper_bound + 9 * G
    )
    assert (
        report.unresolved_exchange_integral_upper_bound
        == 2 * G * report.aggregate_absolute_upper_bound * amplitude / rate
    )
    assert (
        report.storage_remainder_upper_bound
        == 18 * report.scaled_norm_squared_upper_bound
    )
    assert report.integrable_moment_certified and report.transverse_decay_certified
    # Integrating a proved exponential envelope is algebraic; no endpoint,
    # exponential value or response rate fit is produced by this control.
    assert rate * report.phase_moment_integral_upper_bound == amplitude


def test_marked_collar_counterexample_preserves_the_original_windings(mp, report):
    geometry = owner._derive_registered_phase_probe_geometry().successor
    pair = report.marked_witness_pair
    assert pair in geometry.marked_pairs
    pair_index = geometry.marked_pairs.index(pair)
    assert (
        _turns()[pair[1]] - _turns()[pair[0]]
        == geometry.marked_branch_turns[pair_index]
    )
    expected = tuple(
        -DELTA / 6 if j == pair[0] else DELTA / 6 if j == pair[1] else Q(0)
        for j in range(36)
    )
    assert report.marked_witness_phase == expected
    assert sum(expected[:18]) == sum(expected[18:]) == 0
    assert (
        report.marked_witness_gap
        == expected[pair[1]] - expected[pair[0]]
        == DELTA / 3
        > DELTA / 4
    )
    square = _dot(expected, expected)
    assert report.marked_witness_norm_squared == square == DELTA**2 / 18
    assert (
        report.marked_witness_allowance_margin
        == report.initial_centered_phase_upper_bound**2 - square
        > 0
    )
    assert (
        4 * square == 2 * DELTA**2 / 9 < report.initial_scaled_norm_squared_upper_bound
    )
    assert 4 * square < Q(1, 500) ** 2
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, e)
        for turn, e in zip(_turns() * 2, expected, strict=True)
    )
    for side in (0, 1):
        for ring in (0, 1):
            nodes = tuple(18 * side + 9 * ring + j for j in range(9))
            total = sum(
                (theta[nodes[(j + 1) % 9]] - theta[nodes[j]] + mp.pi) % (2 * mp.pi)
                - mp.pi
                for j in range(9)
            )
            assert abs(total / (2 * mp.pi) - (ring + 1)) < mp.mpf("1e-190")
    assert report.strict_identity_nonimplication_certified
    assert not report.strict_marked_identity_certified
    # This larger-class witness proves failure of an isotropic implication,
    # not actual exit of an acquired member from its original marked collar.


def test_absolute_collective_comparison_does_not_certify_uniform_prediction(report):
    matrix = ((-Q(2), 2 * G), (2 * G, Q(0)))
    witness = (G, Q(1))
    applied = tuple(sum(matrix[i][j] * witness[j] for j in (0, 1)) for i in (0, 1))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    quadratic = _dot(witness, applied)
    assert report.comparison_matrix == matrix
    assert report.comparison_determinant == determinant == -4 * G**2 < 0
    assert report.comparison_positive_witness == witness
    assert applied == (Q(0), 2 * G**2)
    assert report.comparison_positive_quadratic_value == quadratic == 2 * G**2 > 0
    assert report.collective_comparison_limitation_certified
    assert not report.uniform_collective_prediction_certified
    # An L1 forcing budget is not a bounded-gain theorem for a comparison
    # with a positive mode. Neither matrix sign diagnoses actual instability.


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_source_geometry_history_and_metric_premises_are_readmitted(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "registration_covariance_certified", False),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "initial_relative_phase_error_upper_bound", Q(0)),
        ((), "centered_entry_norm_upper_bound", Q(0)),
        ((), "initial_registrations_turns", (Q(0), Q(0))),
        ((), "source_correlated_initialization_retained", False),
        ((), "whole_history_covariance_retained", False),
        ((), "state_reset_performed", True),
        ((), "phase_clamped", True),
        ((), "common_storage_reference", "changed reference"),
        ((), "post_duration", Q(16)),
        ((), "switch_time", Q(5385016)),
        ((), "clock", "wrong sustained clock"),
        ((), "registered_observation_error_bound", NOISE),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        (("source",), "source_correlated_initialization_retained", False),
        (("source",), "post_duration", Q(8)),
        (("source",) * 2, "complete_precontact_history_translated", False),
        (("source",) * 2, "registration_turns", Q(0)),
        (("source",) * 3, "initial_relative_phase_mean_assumed_zero", True),
        (("source",) * 3, "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 4, "clock", "wrong renewed clock"),
        (("source",) * 5, "signed_kernel_certified", False),
        (("source",) * 5, "fast_output_tail_upper_bound", Q(0)),
        (("source",) * 6, "cut_form_error_upper_bound", Q(0)),
        (("source",) * 7, "uniform_tube_certified", False),
        (("source",) * 7, "uniform_comparison_upper_bound", 200 * MATURE),
        (("source",) * 7, "bregman_denominator", Q(1)),
        (("source",) * 7, "nominal_cone_certified", False),
        (("source",) * 7, "nominal_cone_margins", (Q(1),) * 3),
        (("source",) * 7, "nominal_contrast_form_cone_factor", Q(2)),
        (("source",) * 7, "initial_bregman_root_upper_bound", Q(0)),
        (("source",) * 7, "bregman_weight_forcing_coefficient", Q(0)),
        (("source",) * 7, "nominal_form_integral_upper_bound", Q(0)),
        (("source",) * 7, "inherited_form_velocity_retained", False),
        (("source",) * 8, "actual_dissipation_certified", False),
        (("source",) * 8, "origin_reference", "new origin"),
        (("source",) * 9, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 9, "prefix_cut_duration", Q(10001)),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_sustained_interaction", lambda **kwargs: changed
        )
        rejected = _assess()
        assert not rejected.certified and not rejected.sharp_source_bound_certified
        assert not rejected.joint_bootstrap_certified
        assert not rejected.all_postattachment_times_covered
        assert not rejected.actual_asymptotic_attraction_certified
    path = ("source",) * 9 + ("handoff", "prefix", "reference")
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
            owner, "_assess_mixed_sustained_interaction", lambda **kwargs: changed
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
            owner, "_assess_mixed_sustained_interaction", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_sustained_interaction", lambda **kwargs: source
    )
    geometry = owner._derive_registered_phase_probe_geometry()
    for changes in (
        dict(bare_gap_lower_bound=Q(3)),
        dict(weighted_gap_lower_bound=Q(2)),
        dict(spectral_gap_certified=False),
        dict(degrees=(8,) + (9,) * 17),
    ):
        altered = replace(geometry, successor=replace(geometry.successor, **changes))
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        rejected = _assess()
        assert (
            not rejected.weighted_gap_certified
            and not rejected.transverse_decay_certified
        )
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
            and not rejected.joint_bootstrap_certified
        )
        assert not rejected.all_postattachment_times_covered and not rejected.certified
        # A failed helper cannot create division by zero in a candidate
        # remainder; the positive law-derived rate is rebuilt independently.
        assert rejected.decay_rate == report.decay_rate > 0


def test_old_finite_and_record_verdicts_do_not_supply_new_conclusions(
    monkeypatch, report
):
    source = report.source
    changed = replace(
        source,
        certified=False,
        centered_identity_certified=False,
        weighted_gap_certified=False,
        damped_centered_bound_certified=False,
        phase_branch_certified=False,
        phase_moment_remainder_certified=False,
        switching_work_certified=False,
        effective_balance_certified=False,
        form_prediction_certified=False,
        phase_prediction_certified=False,
        comparison_limitation_certified=False,
        observation_allowances_admitted=False,
        registered_record_prediction_certified=False,
        aligned_record_prediction_certified=False,
        centered_envelope=replace(
            source.centered_envelope, uniform_phase_norm_upper_bound=Q(0)
        ),
        marked_gap_upper_bound=Q(0),
        registration_margin=Q(0),
        phase_moment_remainder_upper_bound=Q(0),
        storage_remainder_upper_bound=Q(0),
        uniform_form_error_upper_bound=Q(0),
        phase_error_upper_bound=Q(0),
        endpoint_form_error_upper_bound=Q(0),
        registered_record_error_upper_bound=Q(0),
        aligned_record_error_upper_bound=Q(0),
        aligned_storage_jump_bounds=(Q(0), Q(0)),
        registered_storage_jump_bounds=(Q(0), Q(0)),
        status="finite_or_record_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_sustained_interaction", lambda **kwargs: changed
    )
    rebuilt = _assess()
    assert rebuilt.certified and rebuilt.transverse_decay_certified
    assert rebuilt.source_comparison_upper_bound == report.source_comparison_upper_bound
    assert (
        rebuilt.scaled_norm_squared_upper_bound
        == report.scaled_norm_squared_upper_bound
    )
    assert rebuilt.phase_barrier_margin == report.phase_barrier_margin
    assert (
        rebuilt.phase_moment_integral_upper_bound
        == report.phase_moment_integral_upper_bound
    )
    assert (
        rebuilt.registered_storage_jump_bounds == report.registered_storage_jump_bounds
    )
    assert not rebuilt.strict_marked_identity_certified
    assert not rebuilt.uniform_collective_prediction_certified
    assert source == report.source


def test_raw_fixed_width_contract_and_exact_evidence_preserve_limits(
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
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        outside = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not outside.observation_allowances_admitted and not outside.certified
        assert (
            outside.transverse_decay_certified and outside.integrable_moment_certified
        )
        assert outside.all_postattachment_times_covered
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert exact.certified and exact.decay_rate == report.decay_rate
    assert (
        exact.scaled_norm_squared_upper_bound == report.scaled_norm_squared_upper_bound
    )
    assert (
        report.certified
        and report.status == "conditional_transverse_attraction_with_identity_boundary"
    )
    assert report.initial_registrations_turns == (Q(0), Q(1, 4))
    assert report.switch_age == SWITCH_AGE and report.switch_time == 5385000
    assert report.original_clock_start_time * Q(1023, 1024) == report.switch_time
    assert report.previous_certified_duration == 4096
    assert report.scaled_radius_limit == Q(1, 500)
    assert report.relative_phase_excess_limit == Q(1, 1000)
    assert report.phase_sector_bound == pi_interval() / 2 + Q(1, 1000)
    assert report.shared_phase_scale == 2 and report.auxiliary_capacity == Q(4, 9)
    assert report.physical_capacities == (Q(1),) * 36 and report.internal_divisor == 9
    assert report.full_state_dimension == 72
    assert report.newly_observed_aggregates == report.newly_observed_phases == 0
    assert report.numerical_response_allowance == 0
    for name in (
        "all_postattachment_times_covered",
        "actual_asymptotic_attraction_certified",
        "same_complete_and_effective_laws",
        "whole_history_covariance_retained",
        "source_correlated_initialization_retained",
        "original_attachment_state_retained",
        "original_attachment_work_retained_once",
    ):
        assert getattr(report, name) is True
    for name in (
        "strict_marked_identity_certified",
        "uniform_collective_prediction_certified",
        "original_finite_identity_contract_extended",
        "original_finite_prediction_contract_extended",
        "new_endpoint_or_phase_record_claimed",
        "record_allowances_used_for_stability",
        "prior_finite_endpoint_used_as_initial_state",
        "direct_entry_energy_adds_attachment_twice",
        "unequal_internal_blocks_assumed_commuting",
        "stiffness_derivative_consumed",
        "independently_fixed_copy_gauges_assumed",
        "boundary_convexity_assumed",
        "phase_clamped",
        "unresolved_storage_assumed_nonnegative",
        "marked_witness_is_acquired_state",
        "actual_strict_identity_failure_claimed",
        "positive_comparison_direction_is_eigenvector",
        "actual_collective_instability_claimed",
        "all_time_prediction_failure_proved",
        "state_reset_performed",
        "original_source_reacquired",
        "original_source_replayed",
        "original_source_trajectory_evaluated",
        "reduced_trajectory_evaluated",
        "exact_actual_two_coordinate_closure",
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
        report.decay_rate = Q(0)
    for name in (
        "source",
        "horizon",
        "initial_phase",
        "phase_observation",
        "initial_reading",
        "threshold",
        "baseline",
        "port_strength",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached original source work")

    monkeypatch.setattr(owner, "_assess_mixed_sustained_interaction", forbidden)
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
    for name in (
        "switch_age",
        "switch_time",
        "previous_certified_duration",
        "inherited_output_error_upper_bound",
        "initial_relative_phase_error_upper_bound",
        "centered_entry_norm_upper_bound",
        "source_comparison_upper_bound",
        "nominal_centered_form_upper_bound",
        "nominal_centered_phase_upper_bound",
        "initial_centered_form_upper_bound",
        "initial_centered_phase_upper_bound",
        "initial_scaled_norm_squared_upper_bound",
        "normalized_metric_lower_bound",
        "normalized_metric_upper_bound",
        "metric_lower_bound",
        "metric_upper_bound",
        "shared_metric_decay_rate",
        "decay_rate",
        "scaled_norm_squared_upper_bound",
        "scaled_radius_margin",
        "phase_stiffness_lower_bound",
        "phase_stiffness_margin",
        "metric_stiffness_margin",
        "entry_aggregate_upper_bound",
        "aggregate_absolute_upper_bound",
        "initial_full_storage_upper_bound",
        "phase_boundary_storage_lower_bound",
        "phase_barrier_margin",
        "phase_moment_amplitude_upper_bound",
        "phase_moment_integral_upper_bound",
        "unresolved_exchange_integral_upper_bound",
        "storage_remainder_upper_bound",
        "phase_switching_work_allowance",
        "marked_witness_gap",
        "marked_witness_norm_squared",
        "marked_witness_allowance_margin",
        "comparison_determinant",
        "comparison_positive_quadratic_value",
    ):
        record_property(name, str(getattr(report, name)))
    for name in ("aligned_storage_jump_bounds", "registered_storage_jump_bounds"):
        record_property(name, json.dumps([str(v) for v in getattr(report, name)]))
