"""One nonlinear collective law has a conditional carried-state contract.

Independent static rows and storage identities retain hidden information.
The exact model semigroup and its separate error envelopes are not numerical
response evaluations or statements of minimal state dimension.
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
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_effective_interaction as owner
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
SEGMENT = Q(8)
HORIZON = Q(16)


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


def _relative_common(values):
    whole = sum(values) / 36
    return tuple(
        sum(values[18 * side : 18 * (side + 1)]) / 18 - whole
        for side in (0, 1)
        for _ in range(18)
    )


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
    return owner._assess_mixed_effective_interaction(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("effective controls cannot replay or advance a source")

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
        pytest.fail("static effective proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def _copy_center(values):
    return _center(values[:18]) + _center(values[18:])


def _potential(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    return sum(
        1 - mp.cos(theta[18 * side + b] - theta[18 * side + a])
        for side in (0, 1)
        for a, b, _ in _edges((Q(1),) * 63)
    )


def _effective(mp, q, phase):
    gamma = 1 / (1023 * mp.pi)
    return -2 * q - 18 * gamma * mp.sin(phase), 2 * gamma * q / 9


def test_reduced_field_uses_one_nonlinear_law_for_both_registrations(mp):
    flow = owner._mixed_effective_interaction_field()
    gamma = 1 / (1023 * mp.pi)
    q = Q(1, 10000)
    for phase, angle in ((I(0), mp.mpf(0)), (pi_interval() / 2, mp.pi / 2)):
        expected = _effective(mp, _mp(mp, q), angle)
        state = (I(q), phase)
        actual = flow(state)
        assert state == (I(q), phase) and len(actual) == 2
        for interval, value in zip(actual, expected, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        shifted = flow((I(q), phase + 2 * pi_interval()))
        for interval, value in zip(shifted, expected, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        assert abs(expected[1] - 2 * gamma * _mp(mp, q) / 9) < mp.mpf("1e-200")
    aligned = _effective(mp, _mp(mp, q), mp.mpf(0))
    quarter = _effective(mp, _mp(mp, q), mp.pi / 2)
    assert abs(quarter[0] - aligned[0] + 18 * gamma) < mp.mpf("1e-200")
    assert aligned[1] == quarter[1] != 0
    with pytest.raises(TypeError):
        owner._mixed_effective_interaction_field(registration="quarter_turn")


def test_actual_projected_rows_retain_the_internal_moment_remainder(mp):
    forms = _center(tuple(Q((j * 7) % 17 - 8, 10**5) for j in range(36)))
    centered = _copy_center(tuple(Q((j * 5) % 11 - 5, 10**5) for j in range(36)))
    weights = (_weights(mp, 0), _weights(mp, 1))
    gamma = 1 / (1023 * mp.pi)
    q = _mp(mp, -sum(forms[:18]))
    flow = _new_field(weights)
    for shift, angle in ((I(0), mp.mpf(0)), (pi_interval() / 2, mp.pi / 2)):
        phases = tuple(
            _mp(mp, v) + (angle if j >= 18 else 0) for j, v in enumerate(centered)
        )
        state = _state(forms, centered)
        state = state[:54] + tuple(v + shift for v in state[54:])
        dx, dy = _field(mp, forms, phases, weights)
        actual = flow(state)
        for interval, value in zip(actual, dx + dy, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        psi = (sum(phases[18:]) - sum(phases[:18])) / 18
        moment = sum(mp.sin(phases[j + 18] - phases[j]) for j in range(18))
        residual = moment - 18 * mp.sin(psi)
        fq, fp = _effective(mp, q, psi)
        assert abs(-sum(dx[:18]) - fq + gamma * residual) < mp.mpf("1e-200")
        assert abs((sum(dy[18:]) - sum(dy[:18])) / 18 - fp) < mp.mpf("1e-200")
        norm_squared = sum(_mp(mp, v) ** 2 for v in centered)
        assert abs(residual) <= norm_squared
        assert abs(sum(dx)) < mp.mpf("1e-200")
    # A static larger-class witness: two states with the same Q and psi
    # have distinct moments; neither is asserted to be an acquired endpoint.
    a = mp.mpf("0.001")
    assert 0 < 18 - (16 + 2 * mp.cos(a)) <= a**2


def test_coarse_storage_matches_full_storage_with_a_paid_residual(mp):
    weights = (_weights(mp, 0), _weights(mp, 1))
    x = tuple(
        _mp(mp, v)
        for v in _center(tuple(Q((j * 3) % 17 - 8, 10**5) for j in range(36)))
    )
    yc = tuple(
        _mp(mp, v) for v in _copy_center(tuple(Q(j % 5 - 2, 10**5) for j in range(36)))
    )
    xc, common = _copy_center(x), _relative_common(x)
    q = -sum(x[:18])
    hx, gx = _h_apply(mp, x, weights), _exchange(x)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    lc = tuple(
        a + 9 * b for a, b in zip(_h_apply(mp, xc, weights), _exchange(xc), strict=True)
    )
    assert abs(9 * _dot(common, common) - q**2) < mp.mpf("1e-200")
    assert abs(_dot(x, lx) / 2 - q**2 - _dot(xc, lc) / 2) < mp.mpf("1e-200")
    original = _potential(mp, (Q(0),) * 36)
    for psi in (mp.mpf(0), mp.pi / 2):
        phases = tuple(v + (psi if j >= 18 else 0) for j, v in enumerate(yc))
        internal = _potential(mp, phases) - original
        paired = sum(1 - mp.cos(phases[j + 18] - phases[j]) for j in range(18))
        full = _dot(x, lx) / 2 + internal + 9 * paired
        coarse = q**2 + 162 * (1 - mp.cos(psi))
        remainder = full - coarse
        pieces = _dot(xc, lc) / 2 + internal + 9 * (paired - 18 * (1 - mp.cos(psi)))
        assert abs(remainder - pieces) < mp.mpf("1e-200")
        assert abs(internal) <= 9 * _dot(yc, yc)
        assert abs(9 * (paired - 18 * (1 - mp.cos(psi)))) <= 9 * _dot(yc, yc)
        assert abs(remainder) <= 18 * (_dot(xc, xc) + _dot(yc, yc))
    # Nonnegative remainder is not a generic law. This separate static
    # anti-phase witness is outside the two acquired preparation branches.
    a = mp.mpf("0.00001")
    yc = (a,) * 9 + (-a,) * 9 + (-a,) * 9 + (a,) * 9
    phases = tuple(v + (mp.pi if j >= 18 else 0) for j, v in enumerate(yc))
    internal = _potential(mp, phases) - original
    paired = sum(1 - mp.cos(phases[j + 18] - phases[j]) for j in range(18))
    remainder = internal + 9 * paired - 324
    assert remainder < 0
    assert abs(remainder) <= 18 * _dot(yc, yc)


def test_reduced_loss_and_actual_exchange_defect_are_distinct(mp):
    gamma = 1 / (1023 * mp.pi)
    weights = (_weights(mp, 0), _weights(mp, 1))
    common = (-Q(1, 10000),) * 18 + (Q(1, 10000),) * 18
    fine = _copy_center(tuple(Q((j * 5) % 11 - 5, 10**5) for j in range(36)))
    forms = tuple(a + b for a, b in zip(common, fine, strict=True))
    xc = tuple(_mp(mp, value) for value in fine)
    perturb = _copy_center(tuple(Q((j * 3) % 7 - 3, 10**4) for j in range(36)))
    for alpha in (mp.mpf(0), mp.pi / 2):
        phases = tuple(
            _mp(mp, v) + (alpha if j >= 18 else 0) for j, v in enumerate(perturb)
        )
        x = tuple(_mp(mp, v) for v in forms)
        q, psi = -sum(x[:18]), (sum(phases[18:]) - sum(phases[:18])) / 18
        fq, fp = _effective(mp, q, psi)
        derivative_reduced = 2 * q * fq + 162 * mp.sin(psi) * fp
        assert abs(derivative_reduced + 4 * q**2) < mp.mpf("1e-200")
        dx, dy = _field(mp, x, phases, weights)
        dq, dpsi = -sum(dx[:18]), (sum(dy[18:]) - sum(dy[:18])) / 18
        moment = sum(mp.sin(phases[j + 18] - phases[j]) for j in range(18))
        residual = moment - 18 * mp.sin(psi)
        projected_rate = 2 * q * dq + 162 * mp.sin(psi) * dpsi
        exchange = -2 * gamma * q * residual
        assert abs(projected_rate + 4 * q**2 - exchange) < mp.mpf("1e-200")
        lx = tuple(
            a + 9 * b
            for a, b in zip(_h_apply(mp, x, weights), _exchange(x), strict=True)
        )
        lc = tuple(
            a + 9 * b
            for a, b in zip(_h_apply(mp, xc, weights), _exchange(xc), strict=True)
        )
        full_rate = -_dot(lx, lx) / 9
        assert _dot(lc, lc) > 0
        assert abs(
            full_rate - projected_rate + _dot(lc, lc) / 9 - 2 * gamma * q * residual
        ) < mp.mpf("1e-200")
        centered_square = sum(_mp(mp, v) ** 2 for v in perturb)
        assert abs(exchange) <= 2 * _mp(mp, G) * abs(q) * centered_square
    assert abs(exchange) > mp.mpf("1e-15")


def test_error_transfer_prices_damping_and_phase_feedback(report):
    transfer = report.direct_transfer
    a, b, r, h = (
        transfer.initial_form_error,
        transfer.initial_phase_error,
        transfer.moment_error,
        transfer.duration,
    )
    # Invert the independent positive feedback matrix I-L. Its determinant
    # prices returning phase error; simply adding the form forcing misses it.
    matrix = ((Q(1), -9 * G), (-2 * G * h / 9, Q(1)))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    inverse = tuple(
        tuple(v / determinant for v in row)
        for row in ((Q(1), 9 * G), (2 * G * h / 9, Q(1)))
    )
    for row in (0, 1):
        for column in (0, 1):
            assert sum(
                matrix[row][index] * inverse[index][column] for index in (0, 1)
            ) == int(row == column)
    forcing = (a + G * r / 2, b)
    solved = tuple(
        sum(inverse[row][column] * forcing[column] for column in (0, 1))
        for row in (0, 1)
    )
    assert (
        transfer.uniform_form_error_upper_bound,
        transfer.phase_error_upper_bound,
    ) == solved
    assert transfer.feedback_denominator == determinant > 0
    assert transfer.decay_upper_bound == Q(1, 2**32)
    assert (
        transfer.endpoint_form_error_upper_bound
        == a * transfer.decay_upper_bound + 9 * G * solved[1] + G * r / 2
    )
    zero = owner._effective_error_transfer(
        initial_form_error=0, initial_phase_error=0, moment_error=0, duration=8
    )
    assert (
        zero.uniform_form_error_upper_bound
        == zero.phase_error_upper_bound
        == zero.endpoint_form_error_upper_bound
        == 0
    )
    phase_only = owner._effective_error_transfer(
        initial_form_error=0, initial_phase_error=Q(1, 1000), moment_error=0, duration=8
    )
    assert (
        phase_only.uniform_form_error_upper_bound
        > 9 * G * phase_only.initial_phase_error
        > 0
    )
    assert phase_only.phase_error_upper_bound > phase_only.initial_phase_error
    assert (
        phase_only.endpoint_form_error_upper_bound
        == phase_only.uniform_form_error_upper_bound
    )


def test_split_continuation_carries_both_errors_without_new_reading(report):
    first, continued, direct = (
        report.first_transfer,
        report.continued_transfer,
        report.direct_transfer,
    )
    assert first.duration == continued.duration == SEGMENT
    assert direct.duration == first.duration + continued.duration == HORIZON
    assert (
        first.initial_form_error
        == direct.initial_form_error
        == report.inherited_output_error_upper_bound
    )
    assert first.initial_phase_error == direct.initial_phase_error == 6 * ETA
    assert continued.initial_form_error == first.endpoint_form_error_upper_bound
    assert continued.initial_phase_error == first.phase_error_upper_bound > 6 * ETA
    assert (
        first.moment_error
        == continued.moment_error
        == direct.moment_error
        == report.phase_moment_remainder_upper_bound
    )
    assert continued.initial_form_error != report.inherited_output_error_upper_bound
    for transfer in (first, continued, direct):
        u, v, a = (
            transfer.uniform_form_error_upper_bound,
            transfer.phase_error_upper_bound,
            transfer.endpoint_form_error_upper_bound,
        )
        assert 0 < a < u < NOISE and v > 0
        assert transfer.duration == 8 or transfer.duration == 16
    # This invalid reset understates both carried errors. It is a static
    # comparison of error contracts, not a simulated alternate trajectory.
    forgotten = owner._effective_error_transfer(
        initial_form_error=0,
        initial_phase_error=0,
        moment_error=first.moment_error,
        duration=8,
    )
    assert (
        forgotten.endpoint_form_error_upper_bound
        < continued.endpoint_form_error_upper_bound
    )
    assert forgotten.phase_error_upper_bound < continued.phase_error_upper_bound
    assert (
        continued.endpoint_form_error_upper_bound
        != direct.endpoint_form_error_upper_bound
    )
    assert (
        report.first_segment_certified
        and report.continuation_certified
        and report.direct_prediction_certified
    )


def test_common_model_handoff_preserves_autonomy_and_phase_state(mp, report):
    # An arbitrary constructed checkpoint, not an acquired intermediate state.
    # Formal local flow coefficients: two successive durations have the same
    # second-order jet as their sum. No flow or endpoint is evaluated here.
    q, psi = mp.mpf("0.0001"), mp.pi / 2 - mp.mpf("0.00001")
    gamma = 1 / (1023 * mp.pi)
    fq, fp = _effective(mp, q, psi)
    jf = (-2 * fq - 18 * gamma * mp.cos(psi) * fp, 2 * gamma * fq / 9)
    state = (I(Q(1, 10000)), pi_interval() / 2 - I(Q(1, 100000)))
    field = owner._mixed_effective_interaction_field()
    rates = field(state)
    chain = field(
        tuple(Jet((value, rate)) for value, rate in zip(state, rates, strict=True))
    )
    for jet, derivative in zip(chain, jf, strict=True):
        assert _mp(mp, jet.coeffs[1].lo) <= derivative <= _mp(mp, jet.coeffs[1].hi)
    first_duration, second_duration = Q(2, 7), Q(3, 11)
    for component in jf:
        composed_coefficient = (
            _mp(
                mp,
                first_duration**2 / 2
                + first_duration * second_duration
                + second_duration**2 / 2,
            )
            * component
        )
        direct_coefficient = (
            _mp(mp, (first_duration + second_duration) ** 2 / 2) * component
        )
        assert abs(composed_coefficient - direct_coefficient) < mp.mpf("1e-200")
    # A phase reset changes the state even if this small difference is not
    # resolved by the original endpoint tolerance. No necessity claim follows.
    reset = _effective(mp, q, mp.pi / 2)
    assert abs(reset[0] - fq) > mp.mpf("1e-15")
    assert reset[1] == fp
    assert report.memberwise_semigroup_certified


def test_total_window_rechecks_identity_work_and_prediction_budgets(
    report, record_property
):
    entry = 4 * QHI + 200 * MATURE
    denominator = 1 - 4 * G * HORIZON
    norm = entry / denominator
    assert report.centered_entry_norm_upper_bound == entry
    assert report.centered_norm_growth_denominator == denominator > 0
    assert report.centered_norm_upper_bound == norm < Q(1, 100) < Q(1, 32)
    assert report.phase_drift_upper_bound == 6 * G * HORIZON * norm
    assert (
        report.marked_gap_upper_bound
        == QHI + 300 * MATURE + report.phase_drift_upper_bound
    )
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    aggregate = (
        report.slow_amplitude_bounds[1]
        + report.inherited_output_error_upper_bound
        + 9 * G
    )
    phase = 6 * ETA + 2 * G * HORIZON * aggregate / 9
    assert report.aggregate_absolute_upper_bound == aggregate
    assert report.relative_phase_deviation_upper_bound == phase
    assert report.boundary_branch_margin == Q(1, 4) - phase - 2 * norm > 0
    assert report.phase_moment_remainder_upper_bound == norm**2
    assert report.storage_remainder_upper_bound == 18 * norm**2
    assert (
        report.unresolved_exchange_integral_upper_bound
        == 2 * G * HORIZON * aggregate * norm**2
    )
    work = 972 * ETA + Q(9, 2) * entry**3
    assert report.phase_switching_work_allowance == work
    assert report.registered_storage_jump_bounds == (
        162 - work,
        162 + work + 9 * entry**2,
    )
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (entry + 18 * ETA) ** 2)
    assert report.seam_storage_jump == 0
    for name, transfer in (
        ("direct", report.direct_transfer),
        ("continued", report.continued_transfer),
    ):
        expected = (
            18 * norm**2
            + 2 * aggregate * transfer.uniform_form_error_upper_bound
            + 162 * transfer.phase_error_upper_bound
        )
        assert (
            getattr(report, name + "_storage_prediction_error_upper_bound") == expected
        )
    endpoint = report.continued_transfer.endpoint_form_error_upper_bound
    assert report.registered_record_error_upper_bound == endpoint + NOISE
    assert report.aligned_record_error_upper_bound == endpoint + NOISE
    assert (
        report.registered_record_prediction_certified
        and report.aligned_record_prediction_certified
    )
    assert report.centered_identity_certified and report.effective_balance_certified
    assert report.switching_work_certified and report.certified
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        outside = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not outside.observation_allowances_admitted and not outside.certified
        assert outside.direct_prediction_certified and outside.continuation_certified
        assert outside.effective_balance_certified
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert (
        exact.registered_record_error_upper_bound
        == exact.aligned_record_error_upper_bound
        == endpoint
    )
    assert exact.certified
    for name in (
        "switch_age",
        "post_duration",
        "segment_duration",
        "switch_time",
        "seam_time",
        "total_duration",
        "inherited_output_error_upper_bound",
        "initial_relative_phase_error_upper_bound",
        "centered_entry_norm_upper_bound",
        "centered_norm_upper_bound",
        "phase_drift_upper_bound",
        "registration_margin",
        "aggregate_absolute_upper_bound",
        "relative_phase_deviation_upper_bound",
        "boundary_branch_margin",
        "phase_moment_remainder_upper_bound",
        "phase_switching_work_allowance",
        "storage_remainder_upper_bound",
        "unresolved_exchange_integral_upper_bound",
        "direct_storage_prediction_error_upper_bound",
        "continued_storage_prediction_error_upper_bound",
        "registered_record_error_upper_bound",
        "aligned_record_error_upper_bound",
        "prediction_margin",
    ):
        record_property(name, str(getattr(report, name)))
    for name in ("aligned_storage_jump_bounds", "registered_storage_jump_bounds"):
        record_property(name, json.dumps([str(v) for v in getattr(report, name)]))
    for name in ("direct_transfer", "first_transfer", "continued_transfer"):
        transfer = getattr(report, name)
        for field in (
            "initial_form_error",
            "initial_phase_error",
            "duration",
            "moment_error",
            "uniform_form_error_upper_bound",
            "phase_error_upper_bound",
            "endpoint_form_error_upper_bound",
        ):
            record_property(name + "." + field, str(getattr(transfer, field)))


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_source_primitives_and_consumed_ancestor_bounds_are_readmitted(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "registration_covariance_certified", False),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "initial_relative_phase_error_upper_bound", Q(0)),
        ((), "centered_entry_norm_upper_bound", Q(0)),
        ((), "registration_turns", Q(0)),
        ((), "registration_angle_bounds", I(0)),
        ((), "translated_copy", "donor"),
        ((), "complete_precontact_history_translated", False),
        ((), "source_errors_translated", True),
        ((), "later_phase_intervention_performed", True),
        ((), "same_complete_contact_law", False),
        ((), "original_lifts_and_error_association_retained", False),
        ((), "common_storage_reference", "rotated reference"),
        ((), "post_duration", Q(16)),
        ((), "switch_time", Q(5385008)),
        ((), "clock", "wrong registration clock"),
        ((), "registered_observation_error_bound", NOISE),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        (("source",), "initial_relative_phase_mean_assumed_zero", True),
        (("source",), "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 2, "clock", "wrong renewed clock"),
        (("source",) * 3, "signed_kernel_certified", False),
        (("source",) * 3, "fast_output_tail_upper_bound", Q(0)),
        (("source",) * 3, "correlated_initialization_retained", False),
        (("source",) * 4, "cut_form_error_upper_bound", Q(0)),
        (("source",) * 5, "uniform_tube_certified", False),
        (("source",) * 5, "bregman_denominator", Q(1)),
        (("source",) * 5, "nominal_norm_upper_bound", Q(0)),
        (("source",) * 6, "actual_dissipation_certified", False),
        (("source",) * 6, "selected_mu", Q(2)),
        (("source",) * 7, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 7, "prefix_cut_duration", Q(10001)),
        (("source",) * 7 + ("handoff",), "prefix_duration", Q(2000001)),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_relative_registration", lambda **kwargs: changed
        )
        rejected = _assess()
        assert not rejected.certified and not rejected.continuation_certified
        assert not rejected.centered_identity_certified
        assert not rejected.exact_memberwise_reduced_semigroup
    path = ("source",) * 7 + ("handoff", "prefix", "reference")
    reference = (
        source.source.source.source.source.source.source.source.handoff.prefix.reference
    )
    reset = replace(reference.sources[0], phase_reset_error_bound=4 * ETA)
    changed = _replace_at(
        source,
        path,
        sources=(reset, reference.sources[1]),
        donor=replace(reference.donor, source=reset),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_relative_registration", lambda **kwargs: changed
    )
    assert not _assess().certified
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("recipient_isolated_deadline", Q(10**10 - 1)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("clock", "wrong original clock"),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_relative_registration", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_relative_registration", lambda **kwargs: source
    )
    registered = owner._derive_registered_phase_probe_geometry()
    changed = replace(
        registered, successor=replace(registered.successor, bare_gap_lower_bound=Q(3))
    )
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: changed
    )
    assert not _assess().certified


def test_unused_affine_and_old_record_verdicts_do_not_drive_the_common_law(
    monkeypatch, report
):
    original = report.source
    changed = replace(
        original,
        certified=False,
        centered_identity_certified=False,
        relative_phase_drift_certified=False,
        phase_moment_remainder_certified=False,
        switching_work_certified=False,
        affine_prediction_certified=False,
        blind_prediction_obstruction_certified=False,
        common_prediction_obstruction_certified=False,
        observation_allowances_admitted=False,
        registered_record_certified=False,
        aligned_record_certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        registered_true_bounds=(Q(0), Q(0)),
        aligned_true_bounds=(Q(0), Q(0)),
        registered_record_bounds=(Q(0), Q(0)),
        aligned_record_bounds=(Q(0), Q(0)),
        centered_norm_upper_bound=Q(0),
        phase_moment_deficit_upper_bound=Q(0),
        storage_jump_bounds=(Q(0), Q(0)),
        affine_prediction_error_upper_bound=Q(0),
        status="old_short_task_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_relative_registration", lambda **kwargs: changed
    )
    rebuilt = _assess()
    assert rebuilt.certified and rebuilt.continuation_certified
    assert rebuilt.direct_transfer == report.direct_transfer
    assert rebuilt.continued_transfer == report.continued_transfer
    assert (
        rebuilt.registered_storage_jump_bounds == report.registered_storage_jump_bounds
    )
    assert rebuilt.centered_norm_upper_bound == report.centered_norm_upper_bound
    assert original == report.source


def test_raw_reduced_state_and_reader_inputs_fail_before_arithmetic(monkeypatch):
    field = owner._mixed_effective_interaction_field()
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or sine arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_relative_registration", forbidden)
    monkeypatch.setattr(owner, "sin", forbidden)
    monkeypatch.setattr(owner, "jet_sin", forbidden)
    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
    )
    for bad in invalid:
        for side in (0, 1):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                field(tuple(bad if j == side else Q(0) for j in range(2)))
    for bad in (
        (0,),
        (0, 0, 0),
        {0, 1},
        {0: 0, 1: 0},
        repeat(0),
        (Jet((I(0), I(1))), I(0)),
        (Jet((I(0), I(1))), Jet((I(0), I(1), I(1)))),
    ):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            field(bad)
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in invalid + (-Q(1, 10**100),):
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
        for bad in invalid + (-Q(1, 10**100),):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})
    valid = dict(
        initial_form_error=Q(1, 100),
        initial_phase_error=Q(1, 100),
        moment_error=Q(1, 100),
        duration=Q(8),
    )
    for name in valid:
        for bad in invalid + (-Q(1, 10**100),):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                owner._effective_error_transfer(**(valid | {name: bad}))
    for bad in (Q(0), Q(7), Q(9), Q(17), Q(8) + Q(1, 10**100)):
        with pytest.raises(ValueError):
            owner._effective_error_transfer(**(valid | {"duration": bad}))
    normalized = []
    original = owner._exact

    def traced(value, label):
        normalized.append(label)
        return original(value, label)

    monkeypatch.setattr(owner, "_exact", traced)
    with pytest.raises(TypeError):
        owner._effective_error_transfer(
            initial_form_error=-1, initial_phase_error=0, moment_error=0, duration=True
        )
    assert normalized == [
        "initial_form_error",
        "initial_phase_error",
        "moment_error",
        "duration",
    ]


def test_fixed_width_jets_and_immutable_protocol_preserve_the_continuation(mp, report):
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
    arguments = dict(
        initial_form_error=ERROR,
        initial_phase_error=NOISE,
        moment_error=Q(1, 1000),
        duration=Q(8),
    )
    assert owner._effective_error_transfer(
        **arguments
    ) == owner._effective_error_transfer(**{k: fixed(v) for k, v in arguments.items()})
    field = owner._mixed_effective_interaction_field()
    state = (Q(-1, 32), Q(1, 7))
    fixed_rates = field(fixed(v) for v in state)
    scalar_rates = field(state)
    assert fixed_rates == scalar_rates
    directions = (Q(1, 11), Q(-1, 13))
    jets = tuple(
        Jet((I(value), I(direction)))
        for value, direction in zip(state, directions, strict=True)
    )
    jet_rates = field(jets)
    gamma = 1 / (1023 * mp.pi)
    dq, dpsi = (_mp(mp, v) for v in directions)
    derivative = (
        -2 * dq - 18 * gamma * mp.cos(_mp(mp, state[1])) * dpsi,
        2 * gamma * dq / 9,
    )
    for jet, scalar, expected in zip(jet_rates, scalar_rates, derivative, strict=True):
        assert jet.coeffs[0] == scalar
        assert _mp(mp, jet.coeffs[1].lo) <= expected <= _mp(mp, jet.coeffs[1].hi)
    assert report.initial_registrations_turns == (Q(0), Q(1, 4))
    assert report.reduced_state_coordinates == ("Q", "psi")
    assert report.post_duration == 16 and report.segment_duration == 8
    assert report.switch_age == SWITCH_AGE and report.switch_time == 5385000
    assert report.seam_time == 5385008 and report.total_duration == 5385016
    assert report.original_clock_horizon * Q(1023, 1024) == report.total_duration
    assert report.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
    assert (
        report.newly_observed_aggregates == 0 and report.required_endpoint_records == 2
    )
    assert report.numerical_response_allowance == report.seam_storage_jump == 0
    for name in (
        "same_law_for_both_preparations",
        "original_complete_field_unchanged",
        "whole_history_covariance_retained",
        "source_correlated_initialization_retained",
        "predicted_state_carried_at_seam",
        "source_member_correlation_carried_at_seam",
        "phase_lift_and_uncertainty_carried_at_seam",
        "seam_is_bookkeeping_only",
        "exact_memberwise_reduced_semigroup",
        "independent_record_errors",
    ):
        assert getattr(report, name) is True
    for name in (
        "pressure_selected_from_preparation_label",
        "independent_marginal_state_reselection",
        "state_reset_performed",
        "phase_clamped",
        "microscopic_residual_queried_at_seam",
        "actual_state_queried_at_seam",
        "original_source_reacquired",
        "original_source_replayed",
        "original_source_trajectory_evaluated",
        "reduced_trajectory_evaluated",
        "exact_actual_two_coordinate_closure",
        "propagated_bound_equality_claimed",
        "unresolved_storage_assumed_nonnegative",
        "physical_metric_invented_for_error_coordinates",
        "earlier_affine_predictions_invalidated",
        "dynamically_resolved_phase_coordinate_necessary",
        "minimal_state_dimension_claimed",
        "observation_error_cancellation_assumed",
        "all_future_identity_claimed",
        "reference_shift_used_to_erase_switch_work",
        "runtime_event_executed",
        "operator_occurrence_derived",
        "exponentials_evaluated",
        "roots_numerically_evaluated",
        "physical_property_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    with pytest.raises(FrozenInstanceError):
        report.seam_time = Q(0)
    with pytest.raises(FrozenInstanceError):
        report.continued_transfer.initial_phase_error = Q(0)
    for name in (
        "source",
        "horizon",
        "phase_label",
        "initial_phase",
        "initial_reading",
        "threshold",
        "port_strength",
        "baseline",
        "phase_observation",
        "seam_state",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
