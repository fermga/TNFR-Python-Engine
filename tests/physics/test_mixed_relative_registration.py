"""A fixed whole-history phase translation changes a supplied interaction.

Static full-state oracles distinguish internal identity from relative boundary
registration. Constructed states are not evaluated acquired endpoints. The
source, field and prospective response are never advanced by these controls.
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
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
from tnfr.physics import _mixed_reciprocal_partner as reciprocal_owner
from tnfr.physics import _mixed_relative_registration as owner
from tnfr.physics import _mixed_two_channel_circulation as field_owner
from tnfr.physics import _sine_flow as flow_owner
from tnfr.physics._sine_coupling_successor import _coupling_successor_field

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**9)
THRESHOLD = Q(1, 10**8)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
MATURE = Q(1, 2**28)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
QLO, QHI = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
G = Q(1, 3000)
SWITCH_AGE = Q(3375000)
HORIZON = Q(8)
DECAY = Q(1, 2**16)


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
    return owner._assess_mixed_relative_registration(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("registration controls cannot replay or advance a source")

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
        pytest.fail("static registration proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def _copy_center(values):
    return _center(values[:18]) + _center(values[18:])


def _translated_state(forms, phases):
    state = _state(forms, phases)
    return state[:54] + tuple(value + pi_interval() / 2 for value in state[54:])


def _old_fields(weights):
    arguments = dict(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )
    internal = tuple(
        _coupling_successor_field(
            completion="common_sense",
            new_edge_conductances=row,
            loss_normalization="bare_degree",
        )
        for row in weights
    )

    def cut(state):
        first = internal[0](state[:18] + state[36:54])
        second = internal[1](state[18:36] + state[54:])
        return first[:18] + second[:18] + first[18:] + second[18:]

    return (
        (18, field_owner._mixed_two_channel_field(**arguments, law="full")),
        (0, cut),
        (9, reciprocal_owner._mixed_reciprocal_partner_field(**arguments)),
    )


def test_complete_prior_history_is_covariant_under_recipient_translation(mp):
    forms = _center(tuple(Q((j * 7) % 17 - 8, 10**5) for j in range(36)))
    phases = _copy_center(tuple(Q((j * 5) % 11 - 5, 10**5) for j in range(36)))
    weights = (_weights(mp, 0), _weights(mp, 1))
    original = _state(forms, phases)
    translated = _translated_state(forms, phases)
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine = _h_apply(mp, x, weights), _sine(mp, phases)
    gamma = 1 / (1023 * mp.pi)
    for slots, field in _old_fields(weights):
        gx = _exchange(x, slots)
        expected = tuple(
            (-h + gamma * s) / 9 - g for h, s, g in zip(hx, sine, gx, strict=True)
        ) + tuple(gamma * h / 9 for h in hx)
        for state in (original, translated):
            actual = field(state)
            for interval, value in zip(actual, expected, strict=True):
                assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        assert translated[:36] == original[:36]
    # This is vector-field covariance for every stage, not a sampled trajectory.
    # Shifting the associated origin with its lift preserves source errors.
    origin = tuple(2 * mp.pi * _mp(mp, v) for v in _turns())
    errors = tuple(_mp(mp, Q(j % 3 - 1) * ETA) for j in range(18))
    shift = mp.pi / 2
    for a, b, weight in _edges(weights[1]):
        old_difference = origin[b] + errors[b] - origin[a] - errors[a]
        shifted_difference = (
            origin[b] + shift + errors[b] - origin[a] - shift - errors[a]
        )
        assert abs(old_difference - shifted_difference) < mp.mpf("1e-200")
        assert Q(0) < weight <= 1
    for base, error in zip(origin, errors, strict=True):
        assert abs((base + shift + error) - (base + shift) - error) < mp.mpf("1e-200")
    assert sum(_turns()) == 0


def test_same_reciprocal_field_resolves_relative_registration_in_all_rows(mp):
    forms = _center(tuple(Q((j * 7) % 17 - 8, 10**5) for j in range(36)))
    phases = _copy_center(tuple(Q((j * 5) % 11 - 5, 10**5) for j in range(36)))
    shifted = tuple(
        _mp(mp, value) + (mp.pi / 2 if j >= 18 else 0) for j, value in enumerate(phases)
    )
    weights = (_weights(mp, 0), _weights(mp, 1))
    baseline = _field(mp, forms, phases, weights)
    dx, dy = _field(mp, forms, shifted, weights)
    state = _translated_state(forms, phases)
    rates = _new_field(weights)(state)
    for interval, value in zip(rates, dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    gamma = 1 / (1023 * mp.pi)
    q = _mp(mp, -sum(forms[:18]))
    moment = sum(mp.sin(shifted[j + 18] - shifted[j]) for j in range(18))
    assert abs(-sum(dx[:18]) + 2 * q + gamma * moment) < mp.mpf("1e-200")
    assert abs((sum(dy[18:]) - sum(dy[:18])) / 18 - 2 * gamma * q / 9) < mp.mpf(
        "1e-200"
    )
    assert abs(sum(dx)) < mp.mpf("1e-200")
    assert abs(sum(dy)) < mp.mpf("1e-200")
    assert max(abs(a - b) for a, b in zip(dy, baseline[1], strict=True)) < mp.mpf(
        "1e-200"
    )
    assert abs(sum(dx[:18]) - sum(baseline[0][:18])) > mp.mpf("0.005")
    # This same-input symmetry breaking is distinct from changing the port law.
    whole = tuple(value + _mp(mp, Q(2, 7)) for value in shifted)
    invariant = _field(mp, forms, whole, weights)
    assert max(
        abs(a - b)
        for row, other in zip((dx, dy), invariant, strict=True)
        for a, b in zip(row, other, strict=True)
    ) < mp.mpf("1e-200")


def test_moving_centered_rows_exclude_the_large_relative_mean(mp):
    weights = (_weights(mp, 0), _weights(mp, 1))
    xc = tuple(
        _mp(mp, v) for v in _copy_center(tuple(Q(j % 7 - 3, 10**5) for j in range(36)))
    )
    yc = tuple(
        _mp(mp, v) for v in _copy_center(tuple(Q(j % 5 - 2, 10**5) for j in range(36)))
    )
    common = (-mp.mpf("0.002"),) * 18 + (mp.mpf("0.002"),) * 18
    x = tuple(a + b for a, b in zip(xc, common, strict=True))
    hx, hc = _h_apply(mp, x, weights), _h_apply(mp, xc, weights)
    gc = _exchange(xc)
    assert max(abs(a - b) for a, b in zip(hx, hc, strict=True)) < mp.mpf("1e-200")
    assert max(
        abs(a - b) for a, b in zip(_copy_center(_exchange(x)), gc, strict=True)
    ) < mp.mpf("1e-200")
    for psi in (mp.mpf(0), mp.pi / 2, mp.pi):
        phases = tuple(value + (psi if j >= 18 else 0) for j, value in enumerate(yc))
        dx, dy = _field(mp, x, phases, weights)
        cdx, cdy = _copy_center(dx), _copy_center(dy)
        no_common = _field(mp, xc, phases, weights)
        assert max(
            abs(a - b)
            for row, other in zip((cdx, cdy), map(_copy_center, no_common), strict=True)
            for a, b in zip(row, other, strict=True)
        ) < mp.mpf("1e-200")
        assert _dot(xc, cdx) + _dot(yc, cdy) <= 4 * _mp(mp, G) * (
            _dot(xc, xc) + _dot(yc, yc)
        )
        assert _dot(cdy, cdy) <= 16 * _mp(mp, G) ** 2 * _dot(xc, xc)
        paired = tuple(mp.sin(phases[j + 18] - phases[j]) for j in range(18))
        boundary = _copy_center(paired + tuple(-v for v in paired))
        assert _dot(boundary, boundary) <= 4 * _dot(yc, yc)
    # At a uniform quarter-turn, centered phase force vanishes although the
    # common form force is nonzero. The moving centers are not two gauges.
    zero = (mp.mpf(0),) * 36
    phases = (mp.mpf(0),) * 18 + (mp.pi / 2,) * 18
    dx, dy = _field(mp, zero, phases, weights)
    assert max(abs(v) for v in _copy_center(dx) + _copy_center(dy)) < mp.mpf("1e-200")
    assert abs(sum(dx[:18])) > mp.mpf("0.005")
    assert mp.cos(mp.pi) < 0  # No global boundary-convexity premise was used.


def test_quarter_turn_moment_prices_centered_state_and_mean_uncertainty(mp):
    zeta = tuple(
        _mp(mp, v) for v in _center(tuple(Q(j % 5 - 2, 10**4) for j in range(18)))
    )
    centered_square = sum(value * value for value in zeta) / 2
    for epsilon in (mp.mpf(0), mp.mpf("0.001"), -mp.mpf("0.001")):
        psi = mp.pi / 2 + epsilon
        moment = sum(mp.sin(psi + value) for value in zeta)
        remainder = moment - 18 * mp.sin(psi)
        assert abs(remainder) <= centered_square
        assert 0 <= 18 - moment <= 9 * epsilon**2 + centered_square
        assert abs(moment - 18 * psi) > 1  # The near-zero linear chart fails.
    # Larger-class static witness: equal Q and psi need not determine M.
    # It does not assert that these are two acquired source endpoints.
    a = mp.mpf("0.001")
    moment_flat = 18 * mp.sin(mp.pi / 2)
    moment_spread = 16 + 2 * mp.cos(a)
    assert 0 < moment_flat - moment_spread <= a**2
    assert moment_flat != moment_spread
    origins = (-DELTA / 2,) * 9 + (DELTA / 2,) * 9
    assert sum(origins) == 0
    means = (-3 * ETA, 3 * ETA)
    assert means[1] - means[0] == 6 * ETA
    # Translation is exact; only the inherited mean uncertainty remains.
    assert 18 * abs(means[1] - means[0]) == 108 * ETA


def test_same_reference_storage_retains_quarter_turn_switching_work(mp, report):
    x = tuple(
        _mp(mp, v)
        for v in _center(tuple(Q((j * 5) % 17 - 8, 10**5) for j in range(36)))
    )
    yc = tuple(
        _mp(mp, v)
        for v in _copy_center(tuple(Q((j * 3) % 11 - 5, 10**5) for j in range(36)))
    )
    epsilon = 6 * _mp(mp, ETA)
    y = tuple(
        value + (mp.pi / 2 + epsilon if j >= 18 else 0) for j, value in enumerate(yc)
    )
    weights = (_weights(mp, 0), _weights(mp, 1))
    hx, gx = _h_apply(mp, x, weights), _exchange(x)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    cross = tuple(mp.sin(y[j + 18] - y[j]) for j in range(18))
    cross = cross + tuple(-v for v in cross)
    splus = tuple(a + 9 * b for a, b in zip(_sine(mp, y), cross, strict=True))
    dx, dy = _field(mp, x, y, weights)
    derivative = _dot(lx, dx) - _dot(splus, dy)
    assert abs(derivative + _dot(lx, lx) / 9) < mp.mpf("1e-200")
    common = _relative_common(x)
    centered = _copy_center(x)
    c = 9 * (mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi))
    potential = sum(1 - mp.cos(y[j + 18] - y[j]) for j in range(18))
    form_jump = 9 * _dot(x, gx) / 2 - c * _dot(common, common)
    split = (9 - c) * _dot(common, common) + 9 * _dot(centered, _exchange(centered)) / 2
    assert abs(form_jump - split) < mp.mpf("1e-200")
    assert 0 <= form_jump <= 9 * _dot(x, x)
    norm = mp.sqrt(_dot(x, x) + _dot(yc, yc))
    allowance = 972 * _mp(mp, ETA) + mp.mpf(9) * norm**3 / 2
    assert abs(9 * potential - 162) <= allowance
    jump = form_jump + 9 * potential
    assert 162 - allowance <= jump <= 162 + allowance + 9 * norm**2
    assert jump > 161
    # A newly subtracted rotated reference would erase this real comparison
    # jump. The source's old internal storage is translation invariant.
    assert abs(9 * potential) > 161
    entry = 4 * QHI + 200 * MATURE
    paid = 972 * ETA + Q(9, 2) * entry**3
    assert report.phase_switching_work_allowance == paid
    assert report.storage_jump_bounds == (162 - paid, 162 + paid + 9 * entry**2)
    assert report.switching_work_certified


def test_centered_identity_and_matching_branches_have_fresh_budgets(report):
    entry = 4 * QHI + 200 * MATURE
    coupling = 4 * G * HORIZON
    upper = entry / (1 - coupling)
    assert upper == entry + coupling * upper
    assert report.centered_entry_norm_upper_bound == entry
    assert report.centered_norm_growth_denominator == 1 - coupling > 0
    assert report.centered_norm_upper_bound == upper < Q(1, 100) < Q(1, 32)
    assert report.phase_drift_upper_bound == 6 * G * HORIZON * upper
    assert (
        report.marked_gap_upper_bound
        == QHI + 300 * MATURE + report.phase_drift_upper_bound
    )
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    initial = (
        report.slow_amplitude_bounds[1] + report.inherited_output_error_upper_bound
    )
    absolute = initial + 9 * G
    phase = 6 * ETA + 2 * G * HORIZON * absolute / 9
    assert report.entry_aggregate_upper_bound == initial
    assert report.aggregate_absolute_upper_bound == absolute
    assert report.initial_relative_phase_error_upper_bound == 6 * ETA
    assert report.relative_phase_deviation_upper_bound == phase
    assert report.boundary_branch_margin == Q(1, 4) - phase - 2 * upper > 0
    assert report.centered_moment_remainder_upper_bound == upper**2
    assert report.phase_moment_deficit_upper_bound == 9 * phase**2 + upper**2
    assert report.centered_identity_certified and report.relative_phase_drift_certified
    assert (
        report.phase_moment_remainder_certified
        and report.registration_covariance_certified
    )


def test_affine_prediction_and_phase_blind_obstruction_pay_original_errors(
    report, record_property
):
    inherited = report.inherited_output_error_upper_bound
    initial = report.slow_amplitude_bounds[1] + inherited
    remainder = report.phase_moment_deficit_upper_bound
    error = inherited * DECAY + G * remainder / 2
    assert report.affine_initializer_error_upper_bound == inherited * DECAY
    assert report.affine_prediction_error_upper_bound == error < NOISE
    assert report.affine_prediction_margin == NOISE - error > 0
    upper = -9 * Q(1, 3300) + (initial + 9 * Q(1, 3300)) * DECAY + G * remainder / 2
    assert report.registered_true_bounds == (-9 * G, upper)
    assert upper < -THRESHOLD
    assert report.blind_prediction_error_lower_bound == -upper > NOISE
    assert report.blind_prediction_margin == -upper - NOISE > 0
    # Formal d is an admitted decay factor; no exponential is evaluated.
    # The affine law uses declared registration in its supplied pressure.
    gamma, q0, d = Q(1, 3300), Q(1, 10**7), DECAY
    predicted = q0 * d - 9 * gamma * (1 - d)
    derivative = -2 * (q0 + 9 * gamma) * d
    assert derivative == -2 * predicted - 18 * gamma
    assert q0 * d >= 0 > predicted
    aligned_effect = report.aligned_scalar_effect_upper_bound
    aligned_full_upper = initial * DECAY
    assert report.aligned_true_bounds == (
        -aligned_effect,
        aligned_full_upper + aligned_effect,
    )
    assert (
        report.reconstructed_aligned_return_upper_bound
        == aligned_full_upper + aligned_effect
    )
    assert (
        report.affine_prediction_certified
        and report.blind_prediction_obstruction_certified
    )
    assert report.certified
    for name in (
        "switch_age",
        "post_duration",
        "switch_time",
        "total_duration",
        "inherited_output_error_upper_bound",
        "initial_relative_phase_error_upper_bound",
        "centered_entry_norm_upper_bound",
        "centered_norm_upper_bound",
        "phase_drift_upper_bound",
        "registration_margin",
        "entry_aggregate_upper_bound",
        "aggregate_absolute_upper_bound",
        "relative_phase_deviation_upper_bound",
        "boundary_branch_margin",
        "phase_moment_deficit_upper_bound",
        "phase_switching_work_allowance",
        "affine_initializer_error_upper_bound",
        "affine_prediction_error_upper_bound",
        "affine_prediction_margin",
        "aligned_scalar_effect_upper_bound",
        "record_separation_margin",
        "true_response_separation_lower_bound",
        "common_prediction_obstruction_margin",
        "registered_threshold_margin",
        "aligned_threshold_margin",
        "blind_prediction_error_lower_bound",
        "blind_prediction_margin",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "storage_jump_bounds",
        "registered_true_bounds",
        "aligned_true_bounds",
        "registered_record_bounds",
        "aligned_record_bounds",
    ):
        record_property(name, json.dumps([str(v) for v in getattr(report, name)]))


def test_independent_records_separate_registrations_without_noise_cancellation(report):
    lower, upper = report.registered_true_bounds
    aligned_lower, aligned_upper = report.aligned_true_bounds
    assert report.registered_record_bounds == (lower - NOISE, upper + NOISE)
    assert report.aligned_record_bounds == (
        aligned_lower - NOISE,
        aligned_upper + NOISE,
    )
    assert report.record_separation_margin == aligned_lower - upper - 2 * NOISE > 0
    true_gap = aligned_lower - upper
    assert report.true_response_separation_lower_bound == true_gap > 2 * NOISE
    assert report.common_prediction_obstruction_margin == true_gap - 2 * NOISE > 0
    assert report.common_prediction_obstruction_certified
    # A single prediction at the midpoint minimizes the maximum error for
    # the closest possible endpoints, and still exceeds the target.
    midpoint = (aligned_lower + upper) / 2
    assert min(abs(midpoint - aligned_lower), abs(midpoint - upper)) > NOISE
    assert report.registered_threshold_margin == -THRESHOLD - upper - NOISE > 0
    assert report.aligned_threshold_margin == THRESHOLD - aligned_upper - NOISE > 0
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    assert report.registered_record_certified and report.aligned_record_certified
    for registered in (lower - NOISE, upper + NOISE):
        for aligned in (aligned_lower - NOISE, aligned_upper + NOISE):
            assert aligned - registered >= report.record_separation_margin
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert exact.registered_record_bounds == report.registered_true_bounds
    assert exact.aligned_record_bounds == report.aligned_true_bounds
    assert exact.record_separation_margin == report.record_separation_margin + 2 * NOISE
    assert exact.certified
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        noisy = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not noisy.observation_allowances_admitted and not noisy.certified
        assert not noisy.fixed_threshold_certified
        assert noisy.affine_prediction_certified and noisy.centered_identity_certified
        assert noisy.switching_work_certified


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_registration_rechecks_source_history_origins_and_consumed_bounds(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "relative_phase_registration_certified", False),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "slow_amplitude_bounds", (Q(0), Q(1))),
        ((), "phase_mean_error_upper_bound", Q(0)),
        ((), "relative_phase_entry_error_upper_bound", Q(0)),
        ((), "initial_relative_phase_mean_assumed_zero", True),
        ((), "switch_age", Q(3375008)),
        ((), "switch_time", Q(5385008)),
        ((), "post_duration", Q(16)),
        ((), "original_clock_horizon", Q(5385008)),
        ((), "original_baseline_retained", False),
        ((), "state_reset_performed", True),
        ((), "phase_sensitive_observation_error_bound", NOISE),
        ((), "clock", "wrong phase-sensitive clock"),
        ((), "internal_divisor", Q(10)),
        ((), "aggregate_equation", "Q_prime=-2*Q"),
        ((), "relative_phase_equation", "psi_prime=0"),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        (("source",), "clock", "wrong renewed clock"),
        (("source",) * 2, "uniform_approximation_certified", False),
        (("source",) * 2, "signed_kernel_certified", False),
        (("source",) * 2, "fast_output_tail_upper_bound", Q(0)),
        (("source",) * 3, "cut_form_error_upper_bound", Q(0)),
        (("source",) * 2, "total_output_error_upper_bound", Q(0)),
        (("source",) * 2, "correlated_initialization_retained", False),
        (("source",) * 4, "uniform_tube_certified", False),
        (("source",) * 4, "phase_convexity_certified", False),
        (("source",) * 4, "bregman_denominator", Q(1)),
        (("source",) * 4, "nominal_norm_upper_bound", Q(0)),
        (("source",) * 4, "uniform_comparison_budget", Q(0)),
        (("source",) * 4, "original_phase_gauges_and_branches_retained", False),
        (("source",) * 5, "selected_mu", Q(2)),
        (("source",) * 5, "actual_dissipation_certified", False),
        (("source",) * 5, "mean_projector_rank", 2),
        (("source",) * 6, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 5, "origin_reference", "shifted boundary reference"),
        (("source",) * 6, "prefix_cut_duration", Q(9999)),
        (("source",) * 6 + ("handoff",), "prefix_duration", Q(2000001)),
        (("source",) * 6 + ("handoff", "prefix"), "clock", "wrong full/cut clock"),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_phase_sensitive_interface", lambda **kwargs: changed
        )
        rejected = _assess()
        assert not rejected.certified
        assert not rejected.registration_covariance_certified
        assert not rejected.centered_identity_certified
    path = ("source",) * 6 + ("handoff", "prefix", "reference")
    reference = (
        source.source.source.source.source.source.source.handoff.prefix.reference
    )
    for original_field, value in (
        ("phase_reset_error_bound", Q(0)),
        ("phase_reset_error_bound", 4 * ETA),
    ):
        changed_original = replace(reference.sources[0], **{original_field: value})
        changed = _replace_at(
            source,
            path,
            sources=(changed_original, reference.sources[1]),
            donor=replace(reference.donor, source=changed_original),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_phase_sensitive_interface", lambda **kwargs: changed
        )
        assert not _assess().registration_covariance_certified
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("original_relative_phase_gap", Q(0)),
        ("recipient_isolated_deadline", Q(10**10 - 1)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("clock", "wrong original clock"),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_phase_sensitive_interface", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_phase_sensitive_interface", lambda **kwargs: source
    )
    registered = owner._derive_registered_phase_probe_geometry()
    for changes in (
        dict(bare_gap_lower_bound=Q(3)),
        dict(spectral_gap_certified=False),
        dict(held_capacities=(Q(0),) + (Q(1),) * 17),
    ):
        altered = replace(
            registered, successor=replace(registered.successor, **changes)
        )
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().certified


def test_unused_parent_verdicts_cannot_replace_current_admission(monkeypatch, report):
    original = report.source
    changed = replace(
        original,
        certified=False,
        fresh_storage_certified=False,
        finite_identity_certified=False,
        collective_prediction_certified=False,
        scalar_effect_bound_certified=False,
        scalar_tolerance_certified=False,
        observation_allowances_admitted=False,
        phase_sensitive_return_certified=False,
        form_only_return_certified=False,
        same_threshold_classification_certified=False,
        same_member_record_overlap_certified=False,
        phase_sensitive_true_bounds=(Q(0), Q(0)),
        phase_sensitive_record_bounds=(Q(0), Q(0)),
        form_only_true_bounds=(Q(0), Q(0)),
        form_only_record_bounds=(Q(0), Q(0)),
        post_norm_upper_bound=Q(0),
        storage_jump_bounds=(Q(0), Q(0)),
        scalar_effect_upper_bound=Q(0),
        status="old_endpoint_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_phase_sensitive_interface", lambda **kwargs: changed
    )
    rebuilt = _assess()
    assert rebuilt.certified and rebuilt.centered_identity_certified
    assert rebuilt.registered_true_bounds == report.registered_true_bounds
    assert rebuilt.aligned_true_bounds == report.aligned_true_bounds
    assert rebuilt.storage_jump_bounds == report.storage_jump_bounds
    assert (
        rebuilt.affine_prediction_error_upper_bound
        == report.affine_prediction_error_upper_bound
    )
    assert original == report.source
    assert original.relative_phase_entry_error_upper_bound == 6 * ETA
    assert original.initial_relative_phase_mean_assumed_zero is False


def test_raw_registration_inputs_fail_before_source_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid registration primitive reached source work")

    monkeypatch.setattr(owner, "_assess_mixed_phase_sensitive_interface", forbidden)
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


def test_fixed_width_immutable_contract_declares_history_translation(report):
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
    assert report.registration_angle_bounds == pi_interval() / 2
    assert (
        report.registration_turns == Q(1, 4) and report.translated_copy == "recipient"
    )
    assert report.switch_age == SWITCH_AGE and report.post_duration == HORIZON
    assert report.activation_time == 2010000
    assert report.switch_time == 5385000 and report.total_duration == 5385008
    assert report.original_clock_horizon * Q(1023, 1024) == report.total_duration
    assert report.scalar_decay_upper_bound == DECAY
    assert report.kernel_mass_upper_bound == Q(1, 2)
    for name in (
        "complete_precontact_history_translated",
        "source_covariance_is_conditional",
        "internal_identity_and_winding_unchanged_by_translation",
        "prior_form_observations_unchanged_by_translation",
        "same_complete_contact_law",
        "existing_field_reused_unchanged",
        "moving_copy_centering_not_fixed_gauges",
        "aligned_counterpart_retains_same_underlying_source",
        "original_baseline_retained",
        "original_lifts_and_error_association_retained",
        "supplied_registration_retained_by_affine_predictor",
        "independent_record_errors",
    ):
        assert getattr(report, name) is True
    for name in (
        "source_errors_translated",
        "original_source_report_mutated",
        "old_global_acute_ball_reused",
        "boundary_convexity_assumed",
        "boundary_phase_clamped",
        "initial_relative_phase_exactly_quarter_turn",
        "all_scalar_predictions_excluded",
        "dynamically_resolved_phase_coordinate_required",
        "exact_two_coordinate_closure",
        "observation_error_cancellation_assumed",
        "all_future_identity_claimed",
        "reference_shift_used_to_erase_switch_work",
        "zero_physical_switching_cost_claimed",
        "later_phase_intervention_performed",
        "state_reset_performed",
        "original_source_reacquired",
        "original_source_replayed",
        "original_source_trajectory_evaluated",
        "registration_selected_after_response",
        "runtime_event_executed",
        "operator_occurrence_derived",
        "exponentials_evaluated",
        "roots_numerically_evaluated",
        "physical_property_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    assert report.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
    assert (
        report.newly_observed_aggregates == 0 and report.required_endpoint_records == 2
    )
    assert report.numerical_response_allowance == 0
    with pytest.raises(FrozenInstanceError):
        report.registration_turns = Q(0)
    for name in (
        "source",
        "horizon",
        "registration_turns",
        "initial_phase",
        "phase_offset",
        "initial_reading",
        "threshold",
        "port_strength",
        "baseline",
        "phase_observation",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
