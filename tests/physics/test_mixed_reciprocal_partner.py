"""Independent carried-state controls for a reacting finite composite pair.

Static full-field points are not asserted to be acquired endpoints. Reference
matrix exponentials do not integrate nonlinear source histories. Only the
original pre-leak readings initialize the carried prediction.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_reciprocal_partner as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
TIME = Q(1, 32)
AMPLITUDE = Q(1, 16)
PREFIX_NORM = Q(9, 512)
JOINT_NORM = Q(1, 40)
MEAN = Q(3, 512)
RESIDUAL = Q(1, 10**5)
NOISE = Q(1, 10**6)
ARITHMETIC = Q(1, 10**35)


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert interval.width < Q(1, 10**30)


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support():
    bins = tuple(int((9 * value) % 9) for value in _turns())
    pairs = {
        (i, j)
        for i in range(18)
        for j in range(i + 1, 18)
        if min((bins[j] - bins[i]) % 9, (bins[i] - bins[j]) % 9) in (1, 2)
        or (i // 9 != j // 9 and bins[i] == bins[j])
    }
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in (0, 1)
            for j in range(9)
        )
    )
    return old, tuple(sorted(pairs - set(old)))


def _weights(side):
    return tuple(Q(3, 4) + Q((i + 2 * side) % 5, 16) for i in range(63))


def _edges(side, weights=None):
    old, new = _support()
    weights = _weights(side) if weights is None else weights
    return tuple((18 * side + a, 18 * side + b, Q(1)) for a, b in old) + tuple(
        (18 * side + a, 18 * side + b, w)
        for (a, b), w in zip(new, weights, strict=True)
    )


def _state(forms, errors):
    return tuple(I(x) for x in forms) + tuple(
        2 * pi_interval() * turn + error
        for turn, error in zip(_turns() * 2, errors, strict=True)
    )


def _oracle(mp, forms, errors, weights=None):
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns() * 2, errors, strict=True)
    )
    q, sine = [mp.mpf(0)] * 36, [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(side, None if weights is None else weights[side]):
            gradient = _mp(mp, weight) * (x[a] - x[b])
            current = mp.sin(phase[b] - phase[a])
            q[a] += gradient
            q[b] -= gradient
            sine[a] += current
            sine[b] -= current
    gamma = 1 / (1023 * mp.pi)
    intrinsic = tuple((-a + gamma * b) / 9 for a, b in zip(q, sine))
    inputs = [mp.mpf(0)] * 36
    for j in range(9):
        difference = x[j] - x[18 + j]
        inputs[j], inputs[18 + j] = -difference, difference
    dx = tuple(a + b for a, b in zip(intrinsic, inputs))
    phases = tuple(gamma * value / 9 for value in q)
    return x, tuple(q), tuple(sine), tuple(inputs), dx + phases


def _product(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _series(matrix, time):
    """Signed degree60 matrix polynomial with an absolute norm tail."""
    size = len(matrix)
    term = tuple(tuple(Q(int(i == j)) for j in range(size)) for i in range(size))
    total = term
    scaled = tuple(tuple(time * value for value in row) for row in matrix)
    norm = max(sum(abs(value) for value in row) for row in scaled)
    for n in range(1, 61):
        term = tuple(
            tuple(value / n for value in row) for row in _product(term, scaled)
        )
        total = tuple(
            tuple(a + b for a, b in zip(row, other)) for row, other in zip(total, term)
        )
    return total, norm**61 / factorial(61) / (1 - norm / 62)


def _closed_two(mp, k, leakage, time):
    """exp([[-k-leakage,k],[k,-k]] t), using its two exact modes."""
    k, leakage, time = _mp(mp, k), _mp(mp, leakage), _mp(mp, time)
    half_trace = k + leakage / 2
    root = mp.sqrt(k * k + leakage * leakage / 4)
    common = mp.exp(-half_trace * time)
    cosh, sinh = mp.cosh(root * time), mp.sinh(root * time) / root
    return (
        (common * (cosh - leakage * sinh / 2), common * k * sinh),
        (common * k * sinh, common * (cosh + leakage * sinh / 2)),
    )


def _closed_gain(mp, k):
    prefix = _closed_two(mp, k, 1, TIME)
    contact = _closed_two(mp, k, 2, TIME)
    incoming = _matvec(prefix, (1, -1))
    outgoing = _matvec(contact, incoming)
    # First-copy mass loss is one half of the antisymmetric mass loss.
    return mp.mpf(9) / 2 * (sum(incoming) - sum(outgoing))


def _assess(**changes):
    arguments = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        starting_gap_upper_bounds=(DELTA / 8, DELTA / 8),
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_reciprocal_partner(**arguments)


def _predict(**changes):
    arguments = dict(
        regional_mean_readings=(MEAN, -MEAN),
        regional_mean_error_bounds=(NOISE, NOISE),
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._predict_mixed_reciprocal_transfer(**arguments)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("reciprocal controls cannot advance or replay a source")

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
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def field():
    return owner._mixed_reciprocal_partner_field(
        first_new_edge_conductances=_weights(0),
        second_new_edge_conductances=_weights(1),
    )


def test_full_72_rows_conserve_joint_form_with_finite_partner_backreaction(mp, field):
    forms = tuple(Q((7 * i) % 13 - 6, 4096) for i in range(36))
    errors = tuple(Q((3 * i) % 11 - 5, 8192) for i in range(36))
    x, q, sine, inputs, expected = _oracle(mp, forms, errors)
    actual = field(_state(forms, errors))
    assert len(actual) == 72 and len(_edges(0) + _edges(1)) == 162
    for interval, value in zip(actual, expected, strict=True):
        _contains(mp, interval, value)
    _contains(mp, sum(actual[:36], I(0)), mp.mpf(0))
    for side in (0, 1):
        _contains(mp, sum(actual[36 + 18 * side : 54 + 18 * side], I(0)), mp.mpf(0))
    outgoing = sum(x[j] - x[18 + j] for j in range(9))
    assert outgoing != 0
    _contains(mp, -sum(actual[:18], I(0)), outgoing)
    _contains(mp, sum(actual[18:36], I(0)), outgoing)
    assert all(inputs[j] == -inputs[18 + j] for j in range(9))
    assert inputs[9:18] == inputs[27:36] == (mp.mpf(0),) * 9
    assert max(abs(v) for v in q) > 0 and max(abs(v) for v in sine) > 0
    assert any(value != 0 for value in expected[54:])
    # Same held tau clock and d9 in both copies: the exchange is supplied
    # form input, not a cross-phase edge or a rebuilt degree10 graph.


def test_exchange_dissipates_joint_norm_but_not_necessarily_gradient_storage(mp):
    z = (Q(1, 512),) * 9 + (Q(3, 1024),) * 9
    forms = z + (Q(0),) * 18
    x, q, _, inputs, expected = _oracle(mp, forms, (Q(0),) * 36, ((Q(1),) * 63,) * 2)
    differences = tuple(x[j] - x[18 + j] for j in range(9))
    norm_work = sum(a * b for a, b in zip(x, inputs))
    assert abs(norm_work + sum(value**2 for value in differences)) < mp.mpf("1e-190")
    assert norm_work < 0
    storage_work = sum(a * b for a, b in zip(q, inputs))
    expected_work = -sum((q[j] - q[18 + j]) * differences[j] for j in range(9))
    assert abs(storage_work - expected_work) < mp.mpf("1e-190")
    assert storage_work > 0
    assert any(value != 0 for value in expected[18:27])
    # This generic static point distinguishes two balances, not a new
    # acquired endpoint. Either copy may receive form from its finite partner.


def test_jets_include_partner_consumption_and_live_phase_rows(mp, field):
    forms = tuple(Q((i % 9) - 4, 4096) for i in range(36))
    errors = tuple(Q((i % 7) - 3, 8192) for i in range(36))
    direction = tuple(Q(int(18 <= i < 27)) for i in range(36))
    state = _state(forms, errors)
    jets = tuple(
        Jet((value, I(d))) for value, d in zip(state, direction + (Q(0),) * 36)
    )
    gradient = [Q(0)] * 36
    for a, b, w in _edges(0) + _edges(1):
        current = w * (direction[a] - direction[b])
        gradient[a] += current
        gradient[b] -= current
    input_derivative = [Q(0)] * 36
    for j in range(9):
        input_derivative[j], input_derivative[18 + j] = Q(1), -Q(1)
    gamma = 1 / (1023 * mp.pi)
    derivatives = tuple(
        _mp(mp, -value / 9 + added) for value, added in zip(gradient, input_derivative)
    )
    derivatives += tuple(gamma * _mp(mp, value) / 9 for value in gradient)
    expected = _oracle(mp, forms, errors)[-1]
    for result, value, derivative in zip(
        field(jets), expected, derivatives, strict=True
    ):
        _contains(mp, result.coeffs[0], value)
        _contains(mp, result.coeffs[1], derivative)
    assert derivatives[:9] == (mp.mpf(1),) * 9
    assert all(value == 0 for value in derivatives[36:54])
    assert any(value != 0 for value in derivatives[54:])


def test_reference_modes_and_signed_chained_gain_are_independent(
    mp, report, record_property
):
    reference = report.reference
    k = reference.prefix.reference_coupling
    matrix = (
        (-k - 1, k, Q(1), Q(0)),
        (k, -k, Q(0), Q(0)),
        (Q(1), Q(0), -k - 1, k),
        (Q(0), Q(0), k, -k),
    )
    assert reference.generator == matrix
    assert all(sum(row) == 0 for row in matrix)
    assert all(sum(column) == 0 for column in zip(*matrix))
    transform = ((1, 0, 1, 0), (0, 1, 0, 1), (1, 0, -1, 0), (0, 1, 0, -1))
    modes = ((-k, k, 0, 0), (k, -k, 0, 0), (0, 0, -k - 2, k), (0, 0, k, -k))
    assert _product(transform, matrix) == _product(modes, transform)
    common = _closed_two(mp, k, 0, TIME)
    difference = _closed_two(mp, k, 2, TIME)
    for i in range(4):
        for j in range(4):
            sign = 1 if i // 2 == j // 2 else -1
            expected = (common[i % 2][j % 2] + sign * difference[i % 2][j % 2]) / 2
            assert (
                _mp(mp, reference.exponential.lower[i][j])
                <= expected
                <= _mp(mp, reference.exponential.upper[i][j])
            )
    prefix_matrix = ((-1 - k, k), (k, -k))
    difference_matrix = ((-2 - k, k), (k, -k))
    prefix, prefix_tail = _series(prefix_matrix, TIME)
    contact, contact_tail = _series(difference_matrix, TIME)
    incoming = _matvec(prefix, (1, -1))
    outgoing = _matvec(contact, incoming)
    polynomial_gain = Q(9, 2) * (sum(incoming) - sum(outgoing))
    gain_error = 9 * (2 * prefix_tail + contact_tail * (1 + prefix_tail))
    assert 0 <= gain_error < Q(1, 10**120)
    for i, bounds in enumerate(reference.prefix_difference_bounds):
        assert (
            bounds[0]
            <= incoming[i] - 2 * prefix_tail
            <= incoming[i] + 2 * prefix_tail
            <= bounds[1]
        )
    gain = _closed_gain(mp, k)
    lower = 9 * TIME * (1 - 2 * TIME - 3 * TIME / 2)
    assert _mp(mp, lower) <= gain <= _mp(mp, 9 * TIME)
    for side, coefficient in enumerate(reference.response_coefficient_bounds):
        sign = 1 if side == 0 else -1
        assert (
            coefficient[0]
            <= sign * polynomial_gain - gain_error
            <= sign * polynomial_gain + gain_error
            <= coefficient[1]
        )
        assert _mp(mp, coefficient[0]) <= sign * gain <= _mp(mp, coefficient[1])
    assert reference.response_coefficient_bounds[0][0] > 0
    assert reference.response_coefficient_bounds[1][1] < 0
    assert reference.reference_certified
    record_property("chained_reference_gain_decimal", mp.nstr(gain, 22))
    record_property("output_orientation", "outgoing_from_first_copy_to_second")


def test_prefix_carries_both_means_common_form_and_nonzero_initialization_error(mp):
    prediction = _predict(
        regional_mean_readings=(MEAN, -MEAN / 2),
        regional_mean_error_bounds=(NOISE, NOISE / 2),
    )
    k = prediction.reference.prefix.reference_coupling
    prefix = _closed_two(mp, k, 1, TIME)
    column = _matvec(prefix, (1, -1))
    initial = tuple(
        value * _mp(mp, mean) for mean in (MEAN, -MEAN / 2) for value in column
    )
    for i, value in enumerate(initial):
        bounds = prediction.carried_reference_bounds[i]
        assert _mp(mp, bounds[0]) <= value <= _mp(mp, bounds[1])
        radius = (bounds[1] - bounds[0]) / 2
        expected_error = (
            prediction.regional_mean_error_bounds[i // 2] + TIME * RESIDUAL + radius
        )
        assert prediction.carried_state_error_bounds[i] == expected_error > 0
        center = (bounds[0] + bounds[1]) / 2
        assert prediction.carried_mean_centers[i] == center
        assert prediction.carried_mean_bounds[i] == (
            center - expected_error,
            center + expected_error,
        )
    assert sum(initial[:2]) < 0 < sum(initial[2:])
    assert prediction.carried_mean_centers[0] + prediction.carried_mean_centers[1] != 0
    assert prediction.carried_mean_centers[0] != MEAN
    common = _closed_two(mp, k, 0, TIME)
    difference = _closed_two(mp, k, 2, TIME)
    plus = _matvec(common, (initial[0] + initial[2], initial[1] + initial[3]))
    minus = _matvec(difference, (initial[0] - initial[2], initial[1] - initial[3]))
    after = tuple((plus[i] + sign * minus[i]) / 2 for sign in (1, -1) for i in range(2))
    for value, bounds, actual in zip(
        after,
        prediction.contact_reference_bounds,
        prediction.contact_mean_bounds,
        strict=True,
    ):
        assert _mp(mp, bounds[0]) <= value <= _mp(mp, bounds[1])
        assert actual == (
            bounds[0] - NOISE - 2 * TIME * RESIDUAL,
            bounds[1] + NOISE + 2 * TIME * RESIDUAL,
        )
    zero = _predict(
        regional_mean_readings=(0, 0),
        regional_mean_error_bounds=(0, 0),
        readout_error_bound=0,
    )
    assert zero.carried_reference_bounds == ((Q(0), Q(0)),) * 4
    assert zero.carried_state_error_bounds == (TIME * RESIDUAL,) * 4
    assert (
        zero.contact_mean_bounds == ((-2 * TIME * RESIDUAL, 2 * TIME * RESIDUAL),) * 4
    )
    assert zero.forecast == 0 and zero.record_prediction_error_upper_bound > 0


def test_joint_norm_residual_and_registration_cover_both_reacting_sources(
    report, record_property
):
    assert report.certified and report.status == "certified"
    assert (
        report.source_certified
        == report.finite_protocol_ready_ages_exist
        == report.conditional_ready_sources
        == (True, True)
    )
    assert Q(17, 12) ** 2 > 2
    exponent = 2 * Q(1, 3000) * TIME
    joint = Q(17, 12) * PREFIX_NORM / (1 - exponent)
    assert exponent == Q(1, 48000)
    assert report.joint_norm_upper_bound == joint < JOINT_NORM < Q(1, 32)
    assert report.joint_norm_margin == JOINT_NORM - joint > 0
    assert (
        report.full_state_tube_certified
        and report.constituent_identity_retained_certified
    )
    components = (
        Q(10, 9) * JOINT_NORM / 3000,
        Q(10, 9) * JOINT_NORM * 10**9 * ETA,
        Q(2, 3) * JOINT_NORM * report.reference.prefix.coupling_radius,
    )
    assert all(value > 0 for value in components)
    assert (
        sum(components)
        == report.reference.contact_residual_calculated_upper_bound
        < RESIDUAL
    )
    assert (
        report.marked_gap_change_upper_bound
        == 4 * Q(1, 3000) * Q(1, 32) * (3 * TIME)
        == Q(1, 256000)
    )
    assert report.retained_gap_upper_bounds == (DELTA / 8 + Q(1, 256000),) * 2
    assert min(report.registration_margins) > 0
    assert report.contact_storage_work_upper_bound == 36 * JOINT_NORM**2 * TIME
    assert report.supplied_signed_charge_amount == 0
    assert report.supplied_absolute_charge_amount_upper_bound == Q(9, 128)
    assert (
        report.full_state_dimension,
        report.reference_state_dimension,
        report.original_regional_readings,
        report.new_contact_entry_readings,
    ) == (72, 4, 2, 0)
    assert report.contact_pairs == tuple((j, 18 + j) for j in range(9))
    assert report.contact_form_channels == 18
    record_property("joint_norm_upper_bound", str(joint))
    record_property(
        "registered_gap_upper_bound_per_copy", str(report.retained_gap_upper_bounds[0])
    )


def test_original_reading_noise_and_inherited_model_error_cover_signed_outputs(
    mp, report, record_property
):
    gain = _closed_gain(mp, report.reference.prefix.reference_coupling)
    model = 18 * TIME**2 * RESIDUAL + 9 * TIME**2 * RESIDUAL
    for means in ((MEAN, -MEAN), (-MEAN, MEAN), (Q(0), Q(0)), (MEAN, MEAN)):
        for errors in ((-NOISE, NOISE), (NOISE, -NOISE)):
            readings = tuple(a + b for a, b in zip(means, errors))
            prediction = _predict(regional_mean_readings=readings)
            assert prediction.certified and all(prediction.mean_observation_consistent)
            assert prediction.inherited_error_upper_bound == 18 * TIME * (
                NOISE + TIME * RESIDUAL
            )
            assert prediction.contact_error_upper_bound == 9 * TIME**2 * RESIDUAL
            assert prediction.arithmetic_error_upper_bound <= ARITHMETIC
            assert prediction.record_prediction_error_upper_bound < Q(2, 10**6)
            for residual in (-model, model):
                if means[0] == means[1]:
                    # The reference difference is exactly zero; preserve
                    # rational corners instead of adding rounded mp values.
                    actual = residual
                    assert (
                        prediction.true_transfer_bounds[0]
                        <= actual
                        <= prediction.true_transfer_bounds[1]
                    )
                    for readout in (-NOISE, NOISE):
                        recorded = actual + readout
                        assert (
                            prediction.recorded_transfer_bounds[0]
                            <= recorded
                            <= prediction.recorded_transfer_bounds[1]
                        )
                        assert (
                            abs(recorded - prediction.forecast)
                            <= prediction.record_prediction_error_upper_bound
                        )
                else:
                    actual = gain * _mp(mp, means[0] - means[1]) + _mp(mp, residual)
                    assert (
                        _mp(mp, prediction.true_transfer_bounds[0])
                        <= actual
                        <= _mp(mp, prediction.true_transfer_bounds[1])
                    )
                    for readout in (-NOISE, NOISE):
                        recorded = actual + _mp(mp, readout)
                        assert (
                            _mp(mp, prediction.recorded_transfer_bounds[0])
                            <= recorded
                            <= _mp(mp, prediction.recorded_transfer_bounds[1])
                        )
                        assert abs(recorded - _mp(mp, prediction.forecast)) <= _mp(
                            mp, prediction.record_prediction_error_upper_bound
                        )
    nonnumerical = (
        18 * TIME * (NOISE + TIME * RESIDUAL) + 9 * TIME**2 * RESIDUAL + NOISE
    )
    assert nonnumerical == Q(187, 102400000)
    assert (
        report.record_prediction_error_upper_bound
        == nonnumerical + report.uniform_arithmetic_error_upper_bound
        < Q(2, 10**6)
    )
    assert report.positive_information_certified
    record_property(
        "carried_forecast_error_upper_bound_decimal",
        mp.nstr(_mp(mp, report.record_prediction_error_upper_bound), 22),
    )
    record_property("prediction_target", str(Q(2, 10**6)))


def test_partner_actual_histories_have_a_uniform_record_gap_without_extra_data(
    report, record_property
):
    leading = (
        2 * AMPLITUDE * TIME * (1 - TIME / 2) * 9 * TIME * (1 - 2 * TIME - 3 * TIME / 2)
    )
    assert leading == Q(32319, 33554432) > Q(19, 20000)
    loss = 2 * (18 * TIME * (2 * TIME) + 9 * TIME**2) * RESIDUAL
    assert loss + 2 * NOISE < Q(3, 10**6)
    assert report.partner_history_leading_gap == leading
    assert report.partner_history_model_loss_upper_bound == loss
    assert report.actual_partner_history_gap_lower_bound == leading - loss
    assert (
        report.recorded_partner_history_gap_lower_bound
        == leading - loss - 2 * NOISE
        > Q(947, 10**6)
        > Q(9, 10**4)
    )
    assert (
        report.weak_prediction_error_lower_bound
        == report.recorded_partner_history_gap_lower_bound / 2
        > Q(2, 10**6)
    )
    assert report.partner_history_obstruction_certified
    assert (
        report.negative_orientation
        == "Q_with_negative_partner_charge_minus_Q_with_positive_partner_charge"
    )
    # Same first actual history and same second source precede the two
    # supplied partner inputs. No equality of their phase histories, their
    # post-leak global means, or their old transfers is asserted or observed.
    record_property(
        "recorded_partner_history_gap_lower_bound",
        str(report.recorded_partner_history_gap_lower_bound),
    )
    record_property(
        "weak_prediction_error_lower_bound",
        str(report.weak_prediction_error_lower_bound),
    )


def test_both_sources_are_rebuilt_without_using_earlier_response_verdicts(
    monkeypatch, report
):
    first, second = report.prefix_reports

    def install(pair):
        remaining = iter(pair)
        monkeypatch.setattr(
            owner, "_assess_mixed_regional_boundary", lambda **kwargs: next(remaining)
        )

    ignored = tuple(
        replace(
            prefix,
            certified=False,
            positive_information_certified=False,
            global_history_obstruction_certified=False,
        )
        for prefix in (first, second)
    )
    install(ignored)
    assert _assess().certified
    for side in (0, 1):
        for replacement in (
            replace(report.prefix_reports[side], form_error_bound=0),
            replace(
                report.prefix_reports[side],
                norm_tube_upper_bound=PREFIX_NORM + Q(1, 10**100),
            ),
            replace(
                report.prefix_reports[side],
                regions=(tuple(range(9, 18)), tuple(range(9))),
            ),
            replace(
                report.prefix_reports[side],
                edge_weight_deviation_upper_bound=10**9 * ETA + Q(1, 10**100),
            ),
        ):
            pair = tuple(
                replacement if i == side else report.prefix_reports[i] for i in range(2)
            )
            install(pair)
            unavailable = _assess()
            assert not unavailable.source_certified[side] and not unavailable.certified
        no_existence = replace(
            report.prefix_reports[side], finite_protocol_ready_age_exists=False
        )
        install(
            tuple(
                no_existence if i == side else report.prefix_reports[i]
                for i in range(2)
            )
        )
        conditional = _assess()
        assert all(conditional.conditional_ready_sources)
        assert (
            not conditional.finite_protocol_ready_ages_exist[side]
            and not conditional.certified
        )


def test_caps_readiness_and_reading_consistency_do_not_authenticate_or_clamp():
    touching = _predict(regional_mean_readings=(MEAN + NOISE, -MEAN - NOISE))
    assert touching.certified and touching.regional_mean_reading_bounds == (
        (MEAN, MEAN + 2 * NOISE),
        (-MEAN - 2 * NOISE, -MEAN),
    )
    for side in (0, 1):
        readings = tuple(
            MEAN + NOISE + Q(1, 10**100) if i == side else 0 for i in range(2)
        )
        inconsistent = _predict(regional_mean_readings=readings)
        assert (
            not inconsistent.mean_observation_consistent[side]
            and not inconsistent.certified
        )
        outside = _predict(
            regional_mean_error_bounds=tuple(
                NOISE + Q(1, 10**100) if i == side else NOISE for i in range(2)
            )
        )
        assert not outside.allowances_admitted and not outside.certified
        gap = _assess(
            starting_gap_upper_bounds=tuple(
                DELTA / 8 + Q(1, 10**100) if i == side else DELTA / 8 for i in range(2)
            )
        )
        assert gap.source_certified[side] and gap.finite_protocol_ready_ages_exist[side]
        assert not gap.conditional_ready_sources[side] and not gap.certified
        source = _assess(
            form_error_bounds=tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
        )
        assert not source.source_certified[side] and not source.certified
    noisy = _assess(readout_error_bound=NOISE + Q(1, 10**100))
    assert noisy.constituent_identity_retained_certified and not noisy.certified
    zero = _assess(starting_gap_upper_bounds=(0, 0))
    assert all(zero.conditional_ready_sources) and all(
        zero.finite_protocol_ready_ages_exist
    )
    assert zero.existence_gap_threshold == DELTA / 8 > 0
    # Existence remains the positive-threshold assertion, not a certificate
    # that either actual source ever reaches the supplied zero gap in finite time.


def test_arithmetic_allowance_covers_both_exponentials_and_signed_subtraction(
    monkeypatch,
):
    original = owner.metzler_exponential_bounds

    def widened(*args, **kwargs):
        result = original(*args, **kwargs)
        amount = Q(1, 10**25)
        lower = tuple(
            tuple(
                value - (amount if (i, j) == (0, 0) else 0)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(result.lower)
        )
        upper = tuple(
            tuple(
                value + (amount if (i, j) == (0, 0) else 0)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(result.upper)
        )
        return replace(result, lower=lower, upper=upper)

    monkeypatch.setattr(owner, "metzler_exponential_bounds", widened)
    prediction = _predict()
    assert prediction.reference.reference_certified
    assert prediction.record_prediction_error_upper_bound < Q(2, 10**6)
    assert prediction.arithmetic_error_upper_bound > ARITHMETIC
    assert not prediction.arithmetic_certified and not prediction.certified
    assert "arithmetic_allowance_unavailable" in prediction.unresolved_conditions


def test_malformed_two_sources_and_original_readings_precede_reference_work(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed primitive reached source, field or reference work")

    for name in (
        "_assess_mixed_regional_boundary",
        "_regional_reference",
        "_coupling_successor_field",
        "metzler_exponential_bounds",
        "I",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    malformed = (
        True,
        np.bool_(False),
        "0",
        float("nan"),
        float("inf"),
        complex(0, 1),
        None,
    )
    for value in malformed:
        for side in (0, 1):
            pair = tuple(value if i == side else Q(0) for i in range(2))
            for key in ("regional_mean_readings", "regional_mean_error_bounds"):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _predict(**{key: pair})
            for key in (
                "form_error_bounds",
                "phase_error_bounds",
                "starting_gap_upper_bounds",
            ):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{key: pair})
        for consumer in (_predict, _assess):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                consumer(readout_error_bound=value)
    for key in ("regional_mean_readings", "regional_mean_error_bounds"):
        for vector in ((Q(0),), (Q(0),) * 3, repeat(Q(0)), {Q(0), Q(1)}, {0: 0, 1: 0}):
            with pytest.raises((TypeError, ValueError)):
                _predict(**{key: vector})
    for key in ("form_error_bounds", "phase_error_bounds", "starting_gap_upper_bounds"):
        for vector in ((Q(0),), (Q(0),) * 3, repeat(Q(0)), {Q(0), Q(1)}, {0: 0, 1: 0}):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{key: vector})
        with pytest.raises(ValueError):
            _assess(**{key: (Q(0), -Q(1, 10**100))})
    with pytest.raises(ValueError):
        _predict(regional_mean_error_bounds=(-Q(1, 10**100), 0))
    for consumer in (_predict, _assess):
        with pytest.raises(ValueError):
            consumer(readout_error_bound=-Q(1, 10**100))


def test_no_handoff_state_or_reserved_transfer_is_a_decoder_input():
    for key in (
        "handoff_state",
        "handoff_regional_readings",
        "previous_transfer",
        "future_global_mean",
        "partner_waveform",
        "actual_charge_history",
        "source_report",
        "elapsed_time",
        "reference",
        "equilibrium",
    ):
        with pytest.raises(TypeError):
            _predict(**{key: 0})
    for key in ("ready_ages", "source_reports", "contact_duration", "equilibrium"):
        with pytest.raises(TypeError):
            _assess(**{key: 0})


def test_both_weight_vectors_are_admitted_before_either_field_factory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an invalid weight vector reached a compiled field")

    monkeypatch.setattr(owner, "_coupling_successor_field", forbidden)
    for side in (0, 1):
        invalid = (
            _weights(side)[:-1],
            _weights(side) + (Q(1),),
            repeat(Q(1)),
            (True,) + _weights(side)[1:],
            (Q(3, 4) - Q(1, 10**100),) + _weights(side)[1:],
            (Q(1) + Q(1, 10**100),) + _weights(side)[1:],
        )
        for values in invalid:
            arguments = dict(
                first_new_edge_conductances=_weights(0),
                second_new_edge_conductances=_weights(1),
            )
            arguments[
                ("first_new_edge_conductances", "second_new_edge_conductances")[side]
            ] = values
            with pytest.raises((TypeError, ValueError)):
                owner._mixed_reciprocal_partner_field(**arguments)


def test_full_state_admission_intervals_and_fixed_width_vectors_remain_associated(
    mp, field, report
):
    state = _state((Q(0),) * 36, (Q(0),) * 36)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid late state coordinate reached first-copy sine work")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flow_owner, "sin", forbidden)
        patch.setattr(flow_owner, "jet_sin", forbidden)
        for invalid in (
            state[:-1],
            state + (I(0),),
            repeat(I(0)),
            (True,) + state[1:],
            state[:-1] + (float("nan"),),
            (Jet((I(0), I(1))),) + state[1:],
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                field(invalid)
        jets = tuple(Jet((value, I(0))) for value in state)
        with pytest.raises((TypeError, ValueError)):
            field(jets[:-1] + (Jet((I(0), I(0), I(1))),))
    width = Q(1, 10**12)
    interval_state = state[:18] + (I(-width, width),) + state[19:]
    enclosed = field(interval_state)
    for endpoint in (-width, width):
        forms = (Q(0),) * 18 + (endpoint,) + (Q(0),) * 17
        expected = _oracle(mp, forms, (Q(0),) * 36)[-1]
        for interval, value in zip(enclosed, expected, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert enclosed[0].lo <= -width <= width <= enclosed[0].hi

    def fixed(value):
        return Q(np.int64(Q(value).numerator), np.int64(Q(value).denominator))

    one_shot = owner._mixed_reciprocal_partner_field(
        first_new_edge_conductances=(fixed(x) for x in _weights(0)),
        second_new_edge_conductances=(fixed(x) for x in _weights(1)),
    )
    assert one_shot(value for value in state) == field(state)
    assert (
        _predict(
            regional_mean_readings=(fixed(x) for x in (MEAN, -MEAN)),
            regional_mean_error_bounds=(fixed(NOISE) for _ in range(2)),
            readout_error_bound=fixed(NOISE),
        )
        == _predict()
    )
    assert (
        _assess(
            form_error_bounds=(fixed(ERROR) for _ in range(2)),
            phase_error_bounds=(fixed(ERROR) for _ in range(2)),
            starting_gap_upper_bounds=(fixed(DELTA / 8) for _ in range(2)),
            readout_error_bound=fixed(NOISE),
        )
        == report
    )
    with pytest.raises(FrozenInstanceError):
        report.full_state_dimension = 36
    prediction = _predict()
    with pytest.raises(FrozenInstanceError):
        prediction.forecast = Q(0)
