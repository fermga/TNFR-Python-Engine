"""Independent two-channel controls with unknown, independently prepared states.

Generic row and phase points are law controls, not acquired endpoints. The
finite discriminator separates entire recorded-response sets. It requires
new full-norm maturity and does not derive it from older mean readings.
"""

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
from tnfr.physics import _mixed_regional_boundary as prefix_owner
from tnfr.physics import _mixed_two_channel_circulation as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
MATURE = Q(1, 10**8)
PREFIX = 3 * MATURE / 2
DELTA = Q(1, 1024)
GAP = DELTA / 4
JUMP = Q(1, 8192)
JUMP_ERROR = Q(1, 10**8)
TIME = Q(512)
NORM = Q(1, 1024)
GAMMA = Q(1, 3000)
GAMMA_LO = Q(1, 3300)
NOISE = Q(1, 10**6)
WEIGHT_ERROR = Q(2, 10**60)
VSTAR = Q(1, 250000000)
LOWER = Q(1535, 250000000)
COMPARATOR = Q(51, 4) * PREFIX
THRESHOLD = Q(1, 400000)


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
    for j in range(18):
        difference = x[j] - x[18 + j]
        inputs[j], inputs[18 + j] = -difference, difference
    dx = tuple(a + b for a, b in zip(intrinsic, inputs))
    phases = tuple(gamma * value / 9 for value in q)
    return x, tuple(q), tuple(sine), tuple(inputs), dx + phases


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


def _assess(**changes):
    arguments = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        mature_norm_upper_bounds=(MATURE, MATURE),
        phase_jump_error_bound=JUMP_ERROR,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_two_channel_circulation(**arguments)


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def fields():
    arguments = dict(
        first_new_edge_conductances=_weights(0),
        second_new_edge_conductances=_weights(1),
    )
    return tuple(
        owner._mixed_two_channel_field(**arguments, law=law)
        for law in ("full", "without_form_feedback")
    )


def _partners():
    return tuple(9 + (5 * (j - 4) + 4) % 9 for j in range(9))


def _center(vector):
    mean = sum(vector) / len(vector)
    return tuple(value - mean for value in vector)


def _square(vector):
    return sum(value**2 for value in vector)


def test_complete_72_rows_keep_two_separate_channels_and_live_phase(mp, fields):
    forms = tuple(Q((5 * i) % 13 - 6, 4096) for i in range(36))
    errors = tuple(Q((3 * i) % 11 - 5, 8192) for i in range(36))
    state = _state(forms, errors)
    x, q, sine, inputs, full = _oracle(mp, forms, errors)
    gamma = 1 / (1023 * mp.pi)
    comp = tuple(-a / 9 + b for a, b in zip(q, inputs)) + full[36:]
    actual = tuple(field(state) for field in fields)
    for rows, expected in zip(actual, (full, comp)):
        assert len(rows) == 72
        for interval, value in zip(rows, expected):
            _contains(mp, interval, value)
        _contains(mp, sum(rows[:36], I(0)), mp.mpf(0))
        for side in (0, 1):
            _contains(mp, sum(rows[36 + 18 * side : 54 + 18 * side], I(0)), mp.mpf(0))
    for i in range(72):
        _contains(
            mp,
            actual[0][i] - actual[1][i],
            gamma * sine[i] / 9 if i < 36 else mp.mpf(0),
        )
    channel_a = sum(x[j] - x[18 + j] for j in range(9))
    channel_b = sum(x[j] - x[18 + j] for j in range(9, 18))
    assert abs(channel_b) > 0
    for rows in actual:
        _contains(mp, -sum(rows[:18], I(0)), channel_a + channel_b)
    assert max(abs(value) for value in comp[36:]) > 0
    # A-channel flow is not copy total loss once the B-channel is present.
    assert abs(channel_a - (channel_a + channel_b)) > 0
    changed_phase = tuple(I(Q(i - 18, 32)) for i in range(36))
    assert fields[1](state[:36] + changed_phase) == actual[1]


def test_jets_and_global_form_translation_preserve_the_consumed_coordinates(mp, fields):
    forms = tuple(Q((i % 9) - 4, 4096) for i in range(36))
    errors = tuple(Q((i % 7) - 3, 8192) for i in range(36))
    state = _state(forms, errors)
    gamma = 1 / (1023 * mp.pi)
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns() * 2, errors)
    )
    for coordinate in (9, 45):
        direction = tuple(Q(int(i == coordinate)) for i in range(72))
        jets = tuple(Jet((value, I(d))) for value, d in zip(state, direction))
        gradient = [Q(0)] * 36
        sine = [mp.mpf(0)] * 36
        for a, b, weight in _edges(0) + _edges(1):
            q = weight * (direction[a] - direction[b])
            ds = mp.cos(theta[b] - theta[a]) * _mp(
                mp, direction[36 + b] - direction[36 + a]
            )
            gradient[a] += q
            gradient[b] -= q
            sine[a] += ds
            sine[b] -= ds
        exchanged = tuple(direction[18 + i] - direction[i] for i in range(18))
        exchanged += tuple(-value for value in exchanged)
        comp = tuple(_mp(mp, -q / 9 + f) for q, f in zip(gradient, exchanged))
        comp += tuple(gamma * _mp(mp, q) / 9 for q in gradient)
        full = tuple(comp[i] + gamma * sine[i] / 9 for i in range(36)) + comp[36:]
        for field, expected in zip(fields, (full, comp)):
            for jet, value in zip(field(jets), expected):
                _contains(mp, jet.coeffs[1], value)
        if coordinate == 9:
            assert comp[27] == 1 and max(abs(value) for value in comp[36:]) > 0
        else:
            assert all(value == 0 for value in comp)
            assert max(abs(value) for value in full[:36]) > 0
    translated = tuple(value + Q(5, 16) for value in forms)
    expected = _oracle(mp, forms, errors)[-1]
    expected_comp = _oracle(mp, forms, errors)
    expected_comp = (
        tuple(-q / 9 + f for q, f in zip(expected_comp[1], expected_comp[3]))
        + expected_comp[-1][36:]
    )
    for field, oracle in zip(fields, (expected, expected_comp)):
        for interval, value in zip(field(_state(translated, errors)), oracle):
            _contains(mp, interval, value)
    assert sum(translated) != sum(forms)
    assert all(
        translated[j] - translated[18 + j] == forms[j] - forms[18 + j]
        for j in range(18)
    )


def test_zero_charge_prefix_carries_leak_mean_and_phase_jumps_price_work(mp):
    # Constructed points check the supplied maps and balances. They are
    # not a replacement for the actually matured, unobserved source.
    forms = (MATURE / 32,) * 9 + (-MATURE / 32,) * 9
    forms += (-MATURE / 64,) * 9 + (MATURE / 64,) * 9
    errors = tuple(MATURE * Q((i % 9) - 4, 256) for i in range(36))
    x, q, sine, inputs, full = _oracle(mp, forms, errors)
    intrinsic = tuple(value - added for value, added in zip(full[:36], inputs))
    for side in (0, 1):
        state = _state(forms, errors)
        local = (
            state[18 * side : 18 * (side + 1)] + state[36 + 18 * side : 54 + 18 * side]
        )
        charge = prefix_owner._mixed_regional_boundary_field(
            new_edge_conductances=_weights(side), stage="charge", charge_amplitude=0
        )
        leak = prefix_owner._mixed_regional_boundary_field(
            new_edge_conductances=_weights(side), stage="probe"
        )
        for stage, field in (("charge", charge), ("probe", leak)):
            expected = tuple(
                intrinsic[18 * side + i]
                - (x[18 * side + i] if stage == "probe" and i < 9 else 0)
                for i in range(18)
            )
            expected += full[36 + 18 * side : 54 + 18 * side]
            rows = field(local)
            for interval, value in zip(rows, expected):
                _contains(mp, interval, value)
            mean_rate = (
                -sum(x[18 * side : 18 * side + 9]) / 18
                if stage == "probe"
                else mp.mpf(0)
            )
            _contains(mp, sum(rows[:18], I(0)) / 18, mean_rate)
            if stage == "probe":
                assert mean_rate != 0
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns() * 2, errors)
    )
    for amplitude1, amplitude2 in (
        (JUMP - JUMP_ERROR, JUMP + JUMP_ERROR),
        (JUMP + JUMP_ERROR, JUMP - JUMP_ERROR),
    ):
        jumps = (-amplitude1 / 2,) * 9 + (amplitude1 / 2,) * 9
        jumps += (amplitude2 / 2,) * 9 + (-amplitude2 / 2,) * 9
        assert sum(jumps[:18]) == sum(jumps[18:]) == 0
        for side in (0, 1):
            for a, b in _support()[0]:
                assert jumps[18 * side + a] == jumps[18 * side + b]
        work = sum(
            mp.cos(theta[b] - theta[a])
            - mp.cos(theta[b] - theta[a] + _mp(mp, jumps[b] - jumps[a]))
            for a, b, _ in _edges(0) + _edges(1)
        )
        bound = 54 * PREFIX * (JUMP + JUMP_ERROR) + 81 * (JUMP + JUMP_ERROR) ** 2
        assert abs(work) < _mp(mp, bound)
        assert bound < Q(2, 10**6)
    assert 18 * Q(1, 32) * PREFIX**2 > 0
    # The supplied jump keeps forms/support/weights unchanged. Its angle
    # does not rescale with tau, unlike a continuous original-clock rate.


def test_form_gap_and_storage_keep_global_mean_and_actual_unequal_weights(mp, fields):
    vector = tuple(Q((7 * i) % 17 - 8, 4096) for i in range(36))
    centered = _center(vector)
    means = tuple(sum(centered[18 * p : 18 * (p + 1)]) / 18 for p in (0, 1))
    within = tuple(centered[i] - means[i // 18] for i in range(36))
    assert sum(centered) == 0 and means[1] == -means[0]
    assert _square(centered) == _square(within) + 18 * _square(means)
    internal = sum(
        w * (centered[a] - centered[b]) ** 2 / 9 for a, b, w in _edges(0) + _edges(1)
    )
    exchange = sum((centered[j] - centered[18 + j]) ** 2 for j in range(18))
    assert exchange == 18 * (means[0] - means[1]) ** 2 + sum(
        (within[j] - within[18 + j]) ** 2 for j in range(18)
    )
    assert internal >= _square(within) / 3
    assert internal + exchange >= _square(centered) / 3
    assert 18 * (means[0] - means[1]) ** 2 == 2 * 18 * _square(means)
    forms = vector
    errors = tuple(Q((i % 7) - 3, 8192) for i in range(36))
    x, q, sine, inputs, full = _oracle(mp, forms, errors)
    comp = tuple(-a / 9 + b for a, b in zip(q, inputs)) + full[36:]
    gamma = 1 / (1023 * mp.pi)
    work = sum(a * b for a, b in zip(q, inputs))
    assert abs(
        work + sum((q[j] - q[18 + j]) * (x[j] - x[18 + j]) for j in range(18))
    ) < mp.mpf("1e-190")
    dissipation = -sum(value**2 for value in q) / 9
    for rates, expected in (
        (full, dissipation + work),
        (comp, dissipation + work - gamma * sum(a * b for a, b in zip(q, sine)) / 9),
    ):
        derivative = sum(a * b for a, b in zip(q, rates[:36])) - sum(
            a * b for a, b in zip(sine, rates[36:])
        )
        assert abs(derivative - expected) < mp.mpf("1e-190")
    assert _square(tuple(vector[j] - vector[18 + j] for j in range(18))) > 0
    # The heat estimate is for globally centered forms. A nonzero global
    # mean remains conserved and contributes neither phase motion nor flow.
    assert sum(forms) != 0
    rates = fields[1](_state(forms, errors))
    _contains(mp, sum(rates[:36], I(0)), mp.mpf(0))
    assert Q(51, 4) ** 2 > (3**2) * 18


def test_signed_nonlinear_cross_current_and_actual_weight_projection(mp, report):
    minimum = report.marked_gap_lower_bound
    maximum = report.marked_gap_upper_bound
    assert minimum > Q(99, 100) * JUMP > 0
    errors = [Q(0)] * 36
    mean_square_index = sum(Q(i**2) for i in range(9)) / 9
    for side, sign in ((0, 1), (1, -1)):
        for j, partner in enumerate(_partners()):
            base = (Q(j**2) - mean_square_index) / 10**8
            gap = minimum + (maximum - minimum) * Q(j % 3, 3)
            errors[18 * side + j] = base
            errors[18 * side + partner] = base + sign * gap
        errors[18 * side : 18 * (side + 1)] = _center(
            errors[18 * side : 18 * (side + 1)]
        )
    assert _square(errors) < NORM**2
    _, _, sine, _, _ = _oracle(mp, (Q(0),) * 36, errors)
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns() * 2, errors)
    )
    for side, sign in ((0, 1), (1, -1)):
        for a, b in _support()[1]:
            if a < 9 <= b:
                assert (
                    mp.cos(theta[18 * side + b] - theta[18 * side + a]) > mp.mpf(1) / 10
                )
        current = sign * sum(sine[18 * side : 18 * side + 9])
        assert current >= _mp(mp, Q(45, 10) * minimum)
    forcing = (sum(sine[:9]) - sum(sine[18:27])) / (81 * 1023 * mp.pi)
    assert forcing > _mp(mp, 10 * Q(1, 10) * GAMMA_LO * minimum / 9)
    assert report.signed_cross_current_lower_bound == Q(45, 10) * minimum
    # Independently accumulate a fine weight defect, including cancelling
    # within-region edges. These algebraic vectors are not actual endpoints.
    radius = 6 * GAMMA * NORM
    vector = _center(tuple(radius * Q((i % 5) - 2, 32) for i in range(18)))
    assert _square(vector) < radius**2
    gradient = [Q(0)] * 18
    cross = Q(0)
    for a, b in _support()[1]:
        difference = vector[a] - vector[b]
        deviation = WEIGHT_ERROR * (int(difference > 0) - int(difference < 0))
        current = deviation * difference
        gradient[a] += current
        gradient[b] -= current
        if a < 9 <= b:
            cross += current
    assert sum(gradient[:9]) / 81 == cross / 81 > 0
    assert (cross / 81) ** 2 <= (Q(10, 9) * WEIGHT_ERROR) ** 2 * _square(vector)
    assert cross / 81 < Q(10, 9) * WEIGHT_ERROR * radius
    assert report.paired_form_difference_upper_bound == radius
    assert (
        report.regional_weight_residual_upper_bound == Q(10, 9) * WEIGHT_ERROR * radius
    )


def test_circulation_kernel_is_positive_while_total_output_is_exactly_blind(mp):
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    rate = 2 * (k + 1)
    assert 0 < rate < 3
    for time in (Q(0), Q(1, 32), TIME):
        t = _mp(mp, time)
        decay = mp.exp(-rate * t)
        output = 9 * _mp(mp, VSTAR) * (t / rate - (1 - decay) / rate**2)
        first = 9 * _mp(mp, VSTAR) * (1 - decay) / rate
        second = 9 * _mp(mp, VSTAR) * decay
        assert abs(second + rate * first - 9 * _mp(mp, VSTAR)) < mp.mpf("1e-190")
        if time == 0:
            assert output == first == 0
        else:
            assert output > 0 and first > 0
        if time == TIME:
            assert output > _mp(mp, (3 * TIME - 1) * VSTAR)
    assert LOWER == (3 * TIME - 1) * VSTAR
    # Derive the decoupled total mode without assuming a precise fine
    # carrier. Its forcing cancels in each full copy even for nonlinear S.
    a, b, v = Q(3, 17), Q(-7, 19), Q(2, 23)
    rational_k = Q(13, 27)
    da = -(rational_k + 2) * a + rational_k * b + v
    db = rational_k * a - (rational_k + 2) * b - v
    assert da + db == -2 * (a + b)
    b = -a
    da = -(rational_k + 2) * a + rational_k * b + v
    db = rational_k * a - (rational_k + 2) * b - v
    assert db == -da and da == -2 * (rational_k + 1) * a + v
    assert 9 * a + 9 * b == 0
    # Zero initial paired total stays zero, so Q_A+Q_B cannot distinguish
    # these laws, although the two separately associated flows can differ.


def test_independent_source_record_sets_require_more_than_a_paired_gap(
    report, record_property
):
    assert report.paired_transfer_gap_lower_bound == LOWER
    assert report.comparator_transfer_absolute_upper_bound == COMPARATOR
    assert report.comparator_record_bounds == (-COMPARATOR - NOISE, COMPARATOR + NOISE)
    assert report.full_record_lower_bound == LOWER - COMPARATOR - NOISE
    assert report.record_separation_margin == LOWER - 2 * COMPARATOR - 2 * NOISE > 0
    assert report.classifier_threshold == THRESHOLD
    assert COMPARATOR + NOISE < THRESHOLD < LOWER - COMPARATOR - NOISE
    # Distinct hidden preparations may choose opposite nuisance extremes;
    # actuator errors also vary independently. The family bounds, not a
    # common hidden initial state, make this worst cross-fit comparison safe.
    worst_comparator_record = COMPARATOR + NOISE
    worst_full_record = LOWER - COMPARATOR - NOISE
    assert (
        worst_full_record - worst_comparator_record == report.record_separation_margin
    )
    for comp_value in (-COMPARATOR, Q(0), COMPARATOR):
        for error in (-NOISE, NOISE):
            assert comp_value + error < THRESHOLD
    for full_value in (LOWER - COMPARATOR, LOWER, LOWER + COMPARATOR):
        for error in (-NOISE, NOISE):
            assert full_value + error > THRESHOLD
    # A generic counterexample to paired-gap-only identification: f(s)=s+L,
    # g(s)=s have gapL at every shared s but overlapping sets for wide s.
    nuisance = LOWER
    assert -nuisance + LOWER == 0 == Q(0)
    assert -nuisance <= -LOWER <= nuisance
    assert report.independent_hypotheses_separated and report.certified
    record_property("paired_true_gap_lower_bound", str(LOWER))
    record_property("comparator_absolute_transfer_upper_bound", str(COMPARATOR))
    record_property(
        "independent_record_set_separation", str(report.record_separation_margin)
    )
    record_property("fixed_classifier_threshold", str(THRESHOLD))
    record_property("readout_error_per_record", str(NOISE))
    record_property(
        "maturity_association", "new conditional full norm; no authenticated age"
    )


def test_finite_norm_registration_and_work_budgets_retain_the_full_laws(mp, report):
    prefix_exponent = 2 * GAMMA * Q(1, 16)
    assert 2 / (1 - prefix_exponent) ** 2 < Q(9, 4)
    assert mp.sqrt(2) * mp.exp(_mp(mp, prefix_exponent)) < mp.mpf(3) / 2
    exponent = 2 * GAMMA * TIME
    assert 0 < exponent < 1
    assert mp.exp(_mp(mp, exponent)) < _mp(mp, 1 / (1 - exponent))
    initial = PREFIX + 3 * (JUMP + JUMP_ERROR)
    norm = initial / (1 - exponent)
    assert norm == report.calculated_joint_norm_upper_bound < NORM
    assert report.norm_margin == NORM - norm > 0
    assert report.prefix_norm_upper_bound == PREFIX
    assert report.form_generator_gap_lower_bound == Q(1, 3)
    phase_displacement = 6 * GAMMA * PREFIX + 12 * GAMMA**2 * NORM * TIME
    error = Q(3, 2) * (PREFIX + phase_displacement)
    assert error == report.marked_gap_uncertainty_upper_bound < Q(11, 10**7)
    assert JUMP_ERROR + error < JUMP / 100
    assert report.marked_gap_lower_bound == JUMP - JUMP_ERROR - error
    assert report.marked_gap_upper_bound == JUMP + JUMP_ERROR + error < GAP
    assert report.finite_identity_certified and report.phase_intervention_certified
    assert report.forcing_margin > 0 and report.differential_forcing_lower_bound > VSTAR
    assert report.total_transfer_absolute_upper_bound == 3 * PREFIX
    assert report.returning_channel_margin == LOWER - COMPARATOR - 3 * PREFIX > 0
    assert report.paired_total_transfer_difference == 0
    assert report.prefix_storage_work_upper_bound == 18 * Q(1, 32) * PREFIX**2
    assert (
        report.phase_jump_work_upper_bound
        == 54 * PREFIX * (JUMP + JUMP_ERROR) + 81 * (JUMP + JUMP_ERROR) ** 2
    )
    assert report.contact_storage_work_upper_bound == 36 * TIME * NORM**2 == Q(9, 512)
    assert report.work_certified and report.jump_work_margin > 0
    # Independently check the norm differential inequality using all rows,
    # not the stronger and generally false assertion that storage decreases
    # under the external two-copy input for unequal internal conductances.
    forms = tuple(Q((i % 7) - 3, 65536) for i in range(36))
    errors = tuple(Q((i % 9) - 4, 131072) for i in range(36))
    x, q, _, inputs, full = _oracle(mp, forms, errors)
    comp = tuple(-a / 9 + b for a, b in zip(q, inputs)) + full[36:]
    y = tuple(_mp(mp, value) for value in errors)
    norm_squared = _square(x) + _square(y)
    for rows in (full, comp):
        half_derivative = sum(a * b for a, b in zip(x, rows[:36])) + sum(
            a * b for a, b in zip(y, rows[36:])
        )
        assert half_derivative <= 2 * _mp(mp, GAMMA) * norm_squared
    assert NORM < report.phase_radius == Q(1, 32)


def test_fresh_sources_require_complete_degree_loss_convergence_not_old_readiness(
    monkeypatch, report
):
    first, second = report.sources

    def install(changed_first, changed_second):
        rows = iter((changed_first, changed_second))
        monkeypatch.setattr(
            owner, "_assess_sine_coupling_successor", lambda **kwargs: next(rows)
        )

    # Irrelevant old strength-loss contrast and top-level success are not
    # carried as a substitute for the common D source obligations.
    ignored = tuple(
        replace(
            source,
            certified=False,
            normalization_common_form_separation_certified=False,
        )
        for source in report.sources
    )
    install(*ignored)
    assert _assess().certified
    for side, original in enumerate((first, second)):
        common = original.completion_reports[0]
        for changed in (
            replace(original, conditional_image_certified=False),
            replace(original, edge_weight_deviation_upper_bound=2 * WEIGHT_ERROR),
            replace(
                original,
                completion_reports=(
                    replace(common, asymptotic_convergence_certified=False),
                )
                + original.completion_reports[1:],
            ),
            replace(
                original,
                completion_reports=(
                    replace(common, degree_form_limit_bounds=((Q(1), Q(1)),)),
                )
                + original.completion_reports[1:],
            ),
            replace(original, form_error_bound=Q(0)),
        ):
            install(changed if side == 0 else first, changed if side == 1 else second)
            unavailable = _assess()
            assert not unavailable.conditional_mature_sources[side]
            assert not unavailable.certified
    install(first, second)
    original_geometry = owner._derive_registered_phase_probe_geometry

    def corrupt():
        value = original_geometry()
        return replace(
            value, successor=replace(value.successor, weighted_gap_lower_bound=Q(0))
        )

    monkeypatch.setattr(owner, "_derive_registered_phase_probe_geometry", corrupt)
    assert not _assess().source_association_certified


def test_positive_maturity_and_separate_actuator_and_record_errors_are_obligations():
    for side in (0, 1):
        zero = tuple(Q(0) if i == side else MATURE for i in range(2))
        unavailable = _assess(mature_norm_upper_bounds=zero)
        assert unavailable.source_certified[side]
        assert not unavailable.finite_mature_ages_exist[side]
        assert (
            not unavailable.conditional_mature_sources[side]
            and not unavailable.certified
        )
        outside = tuple(
            MATURE + Q(1, 10**100) if i == side else MATURE for i in range(2)
        )
        unavailable = _assess(mature_norm_upper_bounds=outside)
        assert unavailable.finite_mature_ages_exist[side]
        assert (
            not unavailable.mature_norm_domain_admitted[side]
            and not unavailable.certified
        )
        for name in ("form_error_bounds", "phase_error_bounds"):
            errors = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            unavailable = _assess(**{name: errors})
            assert not unavailable.source_certified[side] and not unavailable.certified
    actuator = _assess(phase_jump_error_bound=JUMP_ERROR + Q(1, 10**100))
    assert actuator.source_association_certified
    assert not actuator.phase_intervention_certified and not actuator.certified
    noisy = _assess(readout_error_bound=NOISE + Q(1, 10**100))
    assert noisy.phase_intervention_certified and noisy.circulation_certified
    assert not noisy.independent_hypotheses_separated and not noisy.certified
    exact = _assess(phase_jump_error_bound=0, readout_error_bound=0)
    assert exact.certified and exact.marked_gap_uncertainty_upper_bound > 0
    assert exact.comparator_transfer_absolute_upper_bound > 0


def test_invalid_source_and_law_primitives_precede_downstream_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed input reached source, geometry, or field compilation")

    for name in (
        "_assess_sine_coupling_successor",
        "_derive_registered_phase_probe_geometry",
        "_mixed_reciprocal_partner_field",
        "_mixed_without_form_feedback_field",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for value in (
        True,
        np.bool_(False),
        "0",
        float("nan"),
        float("inf"),
        complex(0, 1),
        None,
    ):
        for side in (0, 1):
            pair = tuple(value if i == side else Q(0) for i in range(2))
            for name in (
                "form_error_bounds",
                "phase_error_bounds",
                "mature_norm_upper_bounds",
            ):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        for name in ("phase_jump_error_bound", "readout_error_bound"):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: value})
    for name in ("form_error_bounds", "phase_error_bounds", "mature_norm_upper_bounds"):
        for values in (
            (0,),
            (0, 0, 0),
            repeat(Q(0)),
            {0, 1},
            {0: 0, 1: 0},
            (Q(0), -Q(1, 10**100)),
        ):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: values})
    for name in ("phase_jump_error_bound", "readout_error_bound"):
        with pytest.raises(ValueError):
            _assess(**{name: -Q(1, 10**100)})
    for law in (
        True,
        0,
        None,
        "FULL",
        "without_form_feedback ",
        "without_phase_feedback",
    ):
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_two_channel_field(
                first_new_edge_conductances=_weights(0),
                second_new_edge_conductances=_weights(1),
                law=law,
            )


def test_all_weights_and_72_coordinates_are_admitted_before_trigonometry(fields):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid late primitive reached sine arithmetic")

    state = _state((Q(0),) * 36, (Q(0),) * 36)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("sin", "jet_sin"):
            patch.setattr(flow_owner, name, forbidden)
        for field in fields:
            for values in (
                state[:-1],
                state + (I(0),),
                repeat(I(0)),
                (True,) + state[1:],
                state[:-1] + (float("nan"),),
                (Jet((I(0), I(1))),) + state[1:],
            ):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    field(values)
            jets = tuple(Jet((value, I(0))) for value in state)
            with pytest.raises(ValueError):
                field(jets[:-1] + (Jet((I(0), I(0), I(1))),))
        for side in (0, 1):
            for values in (
                _weights(side)[:-1],
                _weights(side) + (Q(1),),
                repeat(Q(1)),
                _weights(side)[:-1] + (True,),
                _weights(side)[:-1] + (Q(3, 4) - Q(1, 10**100),),
                _weights(side)[:-1] + (Q(1) + Q(1, 10**100),),
            ):
                arguments = dict(
                    first_new_edge_conductances=_weights(0),
                    second_new_edge_conductances=_weights(1),
                )
                arguments[
                    ("first_new_edge_conductances", "second_new_edge_conductances")[
                        side
                    ]
                ] = values
                for law in ("full", "without_form_feedback"):
                    with pytest.raises((TypeError, ValueError, OverflowError)):
                        owner._mixed_two_channel_field(**arguments, law=law)


def test_one_shot_fixed_width_metadata_and_no_hidden_preparation_claim(fields, report):
    def fixed(value):
        return Q(np.int64(Q(value).numerator), np.int64(Q(value).denominator))

    normalized = _assess(
        form_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_error_bounds=(fixed(ERROR) for _ in range(2)),
        mature_norm_upper_bounds=(fixed(MATURE) for _ in range(2)),
        phase_jump_error_bound=fixed(JUMP_ERROR),
        readout_error_bound=fixed(NOISE),
    )
    assert normalized == report
    for law, expected in zip(("full", "without_form_feedback"), fields):
        field = owner._mixed_two_channel_field(
            first_new_edge_conductances=(fixed(value) for value in _weights(0)),
            second_new_edge_conductances=(fixed(value) for value in _weights(1)),
            law=law,
        )
        state = _state((Q(0),) * 36, (Q(0),) * 36)
        assert field(value for value in state) == expected(state)
    assert report.maturity_is_independent_condition
    assert not report.maturity_authenticated and not report.preparation_age_evaluated
    assert not report.comparator_own_law_source_acquisition_claimed
    assert report.phase_actuator_spatial_uniformity_assumed
    assert report.independently_observed_channels == 1
    assert report.contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert report.contact_duration == TIME and report.charge_amplitude == 0
    assert report.status == "conditional_circulation_discriminator"
    with pytest.raises(FrozenInstanceError):
        report.maturity_authenticated = True
    for name in (
        "sources",
        "initial_state",
        "regional_mean_readings",
        "actual_transfer",
        "source_age",
        "starting_gap_upper_bounds",
        "phase_clamp",
        "contact_duration",
        "classifier_threshold",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
