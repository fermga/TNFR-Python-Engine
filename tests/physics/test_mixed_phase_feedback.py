"""Independent paired-law controls at the same actual contact-entry state.

Generic static points verify laws, not acquired endpoints. The true-response
record obstruction follows from an exact paired bound, never from overlap of
loose calculated envelopes. No trajectories or source producers are run.
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
from tnfr.physics import _mixed_phase_feedback as owner
from tnfr.physics import _mixed_reciprocal_partner as full_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
DELTA = Q(1, 1024)
TIME = Q(1, 32)
PREFIX_NORM = Q(9, 512)
NORM = Q(1, 40)
GAMMA = Q(1, 3000)
NOISE = Q(1, 10**6)
BOUND = Q(17, 491520000)


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
        starting_gap_upper_bounds=(DELTA / 8, DELTA / 8),
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_phase_feedback(**arguments)


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def fields():
    arguments = dict(
        first_new_edge_conductances=_weights(0),
        second_new_edge_conductances=_weights(1),
    )
    return full_owner._mixed_reciprocal_partner_field(
        **arguments
    ), owner._mixed_without_form_feedback_field(**arguments)


def _form_generator():
    matrix = [[Q(0)] * 36 for _ in range(36)]
    edges = tuple((a, b, w / 9) for a, b, w in _edges(0) + _edges(1))
    edges += tuple((j, 18 + j, Q(1)) for j in range(9))
    for a, b, weight in edges:
        matrix[a][a] -= weight
        matrix[b][b] -= weight
        matrix[a][b] += weight
        matrix[b][a] += weight
    return tuple(tuple(row) for row in matrix)


def _matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def test_full_72_row_ablation_removes_only_the_actual_form_sine_force(mp, fields):
    forms = tuple(Q((5 * i) % 13 - 6, 4096) for i in range(36))
    errors = tuple(Q((3 * i) % 11 - 5, 8192) for i in range(36))
    state = _state(forms, errors)
    x, gradient, sine, inputs, expected_full = _oracle(mp, forms, errors)
    gamma = 1 / (1023 * mp.pi)
    expected_comp = (
        tuple(-q / 9 + f for q, f in zip(gradient, inputs)) + expected_full[36:]
    )
    full, comparator = (field(state) for field in fields)
    assert len(full) == len(comparator) == 72
    for i in range(72):
        _contains(mp, full[i], expected_full[i])
        _contains(mp, comparator[i], expected_comp[i])
        force = gamma * sine[i] / 9 if i < 36 else mp.mpf(0)
        _contains(mp, full[i] - comparator[i], force)
    assert max(abs(value) for value in expected_comp[36:]) > 0
    assert max(abs(value) for value in sine) > 0
    for rows in (full, comparator):
        _contains(mp, sum(rows[:36], I(0)), mp.mpf(0))
        _contains(mp, sum(rows[36:54], I(0)), mp.mpf(0))
        _contains(mp, sum(rows[54:], I(0)), mp.mpf(0))
        _contains(mp, -sum(rows[:18], I(0)), sum(x[j] - x[18 + j] for j in range(9)))
    changed_phase = tuple(I(Q(i - 18, 32)) for i in range(36))
    assert fields[1](state[:36] + changed_phase) == comparator
    # Current phase does not drive this triangular comparator's form row;
    # nevertheless its nonzero phase row evolves with the actual form state.


def test_form_generator_is_symmetric_dissipative_and_not_a_phase_freeze(mp, fields):
    matrix = _form_generator()
    assert all(matrix[i][j] == matrix[j][i] for i in range(36) for j in range(36))
    assert all(sum(row) == 0 for row in matrix)
    assert all(matrix[i][j] >= 0 for i in range(36) for j in range(36) if i != j)
    forms = tuple(Q((7 * i) % 17 - 8, 4096) for i in range(36))
    expected = _matvec(matrix, forms)
    quadratic = sum(x * y for x, y in zip(forms, expected))
    dissipation = -sum(
        w * (forms[a] - forms[b]) ** 2 / 9 for a, b, w in _edges(0) + _edges(1)
    )
    dissipation -= sum((forms[j] - forms[18 + j]) ** 2 for j in range(9))
    assert quadratic == dissipation < 0
    state = _state(forms, (Q(0),) * 36)
    actual = fields[1](state)
    full = fields[0](state)
    for interval, value in zip(actual[:36], expected):
        assert interval.lo <= value <= interval.hi
    oracle = _oracle(mp, forms, (Q(0),) * 36)
    assert max(abs(value) for value in oracle[2]) < mp.mpf("1e-190")
    for i in range(72):
        _contains(mp, full[i], oracle[-1][i])
    assert any(value != 0 for value in oracle[-1][36:])
    # S(Theta)=0 only at this generic instantaneous point; moving phase
    # makes it invalid to identify subsequent full trajectories with the
    # comparator or to remove its phase coordinates.


def test_jets_separate_live_phase_motion_from_form_phase_feedback(mp, fields):
    forms = tuple(Q((i % 9) - 4, 4096) for i in range(36))
    errors = tuple(Q((i % 7) - 3, 8192) for i in range(36))
    state = _state(forms, errors)
    gamma = 1 / (1023 * mp.pi)
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns() * 2, errors)
    )
    for phase_direction in (False, True):
        direction = tuple(
            Q(int(i == (36 if phase_direction else 0))) for i in range(72)
        )
        jets = tuple(Jet((value, I(d))) for value, d in zip(state, direction))
        gradient = [Q(0)] * 36
        sine_derivative = [mp.mpf(0)] * 36
        for a, b, weight in _edges(0) + _edges(1):
            q = weight * (direction[a] - direction[b])
            gradient[a] += q
            gradient[b] -= q
            ds = mp.cos(theta[b] - theta[a]) * _mp(
                mp, direction[36 + b] - direction[36 + a]
            )
            sine_derivative[a] += ds
            sine_derivative[b] -= ds
        exchanged = [Q(0)] * 36
        for j in range(9):
            exchanged[j] = direction[18 + j] - direction[j]
            exchanged[18 + j] = -exchanged[j]
        comp = tuple(_mp(mp, -q / 9 + f) for q, f in zip(gradient, exchanged))
        comp += tuple(gamma * _mp(mp, q) / 9 for q in gradient)
        full = (
            tuple(comp[i] + gamma * sine_derivative[i] / 9 for i in range(36))
            + comp[36:]
        )
        for field, expected in zip(fields, (full, comp)):
            for jet, value in zip(field(jets), expected, strict=True):
                _contains(mp, jet.coeffs[1], value)
        if phase_direction:
            assert all(value == 0 for value in comp)
            assert max(abs(value) for value in full[:36]) > 0
        else:
            assert max(abs(value) for value in comp[36:]) > 0


def test_comparator_storage_retains_the_uncancelled_phase_term(mp, fields):
    forms = (Q(1, 1024),) + (Q(0),) * 35
    errors = forms
    x, q, sine, inputs, full = _oracle(mp, forms, errors)
    gamma = 1 / (1023 * mp.pi)
    comp = tuple(-a / 9 + b for a, b in zip(q, inputs)) + full[36:]
    full_derivative = sum(a * b for a, b in zip(q, full[:36])) - sum(
        a * b for a, b in zip(sine, full[36:])
    )
    comp_derivative = sum(a * b for a, b in zip(q, comp[:36])) - sum(
        a * b for a, b in zip(sine, comp[36:])
    )
    old_balance = -sum(value**2 for value in q) / 9 + sum(
        a * b for a, b in zip(q, inputs)
    )
    uncancelled = -gamma * sum(a * b for a, b in zip(q, sine)) / 9
    assert abs(full_derivative - old_balance) < mp.mpf("1e-190")
    assert abs(comp_derivative - old_balance - uncancelled) < mp.mpf("1e-190")
    assert uncancelled > 0 and comp_derivative > full_derivative
    rates = fields[1](_state(forms, errors))
    for interval, value in zip(rates, comp, strict=True):
        _contains(mp, interval, value)
    assert sum(a * b for a, b in zip(x, inputs)) <= 0
    y = tuple(
        _mp(mp, value - sum(errors[18 * (i // 18) : 18 * (i // 18 + 1)]) / 18)
        for i, value in enumerate(errors)
    )
    norm_derivative = sum(a * b for a, b in zip(x, comp[:36]))
    norm_derivative += sum(a * b for a, b in zip(y, comp[36:]))
    norm_identity = -sum(a * b for a, b in zip(x, q)) / 9
    norm_identity -= sum((x[j] - x[18 + j]) ** 2 for j in range(9))
    norm_identity += gamma * sum(a * b for a, b in zip(y, q)) / 9
    assert abs(norm_derivative - norm_identity) < mp.mpf("1e-190")
    assert norm_derivative <= _mp(mp, GAMMA) * sum(value**2 for value in x + y)
    # Removing one skew term changes this storage balance; norm retention
    # must use the separately proved dissipative form generator instead.


def test_complete_norm_and_paired_flux_bounds_use_the_retained_operator_norms(
    report, record_property
):
    assert report.certified and report.status == "shared_record_overlap_certified"
    assert (
        report.source_certified
        == report.finite_protocol_ready_ages_exist
        == report.conditional_ready_sources
        == (True, True)
    )
    assert report.actual_tube_certified and report.comparator_tube_certified
    assert Q(17, 12) ** 2 > 2
    exponent = GAMMA * TIME
    expected = Q(17, 12) * PREFIX_NORM / (1 - exponent)
    assert exponent == Q(1, 96000)
    assert (
        report.comparator_norm_upper_bound
        == expected
        < report.reference.joint_norm_upper_bound
        < NORM
    )
    assert report.comparator_norm_margin == NORM - expected > 0
    assert report.constituent_identity_retained_certified
    assert report.marked_gap_change_upper_bound == Q(1, 256000)
    assert report.retained_gap_upper_bounds == (DELTA / 8 + Q(1, 256000),) * 2
    assert min(report.registration_margins) > 0
    # ||S(theta)|| <=18||y|| gives a forcing norm <=2gM.
    # The symmetric form semigroup contracts; integrate once for e_x,
    # then again through the live phase row and through the original flux.
    force = 2 * GAMMA * NORM
    assert report.form_difference_upper_bound == force * TIME == Q(1, 1920000)
    assert (
        report.phase_difference_upper_bound
        == 2 * GAMMA * (force * TIME**2 / 2)
        == Q(1, 184320000000)
    )
    output_row = (Q(1),) * 9 + (Q(0),) * 9 + (-Q(1),) * 9 + (Q(0),) * 9
    assert sum(value**2 for value in output_row) == 18 < Q(17, 4) ** 2
    assert (
        report.transfer_difference_upper_bound
        == Q(17, 4) * force * TIME**2 / 2
        == BOUND
    )
    assert 0 < BOUND < 2 * NOISE
    assert report.difference_bound_certified
    assert report.overlap_margin == 2 * NOISE - BOUND > 0
    assert report.shared_record_overlap_certified
    assert report.matched_initial_state_assumed
    assert not report.comparator_own_law_source_acquisition_claimed
    assert report.new_contact_entry_readings == 0
    record_property(
        "paired_form_difference_upper_bound", str(report.form_difference_upper_bound)
    )
    record_property(
        "paired_phase_difference_upper_bound", str(report.phase_difference_upper_bound)
    )
    record_property("true_transfer_difference_upper_bound", str(BOUND))
    record_property("main_record_error_allowance", str(NOISE))
    record_property("true_record_overlap_margin", str(report.overlap_margin))


def test_true_record_midpoint_is_shared_without_using_coarse_enclosure_overlap():
    # Algebraic lemma for arbitrary true response levels and a certified
    # paired distance, not selected nonlinear endpoints or measured records.
    for allowance in (NOISE, BOUND / 2):
        for first in (-Q(17, 8), Q(0), Q(17, 8)):
            for gap in (-BOUND, -BOUND / 3, Q(0), BOUND / 3, BOUND):
                second = first + gap
                midpoint = (first + second) / 2
                intervals = (
                    (first - allowance, first + allowance),
                    (second - allowance, second + allowance),
                )
                assert abs(midpoint - first) <= allowance
                assert abs(midpoint - second) <= allowance
                assert (
                    max(row[0] for row in intervals)
                    <= midpoint
                    <= min(row[1] for row in intervals)
                )
                assert (
                    min(row[1] for row in intervals) - max(row[0] for row in intervals)
                    >= 2 * allowance - BOUND
                )
    # Overlapping coarse computed envelopes alone would not establish the
    # same proposition about the two true record intervals.
    true_first, true_second = Q(0), 4 * NOISE
    coarse = (-10 * NOISE, 10 * NOISE)
    assert coarse[0] <= true_first <= true_second <= coarse[1]
    assert true_first + NOISE < true_second - NOISE


def test_equality_and_unavailable_noise_do_not_become_a_separation_verdict():
    equality = _assess(readout_error_bound=BOUND / 2)
    assert equality.difference_bound_certified and equality.readout_allowance_admitted
    assert equality.overlap_margin == 0
    assert equality.shared_record_overlap_certified and equality.certified
    for allowance in (Q(0), BOUND / 2 - Q(1, 10**100)):
        unresolved = _assess(readout_error_bound=allowance)
        assert (
            unresolved.difference_bound_certified
            and unresolved.readout_allowance_admitted
        )
        assert unresolved.overlap_margin < 0
        assert (
            not unresolved.shared_record_overlap_certified and not unresolved.certified
        )
        assert unresolved.status == "unavailable"
        assert (
            "reading_allowance_does_not_certify_overlap"
            in unresolved.unresolved_conditions
        )
        assert unresolved.transfer_difference_upper_bound == BOUND
    outside = _assess(readout_error_bound=NOISE + Q(1, 10**100))
    assert outside.difference_bound_certified and outside.overlap_margin > 0
    assert (
        not outside.readout_allowance_admitted
        and not outside.shared_record_overlap_certified
    )
    assert not outside.certified


def test_both_actual_sources_and_matching_switch_are_re_admitted(monkeypatch, report):
    original = report.reference

    def install(value):
        monkeypatch.setattr(
            owner, "_assess_mixed_reciprocal_partner", lambda **kwargs: value
        )

    ignored = replace(
        original,
        certified=False,
        positive_information_certified=False,
        partner_history_obstruction_certified=False,
        uniform_arithmetic_error_upper_bound=Q(1),
    )
    install(ignored)
    assert _assess().certified
    for side in (0, 1):
        for field in (
            "source_certified",
            "finite_protocol_ready_ages_exist",
            "conditional_ready_sources",
        ):
            install(replace(original, **{field: tuple(i != side for i in range(2))}))
            unavailable = _assess()
            assert (
                not unavailable.difference_bound_certified and not unavailable.certified
            )
        changed = tuple(Q(0) if i == side else ERROR for i in range(2))
        install(replace(original, phase_error_bounds=changed))
        assert not _assess().certified
    for changed in (
        replace(original, contact_rate=Q(0)),
        replace(original, contact_duration=TIME + Q(1, 10**100)),
        replace(original, contact_pairs=tuple((j, 18 + (j + 1) % 9) for j in range(9))),
        replace(original, joint_norm_upper_bound=NORM),
        replace(original, full_state_tube_certified=False),
        replace(original, constituent_identity_retained_certified=False),
    ):
        install(changed)
        assert not _assess().difference_bound_certified
    install(original)
    for side in (0, 1):
        errors = tuple(ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2))
        unavailable = _assess(form_error_bounds=errors)
        assert not unavailable.source_certified[side] and not unavailable.certified


def test_readiness_retains_fixed_positive_existence_without_authenticating_zero():
    for side in (0, 1):
        gaps = tuple(
            DELTA / 8 + Q(1, 10**100) if i == side else DELTA / 8 for i in range(2)
        )
        unavailable = _assess(starting_gap_upper_bounds=gaps)
        assert (
            unavailable.source_certified[side]
            and unavailable.finite_protocol_ready_ages_exist[side]
        )
        assert (
            not unavailable.conditional_ready_sources[side]
            and not unavailable.certified
        )
    zero = _assess(starting_gap_upper_bounds=(0, 0))
    assert zero.certified and zero.conditional_ready_sources == (True, True)
    assert zero.existence_gap_threshold == DELTA / 8 > 0
    assert zero.starting_gap_upper_bounds == (Q(0), Q(0))
    # The supplied exact-zero readiness is conditional. Existence concerns
    # the positive threshold and supplies neither endpoint nor finite age.


def test_malformed_source_and_both_weight_vectors_precede_all_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed primitive reached source or numerical field work")

    for name in (
        "_assess_mixed_reciprocal_partner",
        "_derive_coupling_successor_geometry",
        "_sine_rate_evaluator",
        "_laplacian",
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
            for name in (
                "form_error_bounds",
                "phase_error_bounds",
                "starting_gap_upper_bounds",
            ):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(readout_error_bound=value)
    for name in (
        "form_error_bounds",
        "phase_error_bounds",
        "starting_gap_upper_bounds",
    ):
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
    with pytest.raises(ValueError):
        _assess(readout_error_bound=-Q(1, 10**100))
    for side in (0, 1):
        for values in (
            _weights(side)[:-1],
            _weights(side) + (Q(1),),
            repeat(Q(1)),
            (True,) + _weights(side)[1:],
            (Q(3, 4) - Q(1, 10**100),) + _weights(side)[1:],
            (Q(1) + Q(1, 10**100),) + _weights(side)[1:],
        ):
            arguments = dict(
                first_new_edge_conductances=_weights(0),
                second_new_edge_conductances=_weights(1),
            )
            arguments[
                ("first_new_edge_conductances", "second_new_edge_conductances")[side]
            ] = values
            with pytest.raises((TypeError, ValueError)):
                owner._mixed_without_form_feedback_field(**arguments)


def test_all_72_state_entries_precede_gradient_and_support_is_not_assumed(
    mp, fields, monkeypatch
):
    state = _state((Q(0),) * 36, (Q(0),) * 36)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid late coordinate reached shared Laplacian arithmetic")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_laplacian", forbidden)
        for values in (
            state[:-1],
            state + (I(0),),
            repeat(I(0)),
            (True,) + state[1:],
            state[:-1] + (float("nan"),),
            (Jet((I(0), I(1))),) + state[1:],
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                fields[1](values)
        jets = tuple(Jet((value, I(0))) for value in state)
        with pytest.raises(ValueError):
            fields[1](jets[:-1] + (Jet((I(0), I(0), I(1))),))
    width = Q(1, 10**12)
    enclosed = fields[1](state[:18] + (I(-width, width),) + state[19:])
    for endpoint in (-width, width):
        forms = (Q(0),) * 18 + (endpoint,) + (Q(0),) * 17
        _, q, _, inputs, full = _oracle(mp, forms, (Q(0),) * 36)
        expected = tuple(-a / 9 + b for a, b in zip(q, inputs)) + full[36:]
        for interval, value in zip(enclosed, expected):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    original = owner._derive_coupling_successor_geometry

    def corrupt(**kwargs):
        return replace(original(**kwargs), held_capacities=(Q(2),) * 18)

    monkeypatch.setattr(owner, "_derive_coupling_successor_geometry", corrupt)
    with pytest.raises(ValueError):
        owner._mixed_without_form_feedback_field(
            first_new_edge_conductances=_weights(0),
            second_new_edge_conductances=_weights(1),
        )


def test_one_shot_fixed_width_and_report_boundary_do_not_invent_observations(
    fields, report
):
    def fixed(value):
        return Q(np.int64(Q(value).numerator), np.int64(Q(value).denominator))

    compiled = owner._mixed_without_form_feedback_field(
        first_new_edge_conductances=(fixed(value) for value in _weights(0)),
        second_new_edge_conductances=(fixed(value) for value in _weights(1)),
    )
    state = _state((Q(0),) * 36, (Q(0),) * 36)
    assert compiled(value for value in state) == fields[1](state)
    normalized = _assess(
        form_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_error_bounds=(fixed(ERROR) for _ in range(2)),
        starting_gap_upper_bounds=(fixed(DELTA / 8) for _ in range(2)),
        readout_error_bound=fixed(NOISE),
    )
    assert normalized == report
    with pytest.raises(FrozenInstanceError):
        report.transfer_difference_upper_bound = Q(0)
    for name in (
        "reference",
        "initial_state",
        "actual_transfer",
        "comparator_transfer",
        "measured_phase",
        "elapsed_time",
        "ready_ages",
        "phase_clamp",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
