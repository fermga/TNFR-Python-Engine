"""Independent finite uniform-boundary controls for the acquired composite.

Generic field points below are not reconstructed acquired endpoints. The two
admitted histories are compared by an exact common-mode covariance, from the
same finite-ready source, without integrating or resetting either history.
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
from tnfr.physics import _mixed_composite_boundary as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
RADIUS = Q(1, 32)
TIME = Q(1, 32)
AMPLITUDE = Q(1, 16)
MEAN = AMPLITUDE * TIME
NOISE = Q(1, 10**6)
ARITHMETIC = Q(1, 10**40)


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
        if (
            min((bins[j] - bins[i]) % 9, (bins[i] - bins[j]) % 9) in (1, 2)
            or (i // 9 != j // 9 and bins[i] == bins[j])
        )
    }
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in (0, 1)
            for j in range(9)
        )
    )
    return old, tuple(sorted(pairs - set(old)))


def _weights():
    return tuple(Q(3, 4) + Q(i % 5, 16) for i in range(63))


def _edges():
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, _weights(), strict=True)
    )


def _state(forms, errors):
    return tuple(I(x) for x in forms) + tuple(
        2 * pi_interval() * turn + error
        for turn, error in zip(_turns(), errors, strict=True)
    )


def _oracle(mp, forms, errors):
    x = tuple(_mp(mp, value) for value in forms)
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns(), errors, strict=True)
    )
    q, sine = [mp.mpf(0)] * 18, [mp.mpf(0)] * 18
    for a, b, weight in _edges():
        gradient = _mp(mp, weight) * (x[a] - x[b])
        current = mp.sin(theta[b] - theta[a])
        q[a] += gradient
        q[b] -= gradient
        sine[a] += current
        sine[b] -= current
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 9 for a, b in zip(q, sine))
    phase = tuple(gamma * value / 9 for value in q)
    return x, tuple(q), tuple(sine), dx, phase


def _negative_exp_bounds(time):
    """Alternating series: even partial sum is upper, next odd is lower."""
    term, upper = Q(1), Q(1)
    for n in range(1, 61):
        term *= -time / n
        upper += term
    lower = upper + term * -time / 61
    return lower, upper


def _assess(**changes):
    arguments = dict(
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        starting_gap_upper_bound=DELTA / 8,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_composite_boundary(**arguments)


def _predict(**changes):
    arguments = dict(
        mean_reading=MEAN,
        mean_error_bound=NOISE,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._predict_mixed_composite_transfer(**arguments)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("finite boundary controls cannot advance or replay a source")

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
def fields():
    return {
        "charge": owner._mixed_composite_boundary_field(
            new_edge_conductances=_weights(), stage="charge", charge_amplitude=AMPLITUDE
        ),
        "probe": owner._mixed_composite_boundary_field(
            new_edge_conductances=_weights(), stage="probe"
        ),
    }


def test_all_edges_and_both_complete_rows_match_an_independent_oracle(mp, fields):
    assert len(_edges()) == 81
    assert (
        tuple(sum(i in (a, b) for a, b, _ in _edges()) for i in range(18)) == (9,) * 18
    )
    forms = tuple(Q(-1, 8) + Q((i % 7) - 3, 4096) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    state = _state(forms, errors)
    x, q, sine, intrinsic, phase = _oracle(mp, forms, errors)
    assert max(abs(value) for value in phase) > 0
    assert abs(sum(q)) < mp.mpf("1e-190")
    assert abs(sum(sine)) < mp.mpf("1e-190")
    for stage, field in fields.items():
        inputs = (
            (_mp(mp, AMPLITUDE),) * 18 if stage == "charge" else tuple(-v for v in x)
        )
        expected = tuple(a + b for a, b in zip(intrinsic, inputs)) + phase
        actual = field(state)
        assert len(actual) == 36
        for interval, value in zip(actual, expected, strict=True):
            _contains(mp, interval, value)
        total = 18 * _mp(mp, AMPLITUDE) if stage == "charge" else -sum(x)
        _contains(mp, sum(actual[:18], I(0)), total)
        _contains(mp, sum(actual[18:], I(0)), mp.mpf(0))
    assert fields["charge"](state)[18:] == fields["probe"](state)[18:]


def test_storage_and_signed_input_resource_do_not_invent_a_reservoir(mp, fields):
    forms = tuple(Q(1, 64) + Q((i % 7) - 3, 2048) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 2048) for i in range(18))
    x, q, sine, intrinsic, phase = _oracle(mp, forms, errors)
    qx = sum(a * b for a, b in zip(q, x))
    edge_energy = sum(_mp(mp, w) * (x[a] - x[b]) ** 2 for a, b, w in _edges())
    assert abs(qx - edge_energy) < mp.mpf("1e-190") and qx > 0
    for stage in fields:
        inputs = (
            (_mp(mp, AMPLITUDE),) * 18 if stage == "charge" else tuple(-v for v in x)
        )
        derivative = sum(
            a * (b + c) - d * e
            for a, b, c, d, e in zip(q, intrinsic, inputs, sine, phase)
        )
        expected = -sum(value**2 for value in q) / 9 - (qx if stage == "probe" else 0)
        assert abs(derivative - expected) < mp.mpf("1e-190") and expected < 0
    assert 18 * AMPLITUDE * TIME == Q(9, 256)
    assert -18 * AMPLITUDE * TIME == -Q(9, 256)
    # Zero work in this gradient storage for a common input does not supply
    # physical energy or turn the external boundary into a finite reservoir.


def test_translation_covariance_preserves_every_phase_and_internal_contrast(mp, fields):
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    shift = 2 * MEAN
    state = _state(forms, errors)
    translated = _state(tuple(x + shift for x in forms), errors)
    assert all(
        (forms[a] + shift) - (forms[b] + shift) == forms[a] - forms[b]
        for a, b, _ in _edges()
    )
    for stage, field in fields.items():
        first, second = field(state), field(translated)
        assert second[18:] == first[18:]
        expected = Q(0) if stage == "charge" else -shift
        for a, b in zip(first[:18], second[:18]):
            _contains(mp, b - a, _mp(mp, expected))
    opposite_charge = owner._mixed_composite_boundary_field(
        new_edge_conductances=_weights(), stage="charge", charge_amplitude=-AMPLITUDE
    )
    plus, minus = fields["charge"](translated), opposite_charge(state)
    assert plus[18:] == minus[18:]
    for a, b in zip(plus[:18], minus[:18]):
        _contains(mp, a - b, _mp(mp, 2 * AMPLITUDE))
    # For the actual two histories c(s)=2*MEAN*exp(-s) solves c'=-c.
    # Uniqueness transfers this covariance from any one true finite-ready
    # source to both histories; the generic field point above is not that source.


def test_jets_retain_phase_feedback_and_the_common_direction(mp, fields):
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    state = _state(forms, errors)
    jets = tuple(Jet((value, I(1 if i < 18 else 0))) for i, value in enumerate(state))
    x, _, _, intrinsic, phase = _oracle(mp, forms, errors)
    for stage, field in fields.items():
        actual = field(jets)
        inputs = (
            (_mp(mp, AMPLITUDE),) * 18
            if stage == "charge"
            else tuple(-value for value in x)
        )
        constants = tuple(a + b for a, b in zip(intrinsic, inputs)) + phase
        for i, (value, constant) in enumerate(zip(actual, constants, strict=True)):
            _contains(mp, value.coeffs[0], constant)
            first = -1 if stage == "probe" and i < 18 else 0
            _contains(mp, value.coeffs[1], mp.mpf(first))


def test_finite_gain_has_independent_alternating_and_high_precision_oracles(mp):
    prediction = _predict()
    lower, upper = _negative_exp_bounds(TIME)
    exact = mp.exp(-_mp(mp, TIME))
    assert _mp(mp, lower) <= exact <= _mp(mp, upper)
    assert upper - lower == TIME**61 / factorial(61)
    assert upper - lower < Q(1, 10**160)
    gain_lo, gain_hi = prediction.gain_bounds
    assert gain_lo <= 18 * (1 - upper) <= 18 * (1 - lower) <= gain_hi
    assert _mp(mp, gain_lo) <= 18 * (1 - exact) <= _mp(mp, gain_hi)
    assert gain_hi - gain_lo < Q(1, 10**80)
    assert 0 < 18 * (TIME - TIME**2 / 2) < gain_lo <= gain_hi < 18 * TIME
    assert prediction.forecast == MEAN * (gain_lo + gain_hi) / 2
    assert abs(prediction.forecast - MEAN * 18 * (1 - lower)) <= ARITHMETIC
    assert prediction.arithmetic_error_upper_bound <= ARITHMETIC


def test_noisy_aggregate_decoder_bounds_original_signed_records(mp, record_property):
    true_gain = 18 * (1 - mp.exp(-_mp(mp, TIME)))
    for actual_mean in (-MEAN, Q(0), MEAN):
        for mean_sign in (-1, 1):
            reading = actual_mean + mean_sign * NOISE
            prediction = _predict(mean_reading=reading)
            assert prediction.certified and prediction.mean_observation_consistent
            assert prediction.allowances_admitted and prediction.arithmetic_certified
            assert prediction.mean_reading == reading
            true_value = _mp(mp, actual_mean) * true_gain
            lo, hi = prediction.true_transfer_bounds
            assert _mp(mp, lo) <= true_value <= _mp(mp, hi)
            for record_sign in (-1, 1):
                record = true_value + record_sign * _mp(mp, NOISE)
                lo, hi = prediction.recorded_transfer_bounds
                assert _mp(mp, lo) <= record <= _mp(mp, hi)
                assert abs(record - _mp(mp, prediction.forecast)) <= _mp(
                    mp, prediction.record_prediction_error_upper_bound
                )
            assert prediction.mean_error_contribution <= 18 * TIME * NOISE
            assert prediction.record_prediction_error_upper_bound < 2 * NOISE
    assert 18 * TIME * NOISE + NOISE + ARITHMETIC < 2 * NOISE
    record_property("maximum_declared_record_error", str(2 * NOISE))
    record_property(
        "independent_budget_upper", str(18 * TIME * NOISE + NOISE + ARITHMETIC)
    )


def test_retained_signed_charge_integral_replaces_a_mean_observation(mp):
    for amplitude in (-AMPLITUDE, Q(0), AMPLITUDE):
        integral = amplitude * TIME
        prediction = _predict(
            mean_reading=integral, mean_error_bound=0, readout_error_bound=0
        )
        assert prediction.certified
        assert prediction.mean_error_contribution == 0
        assert prediction.record_prediction_error_upper_bound <= ARITHMETIC
        true_value = 18 * _mp(mp, integral) * (1 - mp.exp(-_mp(mp, TIME)))
        assert abs(true_value - _mp(mp, prediction.forecast)) <= _mp(
            mp, prediction.record_prediction_error_upper_bound
        )
        if amplitude == 0:
            assert prediction.forecast == 0
            assert prediction.true_transfer_bounds == (Q(0), Q(0))
    # This route retains the supplied signed input integral and the known
    # zero precharge mean. It does not estimate it from the reserved output.


def test_actual_history_witness_loses_information_only_when_mean_and_input_are_discarded(
    mp, report, record_property
):
    lower, upper = _negative_exp_bounds(TIME)
    gain_lo, gain_hi = report.gain_bounds
    gap_lo, gap_hi = report.actual_history_transfer_gap_bounds
    assert gap_lo <= 36 * MEAN * (1 - upper)
    assert 36 * MEAN * (1 - lower) <= gap_hi
    assert gap_lo == 2 * MEAN * gain_lo <= 2 * MEAN * gain_hi == gap_hi
    assert report.recorded_history_transfer_gap_lower_bound == gap_lo - 2 * NOISE
    analytic = 36 * MEAN * (TIME - TIME**2 / 2) - 2 * NOISE
    assert report.analytic_history_gap_lower_bound == analytic > Q(1, 1000)
    assert report.recorded_history_transfer_gap_lower_bound > analytic
    assert (
        report.weak_prediction_error_lower_bound
        == report.recorded_history_transfer_gap_lower_bound / 2
    )
    assert report.weak_prediction_error_lower_bound > 2 * NOISE
    assert report.relative_history_obstruction_certified
    assert report.separation_margin > 0
    # The two inputs are supplied histories from one and the same actual
    # ready source. Covariance makes the COMPLETE relative histories equal,
    # without claiming that two arbitrary containing-box points were acquired.
    record_property(
        "recorded_history_gap_lower",
        str(report.recorded_history_transfer_gap_lower_bound),
    )
    record_property(
        "weak_predictor_error_lower", str(report.weak_prediction_error_lower_bound)
    )
    record_property("phase_and_internal_contrast_histories_identical", True)


def test_same_source_readiness_and_finite_retention_do_not_supply_an_absolute_age(
    report,
):
    assert report.source_certified and report.conditional_ready_source
    assert report.finite_protocol_ready_age_exists
    assert report.starting_gap_domain_admitted
    assert (
        report.quotient_retention_certified
        and report.joint_registration_retained_certified
    )
    drift = 4 * Q(1, 3000) * RADIUS * (TIME + TIME)
    assert report.marked_gap_change_upper_bound == drift == Q(1, 384000)
    assert drift < DELTA / 8
    assert report.retained_gap_upper_bound == DELTA / 8 + drift < DELTA / 4
    assert report.registration_margin == DELTA / 4 - report.retained_gap_upper_bound > 0
    assert report.uniform_arithmetic_error_upper_bound <= ARITHMETIC
    assert report.record_prediction_error_upper_bound < 2 * NOISE
    assert report.prediction_margin > 0
    assert report.positive_information_certified and report.certified
    assert not report.unresolved_conditions
    zero = _assess(starting_gap_upper_bound=0)
    assert zero.conditional_ready_source and zero.starting_gap_domain_admitted
    # Existence here always refers to the fixed POSITIVE protocol threshold;
    # the independently supplied zero readiness premise has no finite-age proof.
    assert zero.finite_protocol_ready_age_exists
    assert zero.existence_gap_threshold == report.existence_gap_threshold == DELTA / 8
    assert report.controlled_form_channels == 18
    assert report.predictor_mean_coordinates == 1
    assert report.supplied_absolute_form_amount_upper_bound == Q(9, 256)
    for name in ("absolute_starting_age", "preparation_deadline", "observed_endpoint"):
        assert not hasattr(report, name)


def test_mean_observation_association_is_checked_without_clamping():
    touching = _predict(mean_reading=MEAN + NOISE)
    assert touching.certified and touching.mean_observation_consistent
    assert touching.mean_reading_bounds == (MEAN, MEAN + 2 * NOISE)
    assert touching.preparation_mean_bounds == (-MEAN, MEAN)
    assert touching.mean_reading == MEAN + NOISE
    for sign in (-1, 1):
        inconsistent = _predict(mean_reading=sign * (MEAN + NOISE + Q(1, 10**100)))
        assert not inconsistent.mean_observation_consistent
        assert not inconsistent.certified and inconsistent.unresolved_conditions
        assert inconsistent.mean_reading == sign * (MEAN + NOISE + Q(1, 10**100))


def test_nonnegative_outside_caps_yields_unavailability_not_a_false_record_claim():
    for key in ("mean_error_bound", "readout_error_bound"):
        report = _predict(**{key: NOISE + Q(1, 10**100)})
        assert not report.allowances_admitted and not report.certified
    for key, limit in (
        ("form_error_bound", ERROR),
        ("phase_error_bound", ERROR),
        ("starting_gap_upper_bound", DELTA / 8),
        ("readout_error_bound", NOISE),
    ):
        report = _assess(**{key: limit + Q(1, 10**100)})
        assert not report.certified and report.unresolved_conditions
        if key == "starting_gap_upper_bound":
            assert report.source_certified and report.finite_protocol_ready_age_exists
            assert not report.starting_gap_domain_admitted


def test_source_reassociation_uses_common_degree_premises_not_other_verdicts(
    report, monkeypatch
):
    source = report.successor
    ignored = replace(
        source,
        certified=False,
        status="unavailable",
        normalization_common_form_separation_certified=False,
        disconnected_nonregistration_certified=False,
    )
    monkeypatch.setattr(
        owner, "_assess_sine_coupling_successor", lambda **kwargs: ignored
    )
    assert _assess().certified
    common, split = source.completion_reports
    corruptions = (
        replace(source, form_error_bound=2 * ERROR),
        replace(source, source_certified=False),
        replace(source, conditional_image_certified=False),
        replace(source, phase_radius=2 * RADIUS),
        replace(source, gamma_range=(Q(1, 3300), Q(1, 2999))),
        replace(source, completion_reports=(split, common)),
        replace(
            source,
            completion_reports=(
                replace(common, initial_quotient_radius_margin=0),
                split,
            ),
        ),
        replace(
            source,
            precontact_reference=replace(
                source.precontact_reference, initial_form_means=(MEAN, -MEAN)
            ),
        ),
    )
    for candidate in corruptions:
        monkeypatch.setattr(
            owner,
            "_assess_sine_coupling_successor",
            lambda _candidate=candidate, **kwargs: _candidate,
        )
        result = _assess()
        assert not result.source_certified and not result.certified
        assert not result.positive_information_certified
        assert not result.relative_history_obstruction_certified
    no_existence = replace(
        source,
        completion_reports=(
            replace(common, asymptotic_convergence_certified=False),
            split,
        ),
    )
    monkeypatch.setattr(
        owner, "_assess_sine_coupling_successor", lambda **kwargs: no_existence
    )
    result = _assess()
    assert result.source_certified and result.conditional_ready_source
    assert not result.finite_protocol_ready_age_exists and not result.certified


def test_arithmetic_budget_is_separate_from_the_record_tolerance(monkeypatch):
    original = owner.exp_unit_bounds

    def widened(value):
        lower, upper = original(value)
        return lower, upper + Q(1, 10**30)

    monkeypatch.setattr(owner, "exp_unit_bounds", widened)
    result = _predict()
    assert result.mean_observation_consistent and result.allowances_admitted
    assert result.record_prediction_error_upper_bound < 2 * NOISE
    assert result.arithmetic_error_upper_bound > ARITHMETIC
    assert not result.arithmetic_certified and not result.certified
    protocol = _assess()
    assert protocol.prediction_margin > 0
    assert protocol.uniform_arithmetic_error_upper_bound > ARITHMETIC
    assert not protocol.positive_information_certified and not protocol.certified


def test_malformed_primitives_reject_before_sources_fields_or_exponentials(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitives reached source, field or numerical work")

    for name in (
        "_coupling_successor_field",
        "_assess_sine_coupling_successor",
        "exp_unit_bounds",
        "I",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    invalid_scalars = (True, False, float("nan"), float("inf"), "0", complex(0))
    for value in invalid_scalars:
        for key in ("mean_reading", "mean_error_bound", "readout_error_bound"):
            with pytest.raises((TypeError, ValueError)):
                _predict(**{key: value})
        for key in (
            "form_error_bound",
            "phase_error_bound",
            "starting_gap_upper_bound",
            "readout_error_bound",
        ):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{key: value})
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_composite_boundary_field(
                new_edge_conductances=_weights(), stage="charge", charge_amplitude=value
            )
    for key in ("mean_error_bound", "readout_error_bound"):
        with pytest.raises(ValueError):
            _predict(**{key: -Q(1, 10**100)})
    for key in (
        "form_error_bound",
        "phase_error_bound",
        "starting_gap_upper_bound",
        "readout_error_bound",
    ):
        with pytest.raises(ValueError):
            _assess(**{key: -Q(1, 10**100)})
    for stage in (None, True, 0, b"charge", "CHARGE", "probe "):
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_composite_boundary_field(
                new_edge_conductances=_weights(), stage=stage
            )
    for amplitude in (AMPLITUDE + Q(1, 10**100), -AMPLITUDE - Q(1, 10**100)):
        with pytest.raises(ValueError):
            owner._mixed_composite_boundary_field(
                new_edge_conductances=_weights(),
                stage="charge",
                charge_amplitude=amplitude,
            )
    with pytest.raises(ValueError):
        owner._mixed_composite_boundary_field(
            new_edge_conductances=_weights(),
            stage="probe",
            charge_amplitude=Q(1, 10**100),
        )


def test_shared_weight_state_interval_and_jet_admission_are_preserved(mp, fields):
    for values in (
        (),
        (1,) * 62,
        (1,) * 64,
        repeat(1),
        {0: 1},
        (True,) + (1,) * 62,
        (Q(3, 4) - Q(1, 10**100),) + (1,) * 62,
    ):
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_composite_boundary_field(
                new_edge_conductances=values, stage="probe"
            )
    bad_states = (
        (0,) * 35,
        (0,) * 37,
        repeat(0),
        (True,) + (0,) * 35,
        (float("nan"),) + (0,) * 35,
        (Jet.constant(0, 1),) + (0,) * 35,
        (Jet.constant(0, 1),) + (Jet.constant(0, 2),) * 35,
    )
    for values in bad_states:
        with pytest.raises((TypeError, ValueError)):
            fields["probe"](values)
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(18))
    errors = (Q(0),) * 18
    state = list(_state(forms, errors))
    radius = Q(1, 10**12)
    state[0] += I(-radius, radius)
    actual = fields["probe"](state)
    for sign in (-1, 1):
        x, _, _, rates, phase = _oracle(
            mp, (forms[0] + sign * radius,) + forms[1:], errors
        )
        expected = tuple(a - b for a, b in zip(rates, x)) + phase
        for interval, value in zip(actual, expected, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_fixed_width_components_one_shot_inputs_and_no_hidden_decoder_access(report):
    def fixed(value):
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    assert (
        _assess(
            form_error_bound=fixed(ERROR),
            phase_error_bound=fixed(ERROR),
            starting_gap_upper_bound=fixed(DELTA / 8),
            readout_error_bound=fixed(NOISE),
        )
        == report
    )
    assert (
        _predict(
            mean_reading=fixed(MEAN),
            mean_error_bound=fixed(NOISE),
            readout_error_bound=fixed(NOISE),
        )
        == _predict()
    )
    weights = (fixed(value) for value in _weights())
    first = owner._mixed_composite_boundary_field(
        new_edge_conductances=weights, stage="charge", charge_amplitude=fixed(AMPLITUDE)
    )
    second = owner._mixed_composite_boundary_field(
        new_edge_conductances=_weights(), stage="charge", charge_amplitude=AMPLITUDE
    )
    state = _state((Q(0),) * 18, (Q(0),) * 18)
    assert first(iter(state)) == second(state)
    with pytest.raises(FrozenInstanceError):
        report.certified = False
    for name in (
        "source",
        "report",
        "state",
        "relative_phase",
        "charge_history",
        "reserved_response",
        "elapsed_time",
        "equilibrium",
    ):
        with pytest.raises(TypeError):
            _predict(**{name: 0})
    for name in (
        "source",
        "report",
        "absolute_starting_age",
        "equilibrium",
        "elapsed_time",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
