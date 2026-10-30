"""Independent endpoint phase discrimination on the admitted actual UM family.

The source-inclusive envelopes are compared between independently permitted
preparations and readout errors. No endpoint, trajectory, integrated record,
sensor calibration or source acquisition is executed by these controls.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_recipient_phase_response as owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
WEIGHT_ERROR = Q(2, 10**60)
MATURE = Q(1, 2**28)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
DURATION = Q(2 * 10**6)
NOISE = Q(1, 10**6)
THRESHOLD = Q(1, 800000)
GAMMA_LO, GAMMA_HI = Q(1, 3300), Q(1, 3000)
STIFFNESS_LO = Q(32, 81 * 3300**2)
SIGNAL_FLOOR = Q(13, 5 * 10**6)


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _offset(gap):
    return (-gap / 2,) * 9 + (gap / 2,) * 9


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_recipient_phase_response(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    def forbidden(*args, **kwargs):
        pytest.fail("phase endpoint controls cannot advance or replay a source")

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
        yield


@pytest.fixture(scope="module")
def mp():
    result = mpmath.mp.clone()
    result.dps = 200
    return result


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_endpoint_covector_is_the_signed_marked_recipient_gap_and_cancels_gauge(
    mp, report
):
    row = (Q(0),) * 54 + (Q(1, 9),) * 9 + (-Q(1, 9),) * 9
    assert report.observation_covector == row
    assert sum(row) == 0 and _dot(row, row) == Q(2, 9)
    pairs = tuple((j, 9 + (5 * (j - 4) + 4) % 9) for j in range(9))
    assert report.source_geometry.marked_pairs == pairs
    branch = tuple(_turns()[b] - _turns()[a] for a, b in pairs)
    assert all(value.denominator == 1 for value in branch) and sum(branch) == 0
    for sign in (-1, 1):
        donor = _offset(sign * SECTION)
        recipient = tuple(
            a + Q(i % 3 - 1, 10**8) for i, a in enumerate(_offset(-sign * SECTION / 3))
        )
        expected = sum(recipient[:9]) / 9 - sum(recipient[9:]) / 9
        assert sign * expected > 0
        for donor_gauge, recipient_gauge in ((Q(0), Q(0)), (Q(4, 7), -Q(5, 13))):
            phases = tuple(
                2 * mp.pi * _mp(mp, t) + _mp(mp, y + donor_gauge)
                for t, y in zip(_turns(), donor)
            )
            phases += tuple(
                2 * mp.pi * _mp(mp, t) + _mp(mp, y + recipient_gauge)
                for t, y in zip(_turns(), recipient)
            )
            full = tuple(_mp(mp, Q(i - 17, 31)) for i in range(36)) + phases
            measured = sum(
                _mp(mp, coefficient) * value for coefficient, value in zip(row, full)
            )
            gaps = tuple(
                phases[18 + b] - phases[18 + a] - 2 * mp.pi * _mp(mp, turn)
                for (a, b), turn in zip(pairs, branch)
            )
            assert abs(measured + sum(gaps) / 9) < mp.mpf("1e-190")
            assert abs(measured - _mp(mp, expected)) < mp.mpf("1e-190")
            assert max(abs(gap) for gap in gaps) < _mp(mp, DELTA / 4)


def test_total_integrated_form_exchange_is_blind_to_this_endpoint_phase(report):
    # Both actual sources have zero per-copy form means. The complete mutual
    # law preserves them: mD'=mR-mD and mR'=mD-mR. Therefore the total of
    # all 18 port currents, and its integral at any time, is exactly zero.
    donor = tuple(Q(i % 3 - 1, 100) for i in range(18))
    recipient = tuple(Q((i + 1) % 3 - 1, 200) for i in range(18))
    current = tuple(a - b for a, b in zip(donor, recipient))
    assert sum(donor) == sum(recipient) == sum(current) == 0
    assert max(abs(value) for value in current) > 0
    assert report.reference.initial_form_means == (Q(0), Q(0))
    assert report.connected_true_bounds[0] > 0
    assert report.cut_true_bounds[0] < 0 < report.cut_true_bounds[1]
    assert report.independent_record_sets_separated
    # This is blindness of the TOTAL form flux; no claim about individual
    # regional flux records, and no replacement of O(T) by an integral.
    assert report.observation_covector[:36] == (Q(0),) * 36


def test_damped_integral_bounds_retain_the_inherited_velocity(mp):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    d, e = 2 * k, 2 * k + 2
    assert _mp(mp, Q(8, 9)) < d < 1 and e > _mp(mp, Q(26, 9))
    for time in (Q(0), Q(1), Q(3, 2), DURATION):
        t = _mp(mp, time)
        kernels = []
        for damping in (d, e):
            decay = mp.exp(-damping * t)
            velocity_kernel = (1 - decay) / damping
            force_kernel = t / damping - (1 - decay) / damping**2
            assert velocity_kernel >= 0 and force_kernel >= 0
            # These are independent closed Green-kernel identities; no flow
            # integration or time-step response is used.
            assert abs(decay + damping * velocity_kernel - 1) < mp.mpf("1e-190")
            assert abs(velocity_kernel + damping * force_kernel - t) < mp.mpf("1e-190")
            kernels.append((velocity_kernel, force_kernel))
        assert kernels[0][0] >= kernels[1][0]
        assert kernels[0][1] >= kernels[1][1]
        if time == DURATION:
            assert kernels[0][1] > t - _mp(mp, Q(81, 64))
            assert kernels[1][1] < _mp(mp, Q(9, 26)) * t
            q0 = _mp(mp, SECTION)
            w0 = -(gamma**2) * q0 / 2
            assert w0 < 0
            # The same inherited w0 enters both equations. The declared
            # estimate conservatively discards its favorable cancellation.
            inherited_difference = -w0 * (kernels[0][0] - kernels[1][0])
            assert inherited_difference > 0
            assert -w0 * kernels[1][0] < _mp(mp, Q(9, 26) * GAMMA_HI**2) * q0
            constant_force_difference = q0 * (
                mp.mpf(3) / 4 * kernels[0][1] - kernels[1][1]
            )
            lower = q0 * _mp(mp, Q(21, 52) * DURATION - Q(243, 256))
            assert constant_force_difference > lower > 0


def test_nonlinear_rectangle_and_rational_duhamel_signal_are_independent(mp, report):
    qlo, qhi = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
    fraction = 1 - GAMMA_HI**2 * DURATION
    assert fraction == Q(7, 9)
    sine_factor = 1 - qhi**2 / 6
    assert sine_factor > Q(27, 28)
    assert fraction * sine_factor > Q(3, 4)
    # Independent nonlinear checks over fixed corners/interior of the
    # proved rectangle, not a replacement of its analytical inequality.
    for s in (fraction * qlo, qlo, qhi):
        for z in (fraction * qlo, qlo, qhi):
            restoring_s = 2 * mp.sin(_mp(mp, s) / 2) * mp.cos(_mp(mp, z) / 2)
            restoring_z = 2 * mp.sin(_mp(mp, z) / 2) * mp.cos(_mp(mp, s) / 2)
            assert restoring_s > _mp(mp, Q(3, 4) * qlo)
            assert 0 < restoring_z <= _mp(mp, qhi)
    time_coefficient = Q(21, 52) * DURATION - Q(243, 256)
    penalty = Q(9, 26) * GAMMA_HI**2
    bracket = STIFFNESS_LO * time_coefficient - penalty
    signal = qlo * bracket / 2
    assert time_coefficient > 0 and bracket > 0
    assert signal > SIGNAL_FLOOR
    assert report.nominal_q0_bounds == (qlo, qhi)
    assert report.wedge_fraction_lower_bound == fraction
    assert report.sine_factor_lower_bound == sine_factor
    assert report.stiffness_lower_bound == STIFFNESS_LO
    assert report.duhamel_time_coefficient == time_coefficient
    assert report.inherited_velocity_penalty == penalty > 0
    assert report.nominal_response_bracket == bracket
    assert report.nominal_response_lower_bound == signal
    assert report.nominal_signal_margin == signal - SIGNAL_FLOOR > 0
    assert report.finite_envelope_certified


def test_independent_record_sets_allow_distinct_preparations_and_adversarial_errors(
    report, record_property
):
    signal = report.nominal_response_lower_bound
    allowance = 8 * MATURE
    connected = (signal - allowance, (SECTION + SECTION_ERROR / 2) / 2 + allowance)
    cut = (-allowance, allowance)
    assert report.actual_observation_allowance == allowance
    assert report.connected_true_bounds == connected
    assert report.cut_true_bounds == cut
    assert report.connected_record_bounds == (
        connected[0] - NOISE,
        connected[1] + NOISE,
    )
    assert report.cut_record_bounds == (cut[0] - NOISE, cut[1] + NOISE)
    # Corners are proof variables in independent admissible envelopes, not
    # synthetic measured endpoints and not a paired-state cancellation.
    for true_full in connected:
        for true_cut in cut:
            for full_error in (-NOISE, Q(0), NOISE):
                for cut_error in (-NOISE, Q(0), NOISE):
                    observed_full, observed_cut = (
                        true_full + full_error,
                        true_cut + cut_error,
                    )
                    assert observed_full > THRESHOLD > observed_cut
                    assert observed_full - observed_cut > Q(1, 2 * 10**6)
    assert (
        report.record_separation_margin
        == connected[0] - cut[1] - 2 * NOISE
        > Q(1, 2 * 10**6)
    )
    assert report.connected_threshold_margin == connected[0] - NOISE - THRESHOLD > 0
    assert report.cut_threshold_margin == THRESHOLD - cut[1] - NOISE > 0
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    assert report.certified
    for name, value in (
        ("nominal_response_lower_bound", signal),
        ("connected_record_lower_bound", report.connected_record_bounds[0]),
        ("cut_record_upper_bound", report.cut_record_bounds[1]),
        ("independent_record_gap", report.record_separation_margin),
        ("fixed_phase_threshold", THRESHOLD),
        ("structural_observation_time", DURATION),
    ):
        record_property(name, str(value))
    record_property("measured_endpoint_evaluated", "false")
    record_property("preparations_across_hypotheses", "independent admitted members")


def test_pointwise_state_allowance_and_aggregate_noise_have_distinct_domains(
    monkeypatch, report
):
    monkeypatch.setattr(
        owner,
        "_assess_mixed_resource_transfer_obstruction",
        lambda **kwargs: report.reference,
    )
    no_noise = _assess(observation_error_bound=Q(0))
    assert no_noise.certified
    assert no_noise.connected_record_bounds == report.connected_true_bounds
    assert no_noise.cut_record_bounds == report.cut_true_bounds
    assert (
        no_noise.record_separation_margin == report.record_separation_margin + 2 * NOISE
    )
    assert (
        no_noise.actual_observation_allowance
        == report.actual_observation_allowance
        == 8 * MATURE
    )
    assert no_noise.actual_observation_allowance < DURATION * 8 * MATURE
    assert _dot(report.observation_covector, report.observation_covector) < 1
    assert report.numerical_response_allowance == 0
    assert report.observation_error_cap == NOISE
    assert report.independently_observed_aggregates == 1
    outside = _assess(observation_error_bound=NOISE + Q(1, 10**100))
    assert outside.finite_envelope_certified and outside.identity_certified
    assert outside.record_separation_margin > 0
    assert not outside.observation_allowance_admitted
    assert not outside.independent_record_sets_separated
    assert not outside.fixed_threshold_certified and not outside.certified
    assert outside.status == "unavailable"
    assert (
        "aggregate_observation_error_exceeds_protocol" in outside.unresolved_conditions
    )


def test_prior_target_resource_and_global_verdicts_do_not_decide_this_observation(
    monkeypatch, report
):
    previous = report.reference
    assert (
        previous.connected_target_unreachable
        and previous.disconnected_target_unreachable
    )
    assert previous.positive_donor_resource_certified
    changed = replace(
        previous,
        target_domain_certified=False,
        connected_target_unreachable=False,
        disconnected_target_unreachable=False,
        positive_donor_resource_certified=False,
        certified=False,
        status="unused_old_conclusions",
        unresolved_conditions=("old_target_not_consumed",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_resource_transfer_obstruction", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified and result.finite_envelope_certified
    assert result.connected_record_bounds == report.connected_record_bounds
    assert result.cut_record_bounds == report.cut_record_bounds
    assert result.record_separation_margin == report.record_separation_margin
    assert not result.old_opposite_gap_target_claimed_reached
    assert not result.phase_jump_performed and not result.charge_or_leak_prefix_applied
    # A useful endpoint reading does not reverse the separate old endpoint
    # exclusion. That conclusion is neither a premise nor the new output.
    assert report.reference.connected_target_unreachable


def test_both_original_sources_and_donor_recipient_associations_are_readmitted(
    monkeypatch, report
):
    previous = report.reference
    for side in (0, 1):
        source = previous.sources[side]
        changes = (
            replace(source, source_certified=False),
            replace(source, edge_weight_deviation_upper_bound=2 * WEIGHT_ERROR),
            replace(source, gauge_phase_deviation_upper_bound=7 * ETA),
            replace(source, phase_reset_error_bound=4 * ETA),
            replace(
                source,
                precontact_reference=replace(
                    source.precontact_reference,
                    supplied_phase_origins=(-SECTION / 2, SECTION / 2),
                ),
            ),
            replace(
                source,
                precontact_reference=replace(
                    source.precontact_reference, initial_form_means=(Q(1), Q(0))
                ),
            ),
        )
        for changed_source in changes:
            sources = list(previous.sources)
            sources[side] = changed_source
            if side == 0:
                changed = replace(
                    previous,
                    sources=tuple(sources),
                    donor=replace(previous.donor, source=changed_source),
                )
            else:
                ready_sources = (
                    changed_source,
                    previous.recipient_readiness.sources[1],
                )
                changed = replace(
                    previous,
                    sources=tuple(sources),
                    recipient_readiness=replace(
                        previous.recipient_readiness, sources=ready_sources
                    ),
                )
            assert changed.source_association_certified  # stale flag is insufficient
            monkeypatch.setattr(
                owner,
                "_assess_mixed_resource_transfer_obstruction",
                lambda **kwargs: changed,
            )
            result = _assess()
            assert not result.source_association_certified
            assert not result.finite_envelope_certified and not result.certified
    for name, value in (
        ("donor_source_certified", False),
        ("recipient_source_certified", False),
        ("section_mean_gap", 2 * SECTION),
        ("recipient_mature_radius", 2 * MATURE),
        ("recipient_isolated_deadline", Q(10**9)),
        ("inherited_donor_velocity_retained", False),
        ("inherited_nominal_velocity_relation", "independent_velocity_box"),
        ("initial_form_means", (Q(0), Q(1))),
        ("donor_dwell_applied", True),
        ("phase_jump_performed", True),
        ("clock", "unscaled_time"),
    ):
        changed = replace(previous, **{name: value})
        monkeypatch.setattr(
            owner,
            "_assess_mixed_resource_transfer_obstruction",
            lambda **kwargs: changed,
        )
        assert not _assess().source_association_certified
    monkeypatch.setattr(
        owner, "_assess_mixed_resource_transfer_obstruction", lambda **kwargs: previous
    )
    for name in ("form_error_bounds", "phase_error_bounds"):
        for values in ((ERROR / 2, ERROR), (ERROR, ERROR / 2)):
            assert not _assess(**{name: values}).source_association_certified


def test_trap_metric_wedge_and_fine_error_premises_cannot_be_replaced_by_true_flags(
    monkeypatch, report
):
    previous = report.reference
    failures = (
        ("actual_trap_certified", False, "actual_trap_certified"),
        ("full_norm_squared_upper_bound", Q(1, 100) ** 2, "actual_trap_certified"),
        ("dissipation_margin", Q(1), "actual_trap_certified"),
        ("metric_certified", False, "comparison_certified"),
        (
            "conservative_metric_decay_rate",
            previous.base_metric_decay_rate / 4,
            "comparison_certified",
        ),
        ("metric_growth_squared_margin", Q(0), "comparison_certified"),
        ("comparison_forcing_upper_bound", WEIGHT_ERROR, "comparison_certified"),
        ("initial_comparison_norm_upper_bound", MATURE, "comparison_certified"),
        (
            "uniform_comparison_bound",
            previous.uniform_comparison_bound / 2,
            "comparison_certified",
        ),
        ("uniform_comparison_upper_bound", 4 * MATURE, "comparison_certified"),
        ("registration_margin", Q(0), "identity_certified"),
        ("all_contact_durations_covered", False, "identity_certified"),
        ("nominal_wedges_certified", False, "finite_envelope_certified"),
        ("nominal_q0_bounds", (SECTION, SECTION), "finite_envelope_certified"),
        ("nominal_velocity_factor_upper_bound", Q(0), "finite_envelope_certified"),
    )
    for name, value, failed_flag in failures:
        changed = replace(previous, **{name: value})
        assert changed.certified  # every other old conclusion can remain True
        monkeypatch.setattr(
            owner,
            "_assess_mixed_resource_transfer_obstruction",
            lambda **kwargs: changed,
        )
        result = _assess()
        assert result.source_association_certified
        assert not getattr(result, failed_flag)
        assert not result.independent_record_sets_separated and not result.certified


def test_original_marked_branch_geometry_is_part_of_observation_admission(
    monkeypatch, report
):
    registered = owner._derive_registered_phase_probe_geometry()
    monkeypatch.setattr(
        owner,
        "_assess_mixed_resource_transfer_obstruction",
        lambda **kwargs: report.reference,
    )
    geometry = registered.successor
    for changed_geometry in (
        replace(geometry, held_capacities=(Q(2),) + geometry.held_capacities[1:]),
        replace(
            geometry,
            marked_branch_turns=(geometry.marked_branch_turns[0] + 1,)
            + geometry.marked_branch_turns[1:],
        ),
        replace(
            geometry,
            target_phase_turns=(geometry.target_phase_turns[0] + Q(1, 9),)
            + geometry.target_phase_turns[1:],
        ),
        replace(geometry, components=(tuple(range(9)), tuple(range(9, 18)))),
    ):
        changed = replace(registered, successor=changed_geometry)
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: changed
        )
        result = _assess()
        assert not result.observation_geometry_certified
        assert not result.source_association_certified and not result.certified


def test_raw_pairs_and_noise_are_admitted_before_source_geometry_or_arithmetic(
    monkeypatch,
):
    # Finite nonnegative values beyond the prospective source caps remain
    # candidates with unavailable claims, rather than malformed primitives.
    for name in ("form_error_bounds", "phase_error_bounds"):
        for side in (0, 1):
            values = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            result = _assess(**{name: values})
            assert not result.source_association_certified and not result.certified
            assert result.status == "unavailable"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached source, geometry or derived arithmetic")

    for name in (
        "_assess_mixed_resource_transfer_obstruction",
        "_derive_registered_phase_probe_geometry",
        "_original_source_matches",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    invalid_scalars = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
        -Q(1, 10**100),
    )
    for bad in invalid_scalars:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(observation_error_bound=bad)
        for name in ("form_error_bounds", "phase_error_bounds"):
            for side in (0, 1):
                values = tuple(bad if i == side else Q(0) for i in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: values})
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})


def test_one_shot_fixed_width_and_fixed_protocol_do_not_admit_a_fitted_endpoint(report):
    fixed_error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    fixed_noise = Q(np.int64(NOISE.numerator), np.int64(NOISE.denominator))
    result = _assess(
        form_error_bounds=(fixed_error for _ in range(2)),
        phase_error_bounds=(fixed_error for _ in range(2)),
        observation_error_bound=fixed_noise,
    )
    assert result == report
    assert report.status == "conditional_recipient_phase_response"
    assert report.contact_duration == DURATION
    assert report.original_clock_duration * Q(1023, 1024) == DURATION
    assert report.threshold == THRESHOLD and report.nominal_signal_floor == SIGNAL_FLOOR
    assert report.physical_capacities == (Q(1),) * 36
    assert report.contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert report.full_state_dimension == 72
    assert report.observation_is_endpoint_not_integral
    assert report.independently_selected_hypothesis_sources
    assert report.inherited_donor_velocity_retained
    assert not report.source_synchronization_authenticated
    assert not report.observed_record_supplied and not report.sensor_model_derived
    assert not report.source_trajectory_evaluated and not report.runtime_event_executed
    with pytest.raises(FrozenInstanceError):
        report.threshold = Q(0)
    for name in (
        "reference",
        "donor",
        "recipient",
        "initial_state",
        "observed_phase",
        "observed_transfer",
        "donor_velocity",
        "section_age",
        "contact_duration",
        "observation_covector",
        "threshold",
        "readout_error_bound",
        "mature_norm_upper_bounds",
        "new_edge_conductances",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
