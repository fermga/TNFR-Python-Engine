"""Static early-exclusion and all-future late-retention controls.

No response is integrated. Universal record-family enclosures remain separate
from representative reader calls and conditional comparison-boundary controls.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest

# Import only the pure arithmetic helper, never the old module's fixtures.
# Its current source is recorded as a dependency; no frozen result is loaded.
from tests.physics.test_organized_weak_contact_record import _uniform_inverse
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_contact_maintenance as owner

ERROR = Q(1, 10**10)
ETA = Q(1, 100000)
TOLERANCE = Q(1, 1000)


@pytest.fixture(scope="module", autouse=True)
def no_execution_or_retained_response():
    def forbidden(*args, **kwargs):
        pytest.fail("timing controls cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _assess(**changes):
    return owner._assess_organized_weak_contact_timing(
        **(
            dict(
                form_error_bounds=(ERROR,) * 6,
                phase_error_bounds=(ERROR,) * 6,
                quadratures=(Q(1), Q(0)),
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def actual():
    source_calls, record_calls = [], []
    rebuild = owner._assess_organized_external_interaction
    decode = owner._assess_organized_weak_contact_record

    def counted_source(**kwargs):
        source_calls.append(kwargs)
        return rebuild(**kwargs)

    def counted_record(**kwargs):
        record_calls.append(kwargs)
        return decode(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_assess_organized_external_interaction", counted_source)
        patch.setattr(owner, "_assess_organized_weak_contact_record", counted_record)
        result = _assess()
    assert len(source_calls) == len(record_calls) == 1
    assert source_calls[0] == dict(
        form_error_bounds=(ERROR,) * 6, phase_error_bounds=(ERROR,) * 6
    )
    return result


@pytest.fixture
def source_wiring(monkeypatch, actual):
    monkeypatch.setattr(
        owner,
        "_assess_organized_external_interaction",
        lambda **kwargs: actual.record.source,
    )
    return actual.record.source


def _conditional_record(actual, interval):
    """Comparison-only envelope with separately established basin premises.

    This is not presented as a newly measured raw record. A whole-family
    outer enclosure or a wider valid comparison allowance is supplied only
    to the internal mathematical kernel, never the primitive reader.
    """
    source = actual.record.source
    basin = owner._decide_admitted_weak_contact_basin(
        (ERROR,) * 6, (ERROR,) * 6, (interval.lo, interval.hi), source
    )
    assert basin.status == "certified_unique_registration"
    return replace(
        actual.record,
        donor_origin_bounds=interval,
        basin=basin,
        registration_label=basin.registration_label,
        limiting_relative_origin_turns=basin.limiting_relative_origin_turns,
    )


def test_near_center_record_has_one_fresh_build_and_independent_late_retention(actual):
    assert actual.record.status == "certified_observed_registration"
    assert actual.record.registration_label == 0
    assert actual.record.source is actual.record.basin.source
    assert actual.status == "certified_late_retention_only"
    assert not actual.early_certified and actual.late_certified
    assert actual.initial_distance_lower == 0 and actual.early_margin < 0
    assert actual.relative_form_upper > 0
    assert (
        actual.retained_distance_upper
        == actual.corrected_tube_radius + actual.relative_form_upper
    )
    assert actual.retained_distance_upper < actual.tolerance == TOLERANCE
    assert (
        "initial_interval_too_close_for_early_exclusion" in actual.unavailable_reasons
    )
    with pytest.raises(FrozenInstanceError):
        actual.early_certified = True


def test_deadlines_retain_actual_gamma_both_rows_and_structural_clock(actual):
    gamma = actual.record.source.geometry.gamma_bounds
    assert actual.gamma_bounds == gamma
    assert Q(7, 22 * 1023) < gamma.lo <= gamma.hi < Q(50, 157 * 1023)
    assert gamma.hi < Q(1, 3000)
    assert actual.early_deadline * (1000 * gamma.hi**2 * ETA**5) == 1
    assert actual.late_deadline * (gamma.lo**2 * ETA**5) == 4
    assert actual.burn_time * (gamma.lo**2 * ETA) == 1
    assert 0 < actual.burn_time < actual.early_deadline < actual.late_deadline
    assert actual.record.basin.clock_factor == Q(1, 4)
    assert actual.record.basin.held_capacities == (Q(1),) * 54
    assert len(actual.record.basin.weak_weighted_edges) == 144
    assert actual.clock == "structural_tau_elapsed_since_original_attachment"


def test_early_displacement_keeps_transient_permanent_and_hidden_terms(actual):
    strip = actual.record.basin.strip
    g = actual.gamma_bounds.hi
    lambda_rate, rho_rate = g**2 / 200, 19 * g**2 / 800
    # Independently integrate both decaying torque channels. Their g^2 factors
    # cancel against their own rates; their total is not a pointwise error.
    impulse = (
        Q(3, 10)
        * g**2
        * (
            5 * ETA * strip.transient_coefficient / lambda_rate
            + Q(4, 3) * ETA * strip.fine_initial_upper / rho_rate
        )
    )
    assert impulse == strip.transient_displacement_upper
    permanent = 6_000_000 * ETA**7
    rate = Q(3, 10) * g**2 * (1000 * ETA**5 + permanent)
    assert rate > 0 and permanent == actual.permanent_torque_upper > 0
    assert (
        actual.early_displacement_upper
        == actual.relative_form_upper + impulse + rate * actual.early_deadline
    )
    assert actual.early_displacement_upper < Q(31, 100)
    assert (
        actual.early_margin
        == actual.initial_distance_lower - TOLERANCE - actual.early_displacement_upper
    )


def test_burn_and_exact_taylor_bound_do_not_evaluate_huge_exponentials(actual):
    g = actual.gamma_bounds.lo
    lam, rho = g**2 / 200, 19 * g**2 / 800
    assert actual.burn_exponent_lower == lam * actual.burn_time == 500
    assert rho >= lam and rho * actual.burn_time >= 500
    # Every factor in 20! is at most 20. This independent coarse bound suffices
    # for the torque allowance, without evaluating exp(-lambda*T_late).
    assert 0 < actual.decay_upper < Q(1, 25**20)
    strip = actual.record.basin.strip
    coarse = (5 * ETA * Q(21, 100) + Q(4, 3) * ETA / 10**8) / 25**20
    assert strip.transient_coefficient < Q(21, 100)
    assert actual.transient_torque_upper < coarse < ETA**5 / 10
    assert actual.transient_torque_upper > 0
    assert actual.gamma_bounds.lo**2 * actual.late_deadline / 200 > 4096


def test_whole_annulus_uses_endpoint_concavity_and_restores_persistent_correction(
    actual,
):
    lower, upper = Q(499, 200000), Q(627001, 200000)
    assert 0 < lower < actual.annulus_angle_bounds.lo
    assert actual.annulus_angle_bounds.hi < upper < Q(157, 50)
    # On each endpoint, sin(v)>=5v/6 for 0<=v<=1; reflect the upper endpoint
    # using pi>157/50. Concavity then covers the complete interval between them.
    assert min(Q(5, 6) * lower, Q(5, 6) * (Q(157, 50) - upper)) > Q(1, 500)
    assert len(actual.endpoint_sine_bounds) == 2
    assert (
        min(box.lo for box in actual.endpoint_sine_bounds)
        > actual.restoring_sine_minorant
    )
    assert actual.relative_form_upper < Q(1, 10**6)
    assert actual.corrected_tube_radius == Q(1, 2000)
    assert actual.retained_distance_upper < Q(501, 10**6) < TOLERANCE


def test_late_inward_speed_arrival_and_all_future_retention_have_strict_margins(actual):
    eta5 = ETA**5
    lower_speed = Q(3, 10) * (
        400 * actual.restoring_sine_minorant
        - (actual.permanent_torque_upper + actual.transient_torque_upper) / eta5
    )
    assert actual.normalized_inward_speed_lower == lower_speed > Q(1, 5)
    g = actual.gamma_bounds.lo
    b = actual.record.basin.strip.outer_radius
    assert actual.travel_time_upper * (g**2 * eta5) == 5 * b
    # In the normalized slow clock the burn allowance is eta^4, which remains
    # present in the frozen deadline; no stationary-source reset removes it.
    assert (
        (actual.burn_time + actual.travel_time_upper) * (g**2 * eta5)
        == ETA**4 + 5 * b
        < 4
    )
    assert actual.burn_time + actual.travel_time_upper < actual.late_deadline
    assert all(margin > 0 for _, margin in actual.late_margins)


@pytest.mark.parametrize("label", (0, 1))
def test_frozen_source_and_full_readout_families_have_both_time_windows(actual, label):
    truth, _, _, _, interval = _uniform_inverse(actual.record.source, label)
    assert (interval - truth).abs_max < TOLERANCE
    record = _conditional_record(actual, interval)
    result = owner._time_admitted_weak_contact_record(record)
    assert result.early_certified and result.late_certified
    assert result.status == "certified_early_exclusion_and_late_retention"
    assert result.initial_distance_lower > Q(599, 1000)
    assert result.early_margin > Q(28, 100)
    assert result.retained_distance_upper < TOLERANCE
    assert result.early_deadline == actual.early_deadline
    assert result.late_deadline == actual.late_deadline
    assert result.record.limiting_relative_origin_turns == Q(-label, 5)


@pytest.mark.parametrize("label", (0, 1))
def test_representative_primitive_readings_get_no_truth_or_family_flag(
    source_wiring, label
):
    truth, _, _, _, _ = _uniform_inverse(source_wiring, label)
    h18 = source_wiring.outer_one_host_phase_offset_bounds[2]
    point_gap = truth - h18.midpoint
    reading = cos(point_gap).midpoint, sin(point_gap).midpoint
    result = _assess(quadratures=reading)
    assert result.early_certified and result.late_certified
    assert result.record.quadratures == reading
    assert result.record.registration_label == label


def test_zero_early_margin_abstains_while_late_bound_remains_available(actual):
    # A strictly wider, still valid Gamma creates an exact rational comparison
    # boundary. The source and time policy are unchanged; no record is fabricated
    # for the primitive API and no actual arrival is inferred from this test.
    cost = TOLERANCE + actual.early_displacement_upper - actual.relative_form_upper
    scale = 2**40
    ceiling = Q(-((-cost.numerator * scale) // cost.denominator), scale)
    distance = ceiling + Q(1, 2**24)
    gamma_wider = distance - cost
    assert actual.relative_form_upper < gamma_wider < Q(1, 10**6)
    record = _conditional_record(actual, I(distance))
    strip = replace(record.basin.strip, relative_form_upper=gamma_wider)
    record = replace(record, basin=replace(record.basin, strip=strip))
    result = owner._time_admitted_weak_contact_record(record)
    assert result.early_margin == 0
    assert not result.early_certified and result.late_certified
    assert result.status == "certified_late_retention_only"


def test_late_sufficient_bound_can_fail_without_removing_early_exclusion(actual):
    record = _conditional_record(actual, I(Q(3, 5)))
    # This weaker correction bound remains true but fails the selected late
    # comparison policy. Its failure is not a delayed or escaping trajectory.
    strip = replace(record.basin.strip, relative_form_upper=Q(2, 10**6))
    record = replace(record, basin=replace(record.basin, strip=strip))
    result = owner._time_admitted_weak_contact_record(record)
    assert result.early_certified and not result.late_certified
    assert result.status == "certified_early_exclusion_only"
    assert "persistent_correction_budget" in result.unavailable_reasons


@pytest.mark.parametrize(
    "real,imaginary,label,index",
    (
        (Q(3, 5), Q(4, 5), 1, 1),
        (Q(3, 5), Q(-4, 5), 4, -1),
        (Q(-3, 5), Q(4, 5), 2, 2),
        (Q(-3, 5), Q(-4, 5), 3, -2),
    ),
)
def test_selected_center_keeps_its_signed_lift_across_all_quadrants(
    source_wiring, real, imaginary, label, index
):
    result = _assess(quadratures=(real, imaginary))
    assert result.record.registration_label == label and result.late_certified
    assert result.record.limiting_relative_origin_turns == Q(-index, 5)
    pi = pi_interval()
    exact_endpoints = sorted((Q(2 * index, 5) * pi.lo, Q(2 * index, 5) * pi.hi))
    assert result.selected_origin_center_bounds.lo <= exact_endpoints[0]
    assert exact_endpoints[1] <= result.selected_origin_center_bounds.hi
    assert 0 <= result.initial_distance_lower < Q(5, 8)


@pytest.mark.parametrize(
    "reading,status",
    (
        ((0, 0), "incompatible_observation"),
        ((-1, 0), "unavailable_argument"),
        ((Q(-1), Q(1, 10000) + Q(1, 2**100)), "unavailable_lift"),
    ),
)
def test_unavailable_records_never_gain_time_claims(source_wiring, reading, status):
    result = _assess(quadratures=reading)
    assert result.record.status == status and result.status == "unavailable_record"
    assert not result.early_certified and not result.late_certified
    assert result.early_deadline is result.late_deadline is None


def test_uncertified_registration_has_no_timing(source_wiring):
    angle = pi_interval() / 5
    result = _assess(quadratures=(cos(angle).midpoint, sin(angle).midpoint))
    assert result.record.status == "unavailable_registration"
    assert result.status == "unavailable_record"
    assert result.gamma_bounds is None


def test_failed_source_cannot_gain_a_deadline(actual, monkeypatch):
    bad = replace(
        actual.record.source,
        source_certified=False,
        source_admission_flags=(False,)
        + actual.record.source.source_admission_flags[1:],
    )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: bad
    )
    result = _assess()
    assert result.record.status == "unavailable_source"
    assert result.status == "unavailable_record"
    assert result.early_deadline is result.late_deadline is None


@pytest.mark.parametrize("position", range(14))
def test_wrapper_preserves_all_primitive_admission_before_source(monkeypatch, position):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached source rebuilding")

    monkeypatch.setattr(owner, "_assess_organized_external_interaction", forbidden)
    forms, phases, reading = [ERROR] * 6, [ERROR] * 6, [Q(1), Q(0)]
    if position < 6:
        forms[position] = True
    elif position < 12:
        phases[position - 6] = True
    else:
        reading[position - 12] = True
    with pytest.raises((TypeError, ValueError)):
        _assess(form_error_bounds=forms, phase_error_bounds=phases, quadratures=reading)


@pytest.mark.parametrize(
    "keyword",
    (
        "record",
        "source",
        "beta",
        "family",
        "donor_origin_bounds",
        "tolerance",
        "early_deadline",
        "late_deadline",
        "burn_time",
        "readout_error",
    ),
)
def test_timing_reader_has_no_truth_report_or_policy_override(keyword):
    with pytest.raises(TypeError, match="unexpected keyword"):
        _assess(**{keyword: Q(0)})
