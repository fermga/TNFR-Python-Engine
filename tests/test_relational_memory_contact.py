"""Finite-duration contact contracts without ODE or event execution."""

import math
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from tnfr.physics.relational_memory_contact import (
    certify_relational_memory_contact,
    certify_relational_memory_retention,
)


@pytest.fixture(scope="module")
def certificate():
    return certify_relational_memory_contact()


def test_default_contact_certifies_accumulated_response_against_recovering_control(
    certificate,
):
    report = certificate
    assert report.admitted and report.status == "admitted"
    assert report.readout.admitted and report.readout.decay_blocks == 40
    assert report.duration == Q(1, 4096)
    assert report.end_time == Q(165120) + report.duration
    assert report.contact_signs_separated and report.contact_induced_signs_separated
    assert report.windings_preserved
    assert report.unavailable_reasons == ()
    assert all(passed for _, passed in report.proof_checks)
    for case, initial in zip(report.cases, report.readout.cases):
        assert case.initial_state_radius > report.readout.state_norm_upper_bound
        assert (
            case.initial_state_radius
            < case.whole_time_state_radius_upper_bound
            < report.neighborhood_radius
        )
        assert case.state_change_upper_bound > 0
        assert case.accumulated_form_remainder_bound > 0
        assert case.winding_preserved
        assert case.internal_acute_margin_lower_bound > 0
        assert case.bridge_acute_margin_lower_bound > 0
        assert case.unavailable_reasons == ()
        # Retain an explicit continuous remainder around the initial-rate
        # integral. A successful certificate is not an Euler assertion.
        assert case.left_contact_form_change_bounds[0] < (
            report.duration * initial.left_contact_form_rate_bounds[0]
        )
        assert case.left_contact_form_change_bounds[1] > (
            report.duration * initial.left_contact_form_rate_bounds[1]
        )
        assert case.left_no_contact_form_change_bounds[0] < 0
        assert case.left_no_contact_form_change_bounds[1] > 0
        assert case.right_no_contact_form_change_bounds == (0, 0)
        assert case.right_contact_induced_form_change_bounds == (
            case.right_contact_form_change_bounds
        )
        assert case.left_contact_induced_form_change_bounds[0] < (
            case.left_contact_form_change_bounds[0]
        )
        assert case.left_contact_induced_form_change_bounds[1] > (
            case.left_contact_form_change_bounds[1]
        )
        left = case.left_contact_induced_form_change_bounds
        right = case.right_contact_induced_form_change_bounds
        if case.form_sign == 1:
            assert left[1] < 0 < right[0]
        else:
            assert right[1] < 0 < left[0]


def test_integrated_remainder_encloses_independent_exponential_value(certificate):
    s = pytest.importorskip("sympy")
    report = certificate
    step = report.field_growth_constant * report.duration
    argument = s.Rational(step.numerator, step.denominator)
    # The exact Taylor owner's interval is much narrower than binary64.
    # This independent value uses sufficient precision to resolve that width.
    independent_exp = Q(str(s.N(s.exp(argument), 500)))
    assert report.exponential_bounds[0] < independent_exp < report.exponential_bounds[1]
    for case in report.cases:
        expected = case.initial_state_radius * (independent_exp - 1 - step)
        assert 0 < expected < case.accumulated_form_remainder_bound
        assert case.initial_state_radius * step**2 / 2 < expected
        assert case.accumulated_form_remainder_bound < (
            case.initial_state_radius * step**2
        )


def test_insertion_work_is_inherited_separately_from_continuous_loss(certificate):
    report = certificate
    for case, initial in zip(report.cases, report.readout.cases):
        assert case.contact_storage_bounds == initial.contact_storage_bounds
        assert 0 < case.contact_storage_bounds[0] < case.contact_storage_bounds[1]
        assert case.continuous_loss_upper_bound > 0
        assert case.contact_storage_bounds[1] <= report.required_event_work_upper_bound
    assert report.required_event_work_upper_bound == max(
        case.contact_storage_bounds[1] for case in report.cases
    )
    # The field contains a required event-work bound, not a claim that a
    # reservoir exists or that continuous dissipation supplies insertion.
    assert "event_insertion_storage_is_separate_from_continuous_dissipation" in (
        report.scope
    )


def test_long_admitted_duration_retains_bounds_but_abstains_on_response_signs():
    report = certify_relational_memory_contact(duration=Q(1, 3))
    assert report.duration == report.duration_limit
    assert report.readout.admitted and report.windings_preserved
    assert not report.admitted and report.status == "unavailable"
    assert not report.contact_signs_separated
    assert not report.contact_induced_signs_separated
    for case in report.cases:
        assert case.whole_time_state_radius_upper_bound < report.neighborhood_radius
        for bounds in (
            case.left_contact_form_change_bounds,
            case.right_contact_form_change_bounds,
            case.left_contact_induced_form_change_bounds,
            case.right_contact_induced_form_change_bounds,
        ):
            assert bounds[0] < 0 < bounds[1]
        assert case.unavailable_reasons


def test_exact_short_duration_retains_nonzero_signed_accumulated_response():
    duration = Q(1, 2**80)
    report = certify_relational_memory_contact(duration=duration)
    assert report.duration == duration and report.admitted
    for case in report.cases:
        assert case.accumulated_form_remainder_bound > 0
        assert case.left_contact_form_change_bounds != (0, 0)
        assert case.right_contact_form_change_bounds != (0, 0)


def test_contact_certificate_does_not_execute_a_flow_or_event(monkeypatch):
    from tnfr.dynamics import relational
    from tnfr.physics import relational_observations

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "static certificate attempted live field or contact execution"
        )

    monkeypatch.setattr(relational, "evaluate_relational_exchange", forbidden)
    monkeypatch.setattr(relational, "step_relational_exchange", forbidden)
    monkeypatch.setattr(
        relational_observations, "observe_relational_attachment", forbidden
    )
    assert certify_relational_memory_contact().admitted


@pytest.mark.parametrize("duration", (0, -1, Q(1, 3) + Q(1, 2**60), math.inf, math.nan))
def test_out_of_domain_contact_durations_reject(duration):
    with pytest.raises(ValueError, match="duration"):
        certify_relational_memory_contact(duration=duration)


def test_boolean_duration_is_not_a_structural_clock():
    with pytest.raises(TypeError, match="duration"):
        certify_relational_memory_contact(duration=True)


def test_represented_float_duration_uses_its_exact_binary_value():
    report = certify_relational_memory_contact(duration=0.0002)
    assert report.duration == Q(0.0002)
    assert report.admitted


@pytest.fixture(scope="module")
def retention():
    return certify_relational_memory_retention()


def test_fixed_cut_retains_opposite_receiver_means_and_captures_both_rings(retention):
    report = retention
    assert report.admitted and report.status == "admitted"
    assert report.contact.duration == Q(1, 4096)
    assert report.receiver_mean_signs_separated and report.both_rings_captured
    assert report.removal_storage_nonincrease_certified
    assert report.removal_storage_strict_decrease_certified
    assert report.unavailable_reasons == ()
    geometry = report.contact.readout.memory.cases[0].asymptotic
    assert report.cycle_reference_storage_bounds == geometry.reference_storage_bounds
    assert report.cycle_capture_barrier_bounds == geometry.capture_barrier_bounds
    for case in report.cases:
        assert case.no_contact_receiver_mean_bounds == (0, 0)
        assert case.both_rings_captured and case.receiver_mean_sign_certified
        assert case.isolated_cycle_excess_storage_upper_bound > 0
        assert case.isolated_cycle_capture_margin_lower_bound > Q(13, 1000)
        assert (
            case.isolated_cycle_storage_upper_bound
            < report.cycle_capture_barrier_bounds[0]
        )
        assert case.unavailable_reasons == ()
        if case.form_sign == 1:
            assert case.receiver_persistent_mean_bounds[0] > 0
        else:
            assert case.receiver_persistent_mean_bounds[1] < 0
    assert "receiver_arithmetic_form_mean_conserved_after_removal" in report.scope


def test_receiver_mean_keeps_all_five_coordinate_remainders(retention):
    contact = retention.contact
    for case, episode, initial in zip(
        retention.cases, contact.cases, contact.readout.cases
    ):
        leading = tuple(
            contact.duration * value / 5
            for value in initial.right_contact_form_rate_bounds
        )
        remainder = episode.accumulated_form_remainder_bound
        # Only the receiver port moves initially, but the other four nodes
        # have nonzero possible accumulated remainders during the episode.
        lower_errors = tuple(-remainder for _ in range(5))
        upper_errors = tuple(remainder for _ in range(5))
        assert case.receiver_persistent_mean_bounds == (
            leading[0] + sum(lower_errors) / 5,
            leading[1] + sum(upper_errors) / 5,
        )
        assert case.receiver_persistent_mean_bounds[0] < leading[0] - remainder / 5
        assert case.receiver_persistent_mean_bounds[1] > leading[1] + remainder / 5


def test_cut_storage_encloses_independent_endpoint_costs_and_stays_separate(retention):
    s = pytest.importorskip("sympy")
    for case, episode in zip(retention.cases, retention.contact.cases):
        assert case.removal_storage_change_bounds == tuple(
            -value for value in reversed(case.removal_bridge_storage_bounds)
        )
        assert case.removal_storage_change_bounds[1] < 0
        assert episode.contact_storage_bounds[0] > 0
        assert case.removal_storage_change_bounds != tuple(
            -value for value in reversed(episode.contact_storage_bounds)
        )
        # Endpoint cost, not an assumed refund of the initial insertion.
        form_values = (Q(0), *case.removal_bridge_form_difference_bounds)
        for form in form_values:
            for phase in case.removal_bridge_phase_difference_bounds:
                delta = s.Rational(phase.numerator, phase.denominator)
                cost = (
                    s.Rational(form.numerator, form.denominator) ** 2 / 2
                    + 1
                    - s.cos(delta)
                )
                value = Q(str(s.N(cost, 100)))
                assert case.removal_bridge_storage_bounds[0] <= value
                assert value <= case.removal_bridge_storage_bounds[1]


def test_unresolved_mean_does_not_erase_component_capture_or_removal_evidence(
    retention, monkeypatch
):
    from tnfr.physics import relational_memory_contact as owner

    # A coarser, still valid accumulated remainder can lose the mean sign.
    # Other valid energy/cut evidence must remain independently available.
    coarse_cases = tuple(
        replace(case, accumulated_form_remainder_bound=Q(1, 10**12))
        for case in retention.contact.cases
    )
    coarse_contact = replace(
        retention.contact,
        cases=coarse_cases,
        status="unavailable",
        unavailable_reasons=("coarse_contact_response_not_resolved",),
    )
    monkeypatch.setattr(
        owner, "certify_relational_memory_contact", lambda **_: coarse_contact
    )
    report = owner.certify_relational_memory_retention()
    assert not report.admitted and not report.receiver_mean_signs_separated
    assert report.both_rings_captured and report.removal_storage_nonincrease_certified
    assert report.unavailable_reasons == (
        "persistent_receiver_mean_sign_separation_not_certified",
    )
    for case in report.cases:
        assert (
            case.receiver_persistent_mean_bounds[0]
            < 0
            < case.receiver_persistent_mean_bounds[1]
        )
        assert (
            case.both_rings_captured and case.removal_storage_strict_decrease_certified
        )


def test_retention_never_materializes_endpoint_graph_or_executes_cut(monkeypatch):
    from tnfr.dynamics import relational
    from tnfr.physics import relational_capture, relational_observations

    def forbidden(*args, **kwargs):
        raise AssertionError("static retention attempted field, graph capture or event")

    monkeypatch.setattr(relational, "evaluate_relational_exchange", forbidden)
    monkeypatch.setattr(relational, "step_relational_exchange", forbidden)
    monkeypatch.setattr(
        relational_capture, "certify_relational_cycle_capture", forbidden
    )
    monkeypatch.setattr(relational_capture, "observe_relational_detachment", forbidden)
    monkeypatch.setattr(relational_observations, "observe_relational_reset", forbidden)
    assert certify_relational_memory_retention().admitted


def test_unresolved_capture_retains_independent_mean_and_cut_evidence(
    retention, monkeypatch
):
    from tnfr.physics import relational_memory_contact as owner

    # Enlarging this valid tube loses the sufficient energy margin without
    # changing the separately proved winding, mean or endpoint displacement.
    coarse_contact = replace(
        retention.contact,
        cases=tuple(
            replace(case, whole_time_state_radius_upper_bound=Q(1, 10))
            for case in retention.contact.cases
        ),
    )
    monkeypatch.setattr(
        owner, "certify_relational_memory_contact", lambda **_: coarse_contact
    )
    report = owner.certify_relational_memory_retention()
    assert not report.admitted and not report.both_rings_captured
    assert report.receiver_mean_signs_separated
    assert report.removal_storage_strict_decrease_certified
    assert report.unavailable_reasons == ("both_isolated_cycle_captures_not_certified",)
    for case in report.cases:
        assert case.isolated_cycle_capture_margin_lower_bound < 0
        assert case.receiver_mean_sign_certified
