"""Source admission and static uncertainty-aware weak-contact decisions.

The source assessor rebuilds existing analytical preparation bounds once.
Neither a retained producer nor a trajectory supplies a destination here.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_contact_maintenance as owner

ERROR = Q(1, 10**10)
ETA = Q(1, 100000)
INNER, OUTER = Q(5, 8), Q(627, 1000)


@pytest.fixture(scope="module", autouse=True)
def no_execution_or_retained_response():
    def forbidden(*args, **kwargs):
        pytest.fail("basin controls cannot execute events or trajectories")

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
    return owner._assess_organized_weak_contact_basin(
        **(
            dict(
                form_error_bounds=(ERROR,) * 6,
                phase_error_bounds=(ERROR,) * 6,
                donor_origin_bounds=(INNER, INNER),
            )
            | changes
        )
    )


def _strip(**changes):
    return owner._bound_organized_weak_contact_strip(
        **(
            dict(
                gamma_upper=Q(1, 3000),
                contact_strength=ETA,
                shape_initial_upper=Q(1, 5),
                fine_initial_upper=Q(1, 10**8),
                inner_radius=INNER,
                outer_radius=OUTER,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def actual():
    # This one direct consumer call rebuilds all six independent analytic
    # source histories at all twelve original error ceilings.
    return _assess()


@pytest.fixture
def source_wiring(monkeypatch, actual):
    # Later cases test interval decisions and failure propagation against
    # the same freshly built source. The public API accepts no report input.
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: actual.source
    )
    return actual.source


def test_frozen_positive_origin_uses_actual_full_source_and_weak_law(actual):
    assert actual.status == "certified_unique_registration"
    assert actual.registration_label == 0
    assert actual.limiting_relative_origin_turns == 0
    assert actual.source_certified and actual.strip_certified
    assert actual.donor_origin_bounds == (INNER, INNER) and INNER > Q(3, 5)
    source = actual.source
    assert source.form_error_bounds == source.phase_error_bounds == (ERROR,) * 6
    assert source.relative_preparation_dimension == 96
    assert len(source.host_acquisition.child_sources) == 5
    assert all(
        s.recovery_certified
        for s in (*source.host_acquisition.child_sources, source.donor_recovery)
    )
    assert source.observation_protocol.total_elapsed_scaled_time == Q(1, 5000)
    assert source.observation_protocol.event_multipliers == (1, -2, 1)
    assert actual.initial_relative_form == 0
    assert len(actual.weak_weighted_edges) == 144
    assert sum(w == 1 for _, _, w in actual.weak_weighted_edges) == 99
    assert sum(w == ETA for _, _, w in actual.weak_weighted_edges) == 45
    assert actual.weak_weighted_degrees == (4 + ETA,) * 45 + (2 + 5 * ETA,) * 9
    assert actual.held_capacities == (Q(1),) * 54 and actual.clock_factor == Q(1, 4)
    assert actual.initial_storage_excess_upper < Q(3, 250)
    assert actual.attachment_work_upper < Q(901, 10**6)
    assert actual.source_shape_initial_upper < Q(1, 5)
    assert actual.source_fine_initial_upper < Q(1, 10**8)
    assert all(value > 0 for _, value in actual.source_margins)
    with pytest.raises(FrozenInstanceError):
        actual.registration_label = 1


def test_fixed_mixed_interval_contains_two_exact_different_destinations(source_wiring):
    result = _assess(donor_origin_bounds=(INNER, Q(79, 125)))
    assert result.status == "certified_mixed_registrations"
    assert result.registration_label is result.limiting_relative_origin_turns is None
    witnesses = {(w.center_index, w.offset): w for w in result.witnesses}
    first, second = witnesses[(0, INNER)], witnesses[(1, -INNER)]
    assert first.origin_bounds == I(INNER)
    assert second.origin_bounds == 2 * pi_interval() / 5 - INNER
    assert first.registration_label == 0 and second.registration_label == 1
    assert second.limiting_relative_origin_turns == Q(-1, 5)
    assert INNER < second.origin_bounds.lo < second.origin_bounds.hi < Q(79, 125)
    assert all(
        INNER <= w.origin_bounds.lo <= w.origin_bounds.hi <= Q(79, 125)
        for w in result.witnesses
    )


@pytest.mark.parametrize(
    "interval",
    (
        (Q(628, 1000), Q(629, 1000)),
        (Q(6283, 10000), Q(6283, 10000)),
    ),
)
def test_missing_uniform_certificate_does_not_invent_mixed_outcomes(
    source_wiring, interval
):
    result = _assess(donor_origin_bounds=interval)
    assert result.status == "unavailable_insufficient_certificate"
    assert result.registration_label is None
    assert len({w.registration_label for w in result.witnesses}) < 2


@pytest.mark.parametrize(
    "interval,label,turns",
    (
        ((Q(-1, 100), Q(1, 100)), 0, Q(0)),
        ((Q(6, 5), Q(13, 10)), 1, Q(-1, 5)),
        ((Q(-13, 10), Q(-6, 5)), 4, Q(1, 5)),
        ((Q(12, 5), Q(13, 5)), 2, Q(-2, 5)),
        ((Q(-13, 5), Q(-12, 5)), 3, Q(2, 5)),
    ),
)
def test_principal_lift_labels_retain_destination_sign(
    source_wiring, interval, label, turns
):
    result = _assess(donor_origin_bounds=interval)
    assert result.status == "certified_unique_registration"
    assert result.registration_label == label
    assert result.limiting_relative_origin_turns == turns


def _forbidden_source(**kwargs):
    pytest.fail("unadmitted primitive reached source rebuilding")


@pytest.mark.parametrize(
    "interval",
    (
        (Q(3), Q(4)),
        (Q(-4), Q(-3)),
        (Q(7), Q(7)),
    ),
)
def test_valid_unsupported_chart_abstains_before_source_rebuilding(
    monkeypatch, interval
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    result = _assess(donor_origin_bounds=interval)
    assert result.status == "unavailable_chart" and result.source is None
    assert not result.chart_certified and result.registration_label is None


def test_interval_touching_an_unresolved_pi_enclosure_is_not_wrapped(monkeypatch):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    pi = pi_interval()
    result = _assess(donor_origin_bounds=(pi.lo, pi.hi))
    assert result.status == "unavailable_chart"
    assert result.donor_origin_bounds == (pi.lo, pi.hi)


@pytest.mark.parametrize("position", range(12))
@pytest.mark.parametrize(
    "bad", (True, "0", complex(0), float("inf"), Q(-1, 10**90), ERROR + Q(1, 2**600))
)
def test_every_original_allowance_is_admitted_before_source(monkeypatch, position, bad):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    forms, phases = [ERROR] * 6, [ERROR] * 6
    (forms if position < 6 else phases)[position % 6] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(form_error_bounds=forms, phase_error_bounds=phases)


@pytest.mark.parametrize("bad", (True, float("nan"), float("-inf"), "0", complex(0)))
@pytest.mark.parametrize("endpoint", (0, 1))
def test_both_origin_endpoints_are_admitted_before_source(monkeypatch, bad, endpoint):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    interval = [Q(0), Q(0)]
    interval[endpoint] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(donor_origin_bounds=interval)


@pytest.mark.parametrize(
    "field", ("form_error_bounds", "phase_error_bounds", "donor_origin_bounds")
)
@pytest.mark.parametrize("bad", ((), {0: Q(0)}, {Q(0)}, "00"))
def test_unordered_or_malformed_vectors_reject_before_source(monkeypatch, field, bad):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


def test_reversed_origin_interval_rejects_before_source(monkeypatch):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    with pytest.raises(ValueError, match="ordered"):
        _assess(donor_origin_bounds=(Q(1), Q(0)))


def test_binary64_source_boundary_is_not_decimal_rounded(monkeypatch):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    assert Q(1e-10) > ERROR
    with pytest.raises(ValueError, match="original errors"):
        _assess(form_error_bounds=(1e-10,) * 6)


def test_subgrid_origin_outside_closed_inner_radius_does_not_gain_membership(
    source_wiring,
):
    outside = INNER + Q(1, 2**600)
    result = _assess(donor_origin_bounds=(outside, outside))
    assert result.donor_origin_bounds == (outside, outside)
    assert result.registration_label is None
    assert result.status == "unavailable_insufficient_certificate"


@pytest.mark.parametrize("failure", ("source_flag", "host_energy", "clock"))
def test_missing_source_premise_blocks_labels_despite_positive_legacy_verdict(
    actual, monkeypatch, failure
):
    source = actual.source
    if failure == "source_flag":
        changed = replace(
            source,
            source_certified=False,
            source_admission_flags=(False,) + source.source_admission_flags[1:],
        )
    elif failure == "host_energy":
        changed = replace(
            source,
            host_acquisition=replace(
                source.host_acquisition, endpoint_excess_upper_bound=Q(1)
            ),
        )
    else:
        changed = replace(source, geometry=replace(source.geometry, clock_factor=Q(1)))
    changed = replace(changed, admitted=True, comparison_certified=True)
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: changed
    )
    result = _assess()
    assert result.status == "unavailable_source"
    assert result.registration_label is None and not result.source_certified


def test_old_unit_contact_failures_do_not_control_the_new_weak_decision(
    actual, monkeypatch
):
    changed = replace(
        actual.source,
        admitted=False,
        comparison_certified=False,
        identity_certified=False,
        work_certified=False,
    )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: changed
    )
    result = _assess()
    assert result.status == "certified_unique_registration"
    assert result.source_certified and result.registration_label == 0


@pytest.mark.parametrize(
    "keyword",
    (
        "source",
        "source_certified",
        "shape_initial_upper",
        "strip_inner",
        "initial_relative_form",
    ),
)
def test_high_level_api_accepts_no_claimed_source_envelope_or_correlation(
    monkeypatch, keyword
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    with pytest.raises(TypeError):
        _assess(**{keyword: Q(0)})


def test_strip_comparison_keeps_transient_permanent_and_relative_form_terms():
    strip = _strip()
    assert strip.certified
    gamma, e0, f0 = Q(1, 3000), Q(1, 5), Q(1, 10**8)
    decay, fine_decay = Q(1, 200), Q(19, 800)
    transient = e0 + 3 * f0 / (fine_decay - decay)
    plateau = 6000 * ETA**6 / decay
    impulse = Q(3, 10) * ETA * (5 * transient / decay + Q(4, 3) * f0 / fine_decay)
    chi = gamma**2 * (transient + plateau + 200 * ETA**4 + Q(4, 15) * f0)
    assert strip.transient_coefficient == transient
    assert strip.shape_plateau == plateau
    assert strip.relative_form_upper == chi < Q(1, 10**6)
    assert strip.transient_displacement_upper == impulse < Q(1, 1000)
    assert strip.permanent_torque_error_upper == 5 * ETA * plateau > 0
    assert all(value > 0 for _, value in strip.margins)
    # Independent coarse sine argument; the acceptance strip was frozen.
    lower_v = Q(157, 50) - 5 * (OUTER + Q(1, 10**6))
    upper_v = Q(22, 7) - 5 * (INNER - Q(1, 10**6))
    assert 0 < lower_v < upper_v < 1
    assert Q(5, 6) * lower_v > Q(1, 250)
    assert strip.sine_lower > Q(1, 250)
    assert 400 * ETA**5 / 250 > strip.permanent_torque_error_upper
    assert impulse < OUTER - INNER


def test_strict_transient_strip_boundary_does_not_certify():
    reference = _strip()
    a = Q(1, 10)
    boundary = _strip(
        inner_radius=a, outer_radius=a + reference.transient_displacement_upper
    )
    assert not boundary.certified
    assert dict(boundary.margins)["transient_cannot_cross_strip"] == 0
    assert "transient_cannot_cross_strip" in boundary.unavailable_reasons


@pytest.mark.parametrize(
    "changes",
    (
        {"inner_radius": Q(0)},
        {"outer_radius": INNER},
        {"shape_initial_upper": Q(-1)},
        {"fine_initial_upper": True},
        {"gamma_upper": Q(0)},
        {"contact_strength": Q(0)},
    ),
)
def test_invalid_conditional_strip_domains_reject(changes):
    with pytest.raises((TypeError, ValueError)):
        _strip(**changes)


def test_valid_but_unsupported_comparison_domain_is_unavailable():
    for changes, reason in (
        ({"contact_strength": 2 * ETA}, "contact_strength_outside_proved_weak_domain"),
        ({"gamma_upper": Q(1, 1000)}, "gamma_upper_outside_comparison_domain"),
        ({"outer_radius": Q(1)}, "corrected_strip_outside_positive_sine_chamber"),
    ):
        result = _strip(**changes)
        assert not result.certified and reason in result.unavailable_reasons
