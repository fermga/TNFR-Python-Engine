"""Hidden-state comparison and source gates without the reserved assessment.

The fresh history fixture uses the older source budget. The prospective source
box is evaluated only by its committed instrument, never by these tests.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_electrical_source as owner
from tnfr.physics._organized_joint_resources import _assess_organized_joint_resources


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def original_source():
    return _assess_organized_joint_resources(
        form_error_bounds=(Q(1, 10**10),) * 6,
        phase_error_bounds=(Q(1, 10**10),) * 6,
        quarter_dwell=Q(2 * 10**13),
        recovered_radius=Q(1, 10**70),
        decay_power=512,
        sequential_readout_error=Q(1, 10**15),
        contact_readout_error=Q(1, 10**10),
    )


@pytest.fixture
def assess(original_source, monkeypatch):
    def run(source=original_source):
        monkeypatch.setattr(
            owner, "_assess_organized_joint_resources", lambda **kw: source
        )
        return owner._assess_organized_electrical_source()

    return run


def _transport(**changes):
    arguments = dict(
        initial_form=Q(3, 7),
        initial_phase=Q(5, 11),
        gamma_upper=Q(1),
        laplacian_upper=Q(4),
        duration=Q(1, 5),
    )
    arguments.update(changes)
    return owner._transport_hidden_deviations(**arguments)


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


@pytest.mark.parametrize("duration", (Q(0), Q(1, 1000), Q(1, 5), Q(9, 10)))
@pytest.mark.parametrize(
    "initial", ((Q(0), Q(0)), (Q(0), Q(2)), (Q(3), Q(0)), (Q(3, 7), Q(5, 11)))
)
def test_rational_majorant_bounds_the_continuous_positive_comparison(duration, initial):
    report = _transport(
        initial_form=initial[0], initial_phase=initial[1], duration=duration
    )
    context = mpmath.mp.clone()
    context.dps = 100
    a, z, time = (_mp(context, value) for value in (*initial, duration))
    # The independent equality system a'=z, z'=a realizes the comparison
    # kernel before form dissipation. Its exact flow uses cosh and sinh.
    actual_a = a * context.cosh(time) + z * context.sinh(time)
    actual_z = z * context.cosh(time) + a * context.sinh(time)
    assert actual_a <= _mp(context, report.form_upper)
    assert actual_z <= _mp(context, report.phase_upper)


@pytest.mark.parametrize("parameter", ("duration", "gamma_upper", "laplacian_upper"))
def test_zero_coupling_retains_both_exact_initial_budgets(parameter):
    report = _transport(**{parameter: Q(0)})
    assert report.form_upper == Q(3, 7)
    assert report.phase_upper == Q(5, 11)
    assert report.coupling == 0 and report.domain_margin == 1


def test_unrepresentably_small_positive_budgets_are_not_lost():
    report = _transport(initial_form=Q(1, 10**500), initial_phase=Q(0))
    assert isinstance(report.form_upper, Q) and report.form_upper > 0
    assert isinstance(report.phase_upper, Q) and report.phase_upper > 0
    with pytest.raises(FrozenInstanceError):
        report.form_upper = Q(0)


@pytest.mark.parametrize(
    "parameter",
    ("initial_form", "initial_phase", "gamma_upper", "laplacian_upper", "duration"),
)
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), "0.1", 1j, Q(-1)))
def test_invalid_primitives_are_rejected_before_comparison(parameter, value):
    with pytest.raises((ValueError, TypeError)):
        _transport(**{parameter: value})


@pytest.mark.parametrize("duration", (Q(1), Q(2)))
def test_nonpositive_comparison_domain_is_unavailable(duration):
    with pytest.raises(ValueError, match="comparison requires"):
        _transport(duration=duration)


def test_actual_history_is_carried_with_signed_storage_and_separate_work(
    assess, original_source
):
    report = assess()
    assert report.source is original_source
    assert report.admitted and not report.unresolved_conditions
    assert (
        report.observation_hidden.initial_form
        == original_source.outer_resources.full_form_error_upper_bound
    )
    assert (
        report.observation_hidden.initial_phase
        == original_source.outer_resources.full_phase_error_upper_bound
    )
    assert (
        report.contact_hidden.initial_form
        == report.observation_hidden.form_upper + original_source.recovered_radius
    )
    assert (
        report.contact_hidden.initial_phase
        == report.observation_hidden.phase_upper + original_source.recovered_radius
    )
    assert 0 < report.mean_form_error_upper < original_source.contact_readout_error
    assert 0 < report.mean_phase_error_upper < report.mean_form_error_upper
    assert report.storage_remainder_lower < 0 < report.storage_remainder_upper
    assert report.hidden_loss_rate_upper > 0
    assert report.collective_hidden_work_rate_upper > 0
    assert (
        report.integrated_hidden_loss_upper
        == report.contact_horizon * report.hidden_loss_rate_upper
    )
    assert (
        report.integrated_collective_hidden_work_upper
        == report.contact_horizon * report.collective_hidden_work_rate_upper
    )
    assert 2 * report.regional_closure_upper == report.closure_l1_upper
    with pytest.raises(FrozenInstanceError):
        report.admitted = False


def test_declared_recovery_radius_is_forwarded_before_source_rebuilding(
    monkeypatch, original_source
):
    calls = []

    def rebuilt(**arguments):
        calls.append(arguments)
        return original_source

    monkeypatch.setattr(owner, "_assess_organized_joint_resources", rebuilt)
    report = owner._assess_organized_electrical_source()
    assert len(calls) == 1
    assert calls[0]["recovered_radius"] == Q(1, 10**8)
    assert calls[0]["quarter_dwell"] == Q(8 * 10**10)
    assert calls[0]["decay_power"] == 80
    assert report.source.recovered_radius == Q(1, 10**70)


@pytest.mark.parametrize("truthy", (False, 1))
@pytest.mark.parametrize(
    "stage", ("recovery", "outer_one", "outer_zero", "work", "observation")
)
def test_each_actual_history_flag_requires_true(assess, original_source, stage, truthy):
    if stage == "recovery":
        source = replace(
            original_source,
            recoveries=(
                replace(original_source.recoveries[0], recovery_certified=truthy),
            )
            + original_source.recoveries[1:],
        )
    elif stage == "observation":
        source = replace(
            original_source,
            observation=replace(original_source.observation, admitted=truthy),
        )
    else:
        field = {
            "outer_one": "conditional_capture_certified",
            "outer_zero": "control_conditional_capture_certified",
            "work": "attachment_budget_certified",
        }[stage]
        source = replace(
            original_source,
            outer_resources=replace(original_source.outer_resources, **{field: truthy}),
        )
    report = assess(source)
    assert report.form_resolution_margin > 0
    assert not report.admitted and report.unresolved_conditions


def test_missing_recovery_cannot_be_replaced_by_other_passing_sources(
    assess, original_source
):
    report = assess(
        replace(original_source, recoveries=original_source.recoveries[:-1])
    )
    assert not report.admitted
    assert "six_actual_recoveries" in report.unresolved_conditions


@pytest.mark.parametrize("name", sorted(owner._HISTORY_MARGINS))
def test_each_numeric_history_premise_is_strict(assess, original_source, name):
    margins = tuple(
        (key, Q(0) if key == name else value)
        for key, value in original_source.sufficient_margins
    )
    report = assess(replace(original_source, sufficient_margins=margins))
    assert not report.admitted
    assert name in report.unresolved_conditions


@pytest.mark.parametrize("duplicate", (False, True))
def test_margin_inventory_requires_every_unique_history_premise(
    assess, original_source, duplicate
):
    margins = original_source.sufficient_margins
    changed = margins + (margins[0],) if duplicate else margins[1:]
    report = assess(replace(original_source, sufficient_margins=changed))
    assert not report.admitted
    assert "complete_history_margin_inventory" in report.unresolved_conditions


def test_boolean_history_margin_is_not_a_physical_scalar(assess, original_source):
    margins = (
        (original_source.sufficient_margins[0][0], True),
    ) + original_source.sufficient_margins[1:]
    with pytest.raises((TypeError, ValueError)):
        assess(replace(original_source, sufficient_margins=margins))


def test_unrelated_old_response_failure_does_not_reject_source_description(
    assess, original_source
):
    margins = tuple(
        (name, value if name in owner._HISTORY_MARGINS else Q(-1))
        for name, value in original_source.sufficient_margins
    )
    report = assess(
        replace(original_source, admitted=False, sufficient_margins=margins)
    )
    assert report.admitted
    assert set(dict(report.prerequisite_margins)) == owner._HISTORY_MARGINS


@pytest.mark.parametrize(
    "field,value,condition",
    (
        ("clock_factor", Q(1, 5), "complete_contact_law"),
        ("held_capacities", (Q(2),) * 54, "complete_contact_law"),
        ("phase_domain", "acute", "complete_contact_law"),
        ("weighted_edges", (), "same_slot_product_support"),
        ("node_order", tuple(reversed(range(54))), "same_slot_product_support"),
        ("slot_offsets", (1,) * 5, "same_slot_product_support"),
    ),
)
def test_complete_contact_geometry_is_required(
    assess, monkeypatch, field, value, condition
):
    derive = owner._derive_organized_slot_registration
    monkeypatch.setattr(
        owner,
        "_derive_organized_slot_registration",
        lambda **kw: replace(derive(**kw), **{field: value}),
    )
    report = assess()
    assert not report.admitted
    assert condition in report.unresolved_conditions


@pytest.mark.parametrize(
    "field,value",
    (
        ("marked_child_order", tuple(reversed(range(5)))),
        ("event_multipliers", (1, -1, 0)),
        ("window_command_multipliers", (0, 1, 1, 0)),
        ("total_elapsed_scaled_time", Q(1, 10000)),
    ),
)
def test_actual_observation_word_cannot_be_shortened_or_changed(
    assess, original_source, field, value
):
    report = assess(
        replace(
            original_source,
            observation=replace(original_source.observation, **{field: value}),
        )
    )
    assert not report.admitted
    assert "complete_observation_word" in report.unresolved_conditions


def test_zero_readout_budget_does_not_certify_a_nonzero_closure_error(
    assess, original_source
):
    report = assess(replace(original_source, contact_readout_error=Q(0)))
    assert not report.admitted
    assert (
        "form_error_below_original_raw_readout_budget" in report.unresolved_conditions
    )


@pytest.mark.parametrize(
    "parameter", ("source", "slot_offsets", "horizon", "readout_error", "trajectory")
)
def test_reserved_assessor_accepts_no_retuned_input(parameter):
    with pytest.raises(TypeError):
        owner._assess_organized_electrical_source(**{parameter: None})
