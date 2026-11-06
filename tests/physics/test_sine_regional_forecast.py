"""Regional whole-tube observations on unrelated short numerical controls.

These fixtures test reader contracts, not the reserved environmental
preparation, prospective acquisition, or a physical claim.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_forecast import bound_sine_flow
from tnfr.physics.relational_sine_regional import (
    assess_sine_regional_channels,
    assess_sine_regional_organization,
)

CYCLE = (0, 1, 2, 3, 4)
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_weight=1, phase_domain="regular")


@pytest.fixture(scope="module")
def controls():
    neighbors = tuple(((i - 1) % 5, (i + 1) % 5) for i in range(5))
    result = {}
    for name, phase in (
        ("flat", (0,) * 5),
        ("acute", tuple(Q(5 * i, 4) for i in range(5))),
    ):
        result[name] = bound_sine_flow(
            (Q(0),) * 5 + phase + (Q(1),),
            neighbors=neighbors,
            visible_capacity=(Q(1),) * 4,
            model=MODEL,
            observation_time=Q(0),
            end_time=Q(1, 32),
            time_step=Q(1, 64),
            order=4,
        )
        assert result[name].admitted
    return result


def _assess(forecast, **changes):
    return assess_sine_regional_organization(
        forecast,
        **dict(cycle_indices=CYCLE, minimum_duration=Q(1, 32), acute_margin=Q(1, 16))
        | changes,
    )


def test_actual_acute_tubes_certify_a_continuous_window(controls):
    forecast = controls["acute"]
    report = _assess(forecast)
    assert report.outcome == "finite_acute_retention_certified"
    assert report.horizon_complete and report.unresolved == ()
    assert report.certified_windows == ((Q(0), Q(1, 32)),)
    assert report.clock == "structural_t"
    assert all(step.winding == 1 for step in report.steps)
    assert all(step.edge_turn_offsets == (0, 0, 0, 0, -1) for step in report.steps)
    assert all(step.acute_margin_lower_bound > Q(1, 16) for step in report.steps)
    assert all(step.requested_acute_margin_met for step in report.steps)
    assert all(not step.acute_winding_excluded for step in report.steps)
    # There is no boundary when the receiver is the entire support. The
    # enclosure of E(end)-E(initial) must retain the exact zero net work.
    assert report.endpoint_integrated_regional_work.contains(0)
    assert all(step.regional_boundary_work == I(0) for step in report.steps)
    assert "no_solver_replay_or_authentication" in " ".join(report.scope)


def test_flat_tubes_prove_acute_winding_absent_on_complete_horizon(controls):
    report = _assess(controls["flat"])
    assert report.outcome == "acute_winding_excluded_on_horizon"
    assert report.horizon_complete and not report.certified_windows
    assert report.initial_regional_storage == I(0)
    for step in report.steps:
        assert step.winding == 0
        assert step.acute_winding_excluded
        assert "fixed_branch_zero_winding" in step.exclusion_reasons
        assert "two_hop_pair_nonnegative_cosine" in step.exclusion_reasons


@pytest.mark.parametrize("kind", ("flat", "acute"))
def test_channel_history_preserves_integral_identity_and_initial_phase_scope(
    controls, kind
):
    report = assess_sine_regional_channels(controls[kind], cycle_indices=CYCLE)
    assert report.horizon_complete
    assert report.initial_phase_flat_certified == (kind == "flat")
    assert report.acute_entry_excluded_on_validated_prefix == (kind == "flat")
    assert report.acute_accessibility_barrier == Q(7, 2)
    # A prepared acute state is below the path barrier but was not acquired
    # from consensus; it must not be ruled out by this admission condition.
    assert report.initial_phase_storage.hi < Q(7, 2)
    assert report.cumulative_boundary_form_input == I(0)
    assert report.cumulative_boundary_phase_input == I(0)
    assert report.cumulative_internal_conversion.contains(
        report.endpoint_form_storage.lo - report.initial_form_storage.hi
    )
    for step in report.steps:
        assert step.form_integral_residual.contains(0)
        assert step.phase_integral_residual.contains(0)
        assert step.channels.signed_form_loss == I(0)
    assert "no_replay_or_authentication" in " ".join(report.scope)


def test_channel_history_retains_partial_prefix_and_rejects_invalid_evidence(controls):
    forecast = controls["flat"]
    first = forecast.steps[0]
    partial = replace(
        forecast,
        steps=(first,),
        validated_end_time=first.time + first.duration,
        endpoint=first.endpoint,
        status="unavailable",
        reasons=("stopped_control",),
    )
    report = assess_sine_regional_channels(partial, cycle_indices=CYCLE)
    assert not report.horizon_complete
    assert report.acute_entry_excluded_on_validated_prefix
    assert len(report.steps) == 1
    with pytest.raises(ValueError, match="endpoint|coverage|status"):
        assess_sine_regional_channels(
            replace(partial, status="admitted"), cycle_indices=CYCLE
        )
    with pytest.raises((TypeError, ValueError)):
        assess_sine_regional_channels(
            replace(forecast, visible_capacity=(True,) * 4), cycle_indices=CYCLE
        )


@pytest.mark.parametrize("changes", [{"acute_margin": 1}, {"minimum_duration": 1}])
def test_requested_policy_failure_does_not_claim_acute_absence(controls, changes):
    report = _assess(controls["acute"], **changes)
    assert report.outcome == "unresolved"
    assert report.horizon_complete
    assert not report.certified_windows
    assert not any(step.acute_winding_excluded for step in report.steps)
    assert all(
        step.winding == 1 and step.acute_margin_lower_bound > 0 for step in report.steps
    )


def test_wide_whole_time_box_is_not_replaced_by_its_acute_endpoint(controls):
    forecast = controls["acute"]
    first = forecast.steps[0]
    enlarged = tuple(
        I(value.lo - 4, value.hi + 4) if i < 10 else value
        for i, value in enumerate(first.tube)
    )
    supplied = replace(
        forecast, steps=(replace(first, tube=enlarged), forecast.steps[1])
    )
    report = _assess(supplied)
    assert report.outcome == "unresolved"
    assert report.steps[0].winding is None
    assert report.steps[0].edge_turn_offsets is None
    assert not report.steps[0].acute_winding_excluded
    assert report.steps[1].requested_acute_margin_met
    assert not report.certified_windows


@pytest.mark.parametrize("kind", ["flat", "acute"])
def test_partial_coverage_is_explicit_even_when_observed_steps_have_a_verdict(
    controls, kind
):
    forecast = controls[kind]
    first = forecast.steps[0]
    partial = replace(
        forecast,
        steps=(first,),
        validated_end_time=first.time + first.duration,
        endpoint=first.endpoint,
        status="unavailable",
        reasons=("unrelated_control_stopped_at_first_step",),
    )
    report = _assess(partial, minimum_duration=Q(1, 128))
    assert not report.horizon_complete
    assert "requested_horizon_not_fully_enclosed" in report.unresolved
    if kind == "flat":
        assert report.outcome == "unresolved"
    else:
        assert report.outcome == "finite_acute_retention_certified"
        assert report.certified_windows == ((Q(0), Q(1, 64)),)


def test_reversed_cycle_has_opposite_winding_and_same_observable_storage(controls):
    original = _assess(controls["acute"])
    reverse = _assess(controls["acute"], cycle_indices=(0, 4, 3, 2, 1))
    assert reverse.certified_windows == original.certified_windows
    assert all(step.winding == -1 for step in reverse.steps)
    assert reverse.initial_regional_storage == original.initial_regional_storage
    assert reverse.endpoint_regional_storage == original.endpoint_regional_storage


@pytest.fixture(scope="module")
def attached_control():
    neighbors = ((1, 4, 5), (0, 2), (1, 3), (2, 4), (0, 3), (0,))
    forms = (Q(0),) * 5 + (Q(1, 2),)
    phases = tuple(Q(5 * i, 4) for i in range(5)) + (Q(-1, 4),)
    forecast = bound_sine_flow(
        forms + phases + (Q(1),),
        neighbors=neighbors,
        visible_capacity=(Q(1),) * 5,
        model=MODEL,
        observation_time=Q(0),
        end_time=Q(1, 128),
        time_step=Q(1, 128),
        order=4,
    )
    assert forecast.admitted
    return forecast


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    assert interval.contains(Q(str(value)))


def test_full_support_phase_rates_and_regional_work_enclose_independent_states(
    attached_control,
):
    forecast = attached_control
    report = _assess(forecast, minimum_duration=Q(1, 128))
    step = report.steps[0]
    # A representative is used only for an independent containment check,
    # never substituted for the interval report or an actual trajectory state.
    with mp.workdps(90):
        box = forecast.steps[0].tube
        x = tuple(_mp(value.midpoint) for value in box[:6])
        theta = tuple(_mp(value.midpoint) for value in box[6:12])
        neighbors = forecast.neighbors
        fx = tuple(
            sum(mp.sin(theta[j] - theta[i]) for j in row) / (mp.pi * len(row))
            for i, row in enumerate(neighbors)
        )
        ft = tuple(
            sum(x[i] - x[j] for j in row) / (mp.pi * len(row))
            for i, row in enumerate(neighbors)
        )
        form = sum((x[j] - x[i]) ** 2 / 2 for i, j in zip(CYCLE, CYCLE[1:] + CYCLE[:1]))
        phase = sum(
            1 - mp.cos(theta[j] - theta[i])
            for i, j in zip(CYCLE, CYCLE[1:] + CYCLE[:1])
        )
        power = (x[5] - x[0]) * fx[0] + mp.sin(theta[5] - theta[0]) * ft[0]
        _contains(step.regional_form_storage, form)
        _contains(step.regional_phase_storage, phase)
        _contains(step.regional_storage, form + phase)
        _contains(step.regional_boundary_work, power)
        _contains(
            step.full_storage,
            form + phase + (x[5] - x[0]) ** 2 / 2 + 1 - mp.cos(theta[5] - theta[0]),
        )
        for interval, velocity in zip(step.relative_phase_rate_bounds, ft):
            _contains(interval, velocity - ft[0])
        for interval, velocity in zip(step.phase_rate_bounds, ft):
            _contains(interval, velocity)
    assert step.relative_phase_rate_bounds[0] == I(0)
    assert len(step.relative_phase_rate_bounds) == 6
    assert len(step.phase_rate_bounds) == 6
    assert (
        report.endpoint_integrated_regional_work
        == report.endpoint_regional_storage - report.initial_regional_storage
    )
    assert (
        step.integrated_regional_work
        == step.regional_storage - report.initial_regional_storage
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_duration": True},
        {"minimum_duration": 0},
        {"minimum_duration": -1},
        {"acute_margin": False},
        {"acute_margin": -1},
        {"acute_margin": float("nan")},
    ],
)
def test_invalid_observation_policy_is_not_coerced(controls, changes):
    with pytest.raises((TypeError, ValueError)):
        _assess(controls["flat"], **changes)


@pytest.mark.parametrize(
    "cycle",
    [
        (0, 1, 2, 3),
        (0, 1, 2, 3, True),
        (0, 1, 2, 3, 3),
        (0, 1, 3, 2, 4),
        (0, 1, 2, 3, 8),
    ],
)
def test_receiver_requires_the_actual_induced_five_cycle(controls, cycle):
    with pytest.raises((TypeError, ValueError)):
        _assess(controls["flat"], cycle_indices=cycle)


def test_projection_is_detached_and_frozen(controls):
    report = _assess(controls["acute"])
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.sine-regional-organization.v1"
    assert payload["report"]["outcome"] == report.outcome
    payload["report"]["cycle_indices"].append(8)
    assert report.cycle_indices == CYCLE
    with pytest.raises(FrozenInstanceError):
        report.outcome = "different"
