"""Manufactured full-state controls for the retained saddle-metric adapter.

Only an analytic uniform equilibrium is advanced over a declared tiny test
horizon. These tests do not evaluate the near-saddle preparation, extend a
frozen response or run a formation experiment.
"""

import json
from copy import copy
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.relational_sine_comparison import _comparison_neighbors
from tnfr.physics.relational_sine_forecast import _sine_flow
from tnfr.physics.relational_sine_metric_forecast import (
    _saddle_flow,
    forecast_sine_saddle_metric,
)
from tnfr.sdk import export_to_json, relational_report_to_dict


@pytest.fixture(scope="module")
def equilibrium():
    return _state(epi=(Q(1, 4),) * 10, phase=(Q(0),) * 10)


def _forecast(source, **changes):
    declaration = dict(
        cycle=range(5),
        duration=Q(1, 5000),
        time_step=Q(1, 10000),
        initial_coordinate_radius=Q(1, 2**30),
        phase_radius=Q(3),
        order=3,
        max_steps=2,
    )
    declaration.update(changes)
    return forecast_sine_saddle_metric(source, **declaration)


@pytest.fixture(scope="module")
def forecast(equilibrium):
    return _forecast(equilibrium)


def test_complete_equilibrium_and_uncertain_common_origins_are_retained(forecast):
    assert forecast.status == "admitted"
    assert forecast.reasons == ()
    assert forecast.validated_duration == forecast.duration == Q(1, 5000)
    assert len(forecast.steps) == 2
    assert (
        len(forecast.initial_center)
        == len(forecast.endpoint_center)
        == len(forecast.endpoint)
        == 20
    )
    assert forecast.initial_center == (Q(1, 4),) * 10 + (Q(0),) * 10
    assert forecast.initial_metric_radius >= forecast.initial_coordinate_radius
    assert forecast.endpoint_radius > 0
    # Four exact constant trajectories from independent origin corners belong
    # to the supplied full-coordinate cube. Both common modes must survive.
    for form_sign in (-1, 1):
        for phase_sign in (-1, 1):
            point = (Q(1, 4) + form_sign * forecast.initial_coordinate_radius,) * 10 + (
                phase_sign * forecast.initial_coordinate_radius,
            ) * 10
            assert all(
                bound.lo <= value <= bound.hi
                for bound, value in zip(forecast.endpoint, point)
            )
    assert forecast.sensitivity.nonlinear_tube_implication_certified
    # The separate theorem remains conditional; actual whole-time evidence
    # belongs to the forecast rather than a mutated sensitivity verdict.
    assert not forecast.sensitivity.captured_source_flow_bound_certified


def test_original_clock_growth_and_retained_radius_chain_do_not_rebox(forecast):
    assert (
        forecast.growth_rate_upper_bound
        == forecast.sensitivity.nonlinear_growth_rate_upper_bound / pi_interval().lo
    )
    assert (
        forecast.initial_metric_radius
        == forecast.sensitivity.infinity_to_metric_upper_bound
        * forecast.initial_coordinate_radius
    )
    previous_radius = forecast.initial_metric_radius
    previous_center = forecast.initial_center
    for step in forecast.steps:
        assert step.initial_radius == previous_radius
        assert step.initial_center == previous_center
        assert step.growth_rate_upper_bound == forecast.growth_rate_upper_bound
        assert step.picard_interior_margin > 0
        assert len(step.domain_lower_bounds) == 10
        assert min(step.domain_lower_bounds) > 0
        assert len(step.tube) == len(step.endpoint) == 20
        assert (
            step.endpoint_radius
            >= step.propagated_initial_radius + step.local_metric_error_upper_bound
        )
        reboxed_radius = forecast.sensitivity.infinity_to_metric_upper_bound * max(
            bound.radius for bound in step.endpoint
        )
        assert step.endpoint_radius < reboxed_radius
        previous_radius, previous_center = step.endpoint_radius, step.endpoint_center
    assert forecast.endpoint_radius == previous_radius
    assert forecast.endpoint_center == previous_center


def test_shared_flow_preserves_nonzero_full_rows_and_phase_jet_in_original_clock():
    source = _state(
        epi=tuple(Q(i - 4, 7) for i in range(10)),
        phase=tuple(Q(i - 4, 13) for i in range(10)),
    )
    neighbors = _comparison_neighbors(source)
    values = tuple(I(value) for value in source.epi + source.phase)
    original = _sine_flow(
        values + (I(1),),
        neighbors=neighbors,
        visible_capacity=(Q(1),) * 9,
        model=source.reference_model,
    )
    forward = _saddle_flow(values, source=source, neighbors=neighbors, direction=1)
    reverse = _saddle_flow(values, source=source, neighbors=neighbors, direction=-1)
    assert len(original) == 21 and original[-1] == I(0)
    assert len(forward) == 20
    assert forward == original[:-1]
    assert reverse == tuple(-value for value in forward)
    variables = tuple(Jet((value, I(int(i == 17)))) for i, value in enumerate(values))
    jets = _saddle_flow(variables, source=source, neighbors=neighbors, direction=1)
    assert len(jets) == 20 and all(value.order == 1 for value in jets)
    with mp.workdps(70):
        number = lambda value: mp.mpf(value.numerator) / value.denominator
        phase = tuple(number(value) for value in source.phase)
        form = tuple(number(value) for value in source.epi)
        for i, row in enumerate(neighbors):
            expected_form = sum(mp.sin(phase[j] - phase[i]) for j in row) / (
                len(row) * mp.pi
            )
            expected_phase = sum(form[i] - form[j] for j in row) / (len(row) * mp.pi)
            expected_jet = sum(
                mp.cos(phase[j] - phase[i]) * (int(j == 7) - int(i == 7)) for j in row
            ) / (len(row) * mp.pi)
            for expected, bound in (
                (expected_form, forward[i]),
                (expected_phase, forward[10 + i]),
                (expected_jet, jets[i].coeffs[1]),
            ):
                assert number(bound.lo) <= expected <= number(bound.hi)
            assert jets[10 + i].coeffs[1] == I(0)


def test_unavailable_later_step_keeps_the_last_certified_ball(
    equilibrium, forecast, monkeypatch
):
    import tnfr.physics.relational_sine_metric_forecast as owner

    observed = []

    def controlled_step(center, radius, *args, time, **kwargs):
        observed.append((center, radius, time))
        if time == 0:
            return forecast.steps[0], None, None
        return (
            None,
            tuple(I(value) for value in center),
            "controlled_second_step_domain_failure",
        )

    monkeypatch.setattr(owner, "validated_metric_taylor_step", controlled_step)
    result = _forecast(equilibrium)
    assert result.status == "unavailable"
    assert result.validated_duration == forecast.time_step
    assert result.endpoint_center == forecast.steps[0].endpoint_center
    assert result.endpoint_radius == forecast.steps[0].endpoint_radius
    assert result.endpoint == forecast.steps[0].endpoint
    assert result.reasons == ("controlled_second_step_domain_failure",)
    assert observed[1] == (
        forecast.steps[0].endpoint_center,
        forecast.steps[0].endpoint_radius,
        forecast.time_step,
    )


def test_wrong_phase_tube_is_unavailable_not_silently_recentered(equilibrium):
    result = _forecast(equilibrium, phase_radius=Q(1, 1000))
    assert result.status == "unavailable"
    assert result.validated_duration == 0
    assert not result.steps
    assert result.reasons
    assert result.initial_center == equilibrium.epi + equilibrium.phase


def test_actual_jacobian_mode_covers_the_constant_orbit_outside_saddle_tube(
    equilibrium,
):
    result = _forecast(
        equilibrium,
        phase_radius=None,
        growth_rate_bounds=(Q(0), Q(1)),
        growth_bisections=4,
    )
    assert result.admitted
    assert result.growth_mode == "whole_tube_jacobian"
    assert result.growth_rate_bounds == (Q(0), Q(1))
    assert result.validated_duration == result.duration
    assert result.growth_rate_upper_bound == max(
        step.growth_rate_upper_bound for step in result.steps
    )
    for step in result.steps:
        proof = step.growth_certificate
        assert proof is not None and proof.certified
        assert proof.tube == step.tube
        assert proof.growth_rate_upper_bound == step.growth_rate_upper_bound
        assert proof.bisections_requested == 4
        assert all(
            bound.contains(value)
            for bound, value in zip(step.endpoint, equilibrium.epi + equilibrium.phase)
        )


def test_insufficient_declared_growth_bracket_is_not_expanded(equilibrium):
    result = _forecast(equilibrium, phase_radius=None, growth_rate_bounds=(Q(0), Q(0)))
    assert not result.admitted
    assert result.validated_duration == 0
    assert result.growth_rate_upper_bound is None
    assert not result.steps
    assert "declared_growth_upper_endpoint_not_certified" in result.reasons


@pytest.mark.parametrize(
    "changes",
    (
        {"growth_rate_bounds": (0, 1)},
        {"phase_radius": None, "growth_rate_bounds": (True, 1)},
        {"phase_radius": None, "growth_rate_bounds": (1, 0)},
        {"phase_radius": None, "growth_bisections": True},
        {"phase_radius": None, "growth_bisections": 17},
    ),
)
def test_malformed_or_conflicting_growth_policies_reject(equilibrium, changes):
    with pytest.raises((TypeError, ValueError)):
        _forecast(equilibrium, **changes)


def test_zero_radius_and_reverse_direction_preserve_the_analytic_equilibrium(
    equilibrium,
):
    result = _forecast(
        equilibrium,
        direction=-1,
        duration=Q(1, 10000),
        initial_coordinate_radius=Q(0),
    )
    assert result.status == "admitted"
    assert result.direction == -1
    assert result.initial_metric_radius == 0
    assert len(result.steps) == 1
    for value, bound in zip(equilibrium.epi + equilibrium.phase, result.endpoint):
        assert bound.lo <= value <= bound.hi


def test_forecast_rebuilds_consumed_rates_storage_and_sensitivity(
    equilibrium, forecast
):
    poisoned = replace(
        equilibrium,
        form_rates=(),
        phase_rates=(),
        pressure=(I(999),) * 10,
        form_gradient=(),
        storage=I(-999),
    )
    result = _forecast(poisoned)
    assert result.status == "admitted"
    assert result.initial_center == forecast.initial_center
    assert result.initial_metric_radius == forecast.initial_metric_radius
    assert result.sensitivity.full_metric == forecast.sensitivity.full_metric
    assert result.endpoint_center == forecast.endpoint_center
    assert result.endpoint_radius == forecast.endpoint_radius
    assert result.endpoint == forecast.endpoint
    # No cached field is installed back into the detached input.
    assert poisoned.storage == I(-999)
    assert poisoned.form_rates == ()


def test_node_order_and_label_permutation_preserve_all_physical_coordinates(forecast):
    order = (7, 2, 9, 1, 5, 0, 4, 3, 8, 6)
    source = _state(
        epi=(Q(1, 4),) * 10,
        phase=(Q(0),) * 10,
        order=order,
        label=lambda i: ("node", i),
    )
    result = _forecast(source, cycle=tuple(("node", i) for i in range(5)))
    assert result.status == "admitted"
    assert result.validated_duration == forecast.validated_duration
    positions = {node: i for i, node in enumerate(result.source.nodes)}
    for i in range(10):
        for offset, expected in ((0, Q(1, 4)), (10, Q(0))):
            bounds = result.endpoint[offset + positions[("node", i)]]
            assert bounds.lo <= expected <= bounds.hi
    assert result.sensitivity.saddle.target_phase_turns[positions[("node", 0)]] == Q(
        -1, 3
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"duration": True},
        {"duration": 0},
        {"duration": -1},
        {"duration": float("inf")},
        {"time_step": False},
        {"time_step": 0},
        {"time_step": -1},
        {"initial_coordinate_radius": True},
        {"initial_coordinate_radius": -1},
        {"phase_radius": True},
        {"phase_radius": -1},
        {"direction": True},
        {"direction": Q(1)},
        {"direction": 0},
        {"order": True},
        {"order": 0},
        {"order": 17},
        {"max_steps": True},
        {"max_steps": 0},
        {"max_steps": 257},
    ),
)
def test_malformed_declarations_reject_before_evaluation(equilibrium, changes):
    with pytest.raises((TypeError, ValueError)):
        _forecast(equilibrium, **changes)


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_boolean_primitive_cannot_hide_behind_valid_cached_bounds(equilibrium, field):
    malformed = replace(
        equilibrium, **{field: (True,) + getattr(equilibrium, field)[1:]}
    )
    with pytest.raises((TypeError, ValueError)):
        _forecast(malformed)


def test_law_capacity_support_and_cycle_are_admitted_as_a_complete_model(equilibrium):
    model = copy(equilibrium.reference_model)
    object.__setattr__(model, "epi_weight", True)
    with pytest.raises((TypeError, ValueError)):
        _forecast(replace(equilibrium, reference_model=model))
    with pytest.raises((TypeError, ValueError)):
        _forecast(replace(equilibrium, capacity=(Q(0),) + equilibrium.capacity[1:]))
    with pytest.raises((TypeError, ValueError)):
        _forecast(replace(equilibrium, edges=equilibrium.edges + ((0, 2),)))
    with pytest.raises((TypeError, ValueError)):
        _forecast(equilibrium, cycle=(0, 1, 2, 3, 3))


def test_sdk_export_keeps_full_state_and_read_only_scope(forecast, tmp_path):
    payload = relational_report_to_dict(forecast)
    assert payload["report_type"] == "SineSaddleMetricForecast"
    assert len(payload["report"]["endpoint"]) == 20
    assert len(payload["report"]["source"]["capacity"]) == 10
    assert payload["report"]["direction"] == 1
    destination = tmp_path / "manufactured-metric-forecast.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload

    @dataclass(frozen=True)
    class Opaque:
        value: int

    nested = replace(
        forecast.sensitivity.saddle,
        cycle=(Opaque(0),) + forecast.sensitivity.saddle.cycle[1:],
    )
    with pytest.raises((TypeError, ValueError)):
        relational_report_to_dict(
            replace(forecast, sensitivity=replace(forecast.sensitivity, saddle=nested))
        )
