"""Adversarial admission of synthetic regional-observer evidence.

These manually assembled records exercise consumer structure and Picard
admission only. They are not a research trajectory or evidence that the H=3
preparation reaches an organized state. Taylor-production provenance remains
with the separately frozen producer and source archive.
"""

from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import ValidatedTaylorStep
from tnfr.physics.relational_sine_forecast import SineForecast
from tnfr.physics.relational_sine_regional import assess_sine_regional_organization

CYCLE = tuple(range(5, 10))
STEP = Q(1, 64)


@pytest.fixture(scope="module")
def synthetic_forecast():
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(
        [(o + i, o + (i + 1) % 5) for o in (0, 5) for i in range(5)]
        + [(0, 10), (10, 5), (1, 6)]
    )
    initial = (I(0),) * 22 + (I(1),)
    tube = tuple(I(value.lo - Q(1, 100), value.hi + Q(1, 100)) for value in initial)
    first = ValidatedTaylorStep(
        time=Q(0),
        duration=STEP,
        tube=tube,
        endpoint=initial,
        picard_interior_margin=Q(1, 1000),
        domain_lower_bounds=(Q(1),),
        propagated_initial_radii=(Q(0),) * 23,
        local_remainder_bounds=(I(0),) * 23,
    )
    return SineForecast(
        model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
        neighbors=tuple(tuple(sorted(graph[node])) for node in graph),
        visible_capacity=(Q(1),) * 10,
        initial_box=initial,
        observation_time=Q(0),
        end_time=2 * STEP,
        time_step=STEP,
        order=4,
        steps=(first, replace(first, time=STEP)),
        validated_end_time=2 * STEP,
        endpoint=initial,
        failed_tube=None,
        status="admitted",
        reasons=(),
    )


def _assess(forecast):
    return assess_sine_regional_organization(
        forecast, cycle_indices=CYCLE, minimum_duration=STEP, acute_margin=Q(1, 16)
    )


def test_structured_synthetic_records_exercise_observer_not_research_solver(
    synthetic_forecast,
):
    report = _assess(synthetic_forecast)
    assert report.horizon_complete
    assert report.outcome == "acute_winding_excluded_on_horizon"
    assert not report.certified_windows
    assert len(report.steps) == len(synthetic_forecast.steps)


def test_cached_positive_picard_margin_does_not_replace_actual_inclusion(
    synthetic_forecast,
):
    first = synthetic_forecast.steps[0]
    # Both declared endpoints lie in this collapsed coordinate, but no
    # positive Picard interior exists there. The cached positive number lies.
    tube = (I(0),) + first.tube[1:]
    forged = replace(first, tube=tube, picard_interior_margin=Q(1))
    with pytest.raises((TypeError, ValueError)):
        _assess(
            replace(synthetic_forecast, steps=(forged,) + synthetic_forecast.steps[1:])
        )


def test_changed_initial_state_inside_tube_still_requires_picard_flow_inclusion(
    synthetic_forecast,
):
    initial = list(synthetic_forecast.initial_box)
    initial[10] = I(Q(999, 100000))
    assert synthetic_forecast.steps[0].tube[10].contains(initial[10])
    # The changed environmental state leaves too little room for the entire
    # interval field image, despite pointwise containment in the old tube.
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(synthetic_forecast, initial_box=tuple(initial)))


@pytest.mark.parametrize(
    "field,value",
    [("phase_weight", True), ("epi_weight", False), ("storage_scale", float("nan"))],
)
def test_forged_model_scalars_are_readmitted_before_coefficient_equality(
    synthetic_forecast, field, value
):
    model = replace(synthetic_forecast.model)
    object.__setattr__(model, field, value)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(synthetic_forecast, model=model))


def test_boolean_capacity_cannot_supply_physical_rates(synthetic_forecast):
    with pytest.raises((TypeError, ValueError)):
        _assess(
            replace(
                synthetic_forecast,
                visible_capacity=(True,) + synthetic_forecast.visible_capacity[1:],
            )
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"observation_time": True},
        {"observation_time": STEP},
        {"time_step": 2 * STEP},
        {"end_time": 3 * STEP},
        {"order": True},
        {"freeze_hidden": True},
        {"forecast_start": -STEP},
    ],
)
def test_changed_clock_grid_or_execution_premises_reject_the_old_chain(
    synthetic_forecast, changes
):
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(synthetic_forecast, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"time": STEP},
        {"duration": 2 * STEP},
        {"picard_interior_margin": True},
        {"domain_lower_bounds": (True,)},
        {"propagated_initial_radii": (True,) + (Q(0),) * 22},
        {"propagated_initial_radii": (Q(-1),) + (Q(0),) * 22},
        {"local_remainder_bounds": (I(0),) * 22},
        {"local_remainder_bounds": (True,) + (I(0),) * 22},
    ],
)
def test_malformed_step_evidence_is_not_accepted_as_a_solver_certificate(
    synthetic_forecast, changes
):
    first = replace(synthetic_forecast.steps[0], **changes)
    with pytest.raises((TypeError, ValueError)):
        _assess(
            replace(synthetic_forecast, steps=(first,) + synthetic_forecast.steps[1:])
        )


def test_held_capacity_change_is_rejected_even_inside_the_reported_tubes(
    synthetic_forecast,
):
    endpoint = synthetic_forecast.endpoint[:-1] + (I(Q(1001, 1000)),)
    assert synthetic_forecast.steps[0].tube[-1].contains(endpoint[-1])
    steps = tuple(replace(step, endpoint=endpoint) for step in synthetic_forecast.steps)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(synthetic_forecast, steps=steps, endpoint=endpoint))


def test_forecast_endpoint_cannot_disagree_with_the_last_step(synthetic_forecast):
    endpoint = (I(Q(1, 1000)),) + synthetic_forecast.endpoint[1:]
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(synthetic_forecast, endpoint=endpoint))


def test_incomplete_excluded_prefix_is_unresolved_for_the_declared_horizon(
    synthetic_forecast,
):
    prefix = replace(
        synthetic_forecast,
        steps=synthetic_forecast.steps[:1],
        validated_end_time=STEP,
        status="unavailable",
        reasons=("synthetic_budget_stop",),
        failed_tube=synthetic_forecast.steps[-1].tube,
    )
    report = _assess(prefix)
    assert not report.horizon_complete
    assert report.outcome == "unresolved"
    assert not report.certified_windows
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(prefix, status="admitted", reasons=(), failed_tube=None))


def test_missing_validated_steps_do_not_become_a_zero_state_verdict(synthetic_forecast):
    absent = replace(
        synthetic_forecast,
        steps=(),
        validated_end_time=Q(0),
        status="unavailable",
        reasons=("synthetic_initial_step_unresolved",),
        failed_tube=synthetic_forecast.steps[0].tube,
    )
    report = _assess(absent)
    assert not report.horizon_complete
    assert not report.steps
    assert report.outcome == "unresolved"


def test_wide_complete_enclosures_are_unresolved_not_proof_of_no_organization(
    synthetic_forecast,
):
    wide = (I(-10, 10),) * 22 + (synthetic_forecast.steps[0].tube[-1],)
    forecast = replace(
        synthetic_forecast,
        steps=tuple(replace(step, tube=wide) for step in synthetic_forecast.steps),
    )
    report = _assess(forecast)
    assert report.horizon_complete
    assert report.outcome == "unresolved"
    assert not report.certified_windows
