"""Independent finite-amplitude error bounds, without numerical trajectories."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tests.physics.test_sine_consensus_observation import _apply
from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_consensus_observation import (
    assess_sine_consensus_observation,
)
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

FORM_BOUNDS = (Q(1, 64), Q(1, 128), Q(1, 256))
PHASE_BOUNDS = (Q(1, 128), Q(1, 256), Q(1, 128))


def _source(
    *,
    weights=(2, 3),
    capacities=(2, 3, 5),
    beta=2,
    forms=(Q(1, 128), Q(-1, 256), Q(1, 512)),
    phases=(Q(-1, 128), Q(1, 512), Q(1, 256)),
):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, (x, phase, capacity) in enumerate(zip(forms, phases, capacities)):
        graph.add_node(node, EPI=float(x), theta=float(phase), nu_f=float(capacity))
    for edge, weight in zip(((0, 1), (1, 2)), weights):
        graph.add_edge(*edge, weight=float(weight))
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )


def _observation(source):
    return assess_sine_consensus_observation(
        source, regions=((0,), (1, 2)), max_rank_calls=128
    )


def _bound(
    observation,
    *,
    duration=Q(1, 64),
    form_origin=0,
    phase_origin=0,
    forms=FORM_BOUNDS,
    phases=PHASE_BOUNDS,
):
    return observation.nonlinear_response_bounds(
        duration=duration,
        consensus_form=form_origin,
        consensus_phase=phase_origin,
        form_amplitude_bounds=forms,
        phase_amplitude_bounds=phases,
    )


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("nonlinear validity uses bounds, not numerical trajectories")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def observation():
    return _observation(_source())


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.mark.parametrize("exchange", (Q(1), Q(3, 2)))
def test_independent_channel_bootstrap_cubic_residual_and_propagation(
    observation, mp, exchange
):
    model = copy(observation.source.reference_model)
    object.__setattr__(model, "phase_weight", exchange)
    current = _observation(replace(observation.source, reference_model=model))
    report = _bound(current)
    h = Q(1, 64)
    for actuals, coefficients in (
        (report.form_phase_lipschitz_upper_bounds, (4, 6, 10)),
        (report.phase_form_lipschitz_upper_bounds, (4, Q(15, 2), 15)),
    ):
        for actual, coefficient in zip(actuals, coefficients):
            expected = _mp(mp, exchange * coefficient) / mp.pi
            assert 0 <= _mp(mp, actual) - expected < mp.mpf("1e-30")
    a = max(report.form_phase_lipschitz_upper_bounds)
    b = max(report.phase_form_lipschitz_upper_bounds)
    denominator = 1 - a * b * h**2
    x = (max(FORM_BOUNDS) + a * h * max(PHASE_BOUNDS)) / denominator
    theta = (max(PHASE_BOUNDS) + b * h * max(FORM_BOUNDS)) / denominator
    assert report.comparison_denominator == denominator > 0
    assert report.form_amplitude_supremum_upper_bound == x
    assert report.phase_amplitude_supremum_upper_bound == theta
    assert report.whole_window_form_amplitude_bounds == tuple(
        value + coefficient * h * theta
        for value, coefficient in zip(
            FORM_BOUNDS, report.form_phase_lipschitz_upper_bounds
        )
    )
    assert report.whole_window_phase_amplitude_bounds == tuple(
        value + coefficient * h * x
        for value, coefficient in zip(
            PHASE_BOUNDS, report.phase_form_lipschitz_upper_bounds
        )
    )
    phase_radii = report.whole_window_phase_amplitude_bounds
    gaps = (phase_radii[0] + phase_radii[1], phase_radii[1] + phase_radii[2])
    assert report.edge_phase_difference_upper_bounds == gaps
    sums = (gaps[0] ** 3, sum(gap**3 for gap in gaps), gaps[1] ** 3)
    for i, (capacity, degree, total) in enumerate(zip((2, 3, 5), (1, 2, 1), sums)):
        exact = _mp(mp, exchange * capacity * total / (6 * degree)) / mp.pi
        assert (
            0
            <= _mp(mp, report.form_cubic_remainder_upper_bounds[i]) - exact
            < mp.mpf("1e-30")
        )
    remainder = max(report.form_cubic_remainder_upper_bounds)
    error_x = h * remainder / denominator
    error_theta = b * h**2 * remainder / denominator
    assert report.max_form_cubic_remainder_upper_bound == remainder
    assert report.form_error_supremum_upper_bound == error_x
    assert report.phase_error_supremum_upper_bound == error_theta
    assert report.whole_window_form_error_bounds == tuple(
        h * residual + coefficient * h * error_theta
        for residual, coefficient in zip(
            report.form_cubic_remainder_upper_bounds,
            report.form_phase_lipschitz_upper_bounds,
        )
    )
    assert report.whole_window_phase_error_bounds == tuple(
        coefficient * h * error_x
        for coefficient in report.phase_form_lipschitz_upper_bounds
    )
    assert report.initial_form_displacements == current.source.epi
    assert report.initial_phase_displacements == current.source.phase
    initial = current.source.epi + current.source.phase
    assert report.initial_observation_state == _apply(
        current.linear_observation.observation, initial
    )
    assert report.initial_outputs == _apply(current.output_rows, initial)
    errors = (
        report.whole_window_form_error_bounds + report.whole_window_phase_error_bounds
    )
    for rows, projected in (
        (
            current.linear_observation.observation,
            report.selected_state_error_upper_bounds,
        ),
        (current.output_rows, report.output_error_upper_bounds),
    ):
        assert projected == tuple(
            sum(abs(value) * bound for value, bound in zip(row, errors)) for row in rows
        )
    assert (
        report.status == "enclosed"
        and report.observation.source.reference_model.phase_weight == exchange
    )
    assert observation.source.reference_model.phase_weight == 1
    with pytest.raises(FrozenInstanceError):
        report.duration = Q(2)


def test_doubled_amplitude_has_cubic_error_scaling_not_linear_projection_error(
    observation, mp
):
    base = _bound(observation)
    doubled = _bound(
        observation,
        forms=tuple(2 * value for value in FORM_BOUNDS),
        phases=tuple(2 * value for value in PHASE_BOUNDS),
    )
    assert (
        doubled.form_amplitude_supremum_upper_bound
        == 2 * base.form_amplitude_supremum_upper_bound
    )
    assert (
        doubled.phase_amplitude_supremum_upper_bound
        == 2 * base.phase_amplitude_supremum_upper_bound
    )
    for field in (
        "max_form_cubic_remainder_upper_bound",
        "form_error_supremum_upper_bound",
        "phase_error_supremum_upper_bound",
    ):
        assert abs(
            _mp(mp, getattr(doubled, field) - 8 * getattr(base, field))
        ) < mp.mpf("1e-30")
    assert doubled.initial_observation_state == base.initial_observation_state


@pytest.mark.parametrize("duration", (Q(1, 64), Q(7)))
def test_two_node_zero_weight_exact_history_is_enclosed_without_integration(
    duration, mp
):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    graph.add_node(0, EPI=Q(1, 32), theta=-Q(1, 16), nu_f=2)
    graph.add_node(1, EPI=-Q(1, 64), theta=Q(1, 32), nu_f=5)
    graph.add_edge(0, 1, weight=0)
    source = bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            3, epi_weight=0, phase_domain="regular"
        ),
    )
    model = copy(source.reference_model)
    object.__setattr__(model, "phase_weight", Q(3, 2))
    current = assess_sine_consensus_observation(
        replace(source, reference_model=model), regions=((0,), (1,))
    )
    report = _bound(
        current,
        duration=duration,
        forms=(Q(1, 32), Q(1, 64)),
        phases=(Q(1, 16), Q(1, 32)),
    )
    assert report.comparison_denominator == 1
    assert report.whole_window_phase_amplitude_bounds == (Q(1, 16), Q(1, 32))
    assert report.phase_error_supremum_upper_bound == 0
    assert report.whole_window_phase_error_bounds == (0, 0)
    delta = _mp(mp, Q(3, 32))
    error = _mp(mp, Q(3, 2) * duration) * (mp.sin(delta) - delta) / mp.pi
    exact_errors = (2 * error, -5 * error)
    for exact, bound in zip(exact_errors, report.whole_window_form_error_bounds):
        assert 0 < abs(exact) <= _mp(mp, bound)
    assert abs(exact_errors[0] - exact_errors[1]) <= _mp(
        mp, report.output_error_upper_bounds[0]
    )
    assert report.output_error_upper_bounds[1] == 0
    assert report.status == "enclosed"


def test_hidden_initialization_is_retained_beyond_zero_requested_outputs():
    a = Q(1, 1024)
    current = _observation(_source(forms=(0, 0, 0), phases=(0, 3 * a, -10 * a)))
    report = _bound(current, forms=(Q(1, 64),) * 3, phases=(Q(1, 64),) * 3)
    assert report.initial_outputs == (0, 0)
    assert any(report.initial_observation_state)
    assert report.initial_observation_state == _apply(
        current.linear_observation.observation, (Q(0),) * 3 + (0, 3 * a, -10 * a)
    )


def test_zero_box_is_exact_for_any_duration_at_the_admitted_consensus_source():
    current = _observation(_source(forms=(Q(7, 16),) * 3, phases=(Q(-3, 8),) * 3))
    report = _bound(
        current,
        duration=100,
        form_origin=Q(7, 16),
        phase_origin=Q(-3, 8),
        forms=(0, 0, 0),
        phases=(0, 0, 0),
    )
    assert report.comparison_denominator < 0
    assert report.status == "enclosed"
    assert (
        report.form_error_supremum_upper_bound
        == report.phase_error_supremum_upper_bound
        == 0
    )
    assert (
        report.selected_state_error_upper_bounds
        == (0,) * current.linear_observation.dimension
    )
    assert report.initial_outputs == report.output_error_upper_bounds == (0, 0)


def test_insufficient_window_has_no_invented_error_enclosure(observation):
    report = _bound(observation, duration=100)
    assert report.comparison_denominator < 0
    assert report.status == "unavailable" and report.unavailable_reasons
    for field in (
        "form_amplitude_supremum_upper_bound",
        "phase_amplitude_supremum_upper_bound",
        "whole_window_form_amplitude_bounds",
        "whole_window_phase_amplitude_bounds",
        "edge_phase_difference_upper_bounds",
        "form_cubic_remainder_upper_bounds",
        "max_form_cubic_remainder_upper_bound",
        "form_error_supremum_upper_bound",
        "phase_error_supremum_upper_bound",
        "whole_window_form_error_bounds",
        "whole_window_phase_error_bounds",
        "selected_state_error_upper_bounds",
        "output_error_upper_bounds",
    ):
        assert getattr(report, field) is None
    assert report.initial_form_displacements == observation.source.epi
    assert report.initial_phase_displacements == observation.source.phase


def test_declared_origins_translate_full_source_without_wrapping_or_reset(observation):
    moved = _observation(
        replace(
            observation.source,
            epi=tuple(x + Q(3, 8) for x in observation.source.epi),
            phase=tuple(theta - Q(5, 4) for theta in observation.source.phase),
        )
    )
    report = _bound(moved, form_origin=Q(3, 8), phase_origin=Q(-5, 4))
    reference = _bound(observation)
    for field in (
        "initial_form_displacements",
        "initial_phase_displacements",
        "initial_observation_state",
        "form_error_supremum_upper_bound",
        "phase_error_supremum_upper_bound",
        "output_error_upper_bounds",
    ):
        assert getattr(report, field) == getattr(reference, field)
    assert report.observation.source.phase == moved.source.phase


@pytest.mark.parametrize(
    "mutation", ("form_outside", "phase_outside", "represented_turn")
)
def test_source_membership_is_exact_and_never_silently_projected(observation, mutation):
    source = observation.source
    if mutation == "form_outside":
        source = replace(source, epi=(FORM_BOUNDS[0] + Q(1, 2**100),) + source.epi[1:])
    elif mutation == "phase_outside":
        source = replace(
            source, phase=(PHASE_BOUNDS[0] + Q(1, 2**100),) + source.phase[1:]
        )
    else:
        source = replace(
            source,
            phase=(source.phase[0] + Q.from_float(6.283185307179586),)
            + source.phase[1:],
        )
    with pytest.raises(ValueError):
        _bound(_observation(source))


@pytest.mark.parametrize(
    "field,value",
    (
        ("duration", True),
        ("duration", 0),
        ("duration", float("nan")),
        ("form_origin", False),
        ("phase_origin", float("inf")),
        ("forms", (True, 0, 0)),
        ("forms", (-1, 0, 0)),
        ("forms", (0, 0)),
        ("phases", (0, 0, 0, 0)),
        ("phases", (0, float("inf"), 0)),
    ),
)
def test_invalid_primitive_budgets_reject_before_any_certificate(
    observation, field, value
):
    with pytest.raises((TypeError, ValueError)):
        _bound(observation, **{field: value})


def test_fresh_sdk_export_rebuilds_observation_initialization_and_all_error_claims(
    observation, tmp_path
):
    report = _bound(observation)
    poisoned_observation = replace(
        observation,
        rational_generator=((Q(0),) * 6,) * 6,
        output_rows=((Q(0),) * 6,) * 2,
        linear_observation=replace(
            observation.linear_observation, observation=((Q(0),) * 6,) * 4
        ),
    )
    poisoned = replace(
        report,
        observation=poisoned_observation,
        initial_observation_state=(Q(0),) * 4,
        initial_outputs=(Q(0), Q(0)),
        max_form_cubic_remainder_upper_bound=Q(0),
        whole_window_form_error_bounds=(Q(0),) * 3,
        output_error_upper_bounds=(Q(0), Q(0)),
        status="unavailable",
    )
    expected = relational_report_to_dict(report)
    assert relational_report_to_dict(poisoned) == expected
    path = tmp_path / "nonlinear-response.json"
    export_to_json(relational_report_to_dict(poisoned), path)
    assert json_loads(path.read_text(encoding="utf-8")) == expected
    before = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(
                replace(poisoned, phase_amplitude_bounds=(False, 0, 0))
            ),
            path,
        )
    assert path.read_bytes() == before
    unavailable = relational_report_to_dict(_bound(observation, duration=100))
    assert unavailable["report"]["output_error_upper_bounds"] is None


def test_invalid_source_cannot_use_a_forged_valid_observation(observation):
    poisoned = replace(
        observation, source=replace(observation.source, capacity=(True, 3, 5))
    )
    with pytest.raises((TypeError, ValueError)):
        _bound(poisoned)


def test_response_export_never_dispatches_a_fake_observation_method(observation):
    class Fake:
        def nonlinear_response_bounds(self, **kwargs):
            raise AssertionError("unadmitted dispatch")

    with pytest.raises(TypeError):
        replace(_bound(observation), observation=Fake()).to_dict()
