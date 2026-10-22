"""Synthetic retained-cell controls; no actual midpoint or response is read."""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as Q
from itertools import combinations
from types import SimpleNamespace

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import validated_box_taylor_step
from tnfr.physics._sine_flow import _held_sine_neighbor_field
from tnfr.research import maintained_child_sample_certificate as owner
from tnfr.research.maintained_sha_retention_evidence import RetainedShaStep


@pytest.fixture(autouse=True)
def no_actual_evidence(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic controls must not load actual retained evidence")

    monkeypatch.setattr(owner, "load_child_joint_window", forbidden)


@pytest.fixture(scope="module")
def synthetic_source():
    """A separate weighted 21-node example, never the retained child source."""
    edges = tuple(combinations(range(21), 2))[:53]
    weights = tuple(Q(1, 2) + Q(index % 3, 8) for index in range(53))
    capacities = (Q(1),) * 20 + (Q(4),)
    rows = tuple(
        tuple(
            (right if left == i else left, 1, weight)
            for (left, right), weight in zip(edges, weights, strict=True)
            if i in (left, right)
        )
        for i in range(21)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    flow = _held_sine_neighbor_field(model, tuple(map(len, rows)), capacities, rows)
    initial = tuple(map(I, (Q(1, 2),) * 20 + (Q(1, 4),))) + tuple(
        map(I, (Q(1, 8),) * 20 + (Q(0),))
    )
    step, _, reason = validated_box_taylor_step(
        initial,
        2 * owner.SAMPLE_STEP,
        flow,
        lambda state: (Q(1),),
        time=owner.CELL_START,
        order=4,
    )
    assert reason is None and step is not None
    retained = RetainedShaStep(
        owner.CELL_INDEX,
        owner.CELL_START,
        step.duration,
        step.initial_box,
        step.tube,
        step.endpoint,
        (Q(1),) * 53,
        (Q(1),) * 53,
    )
    records = ({},) * owner.CELL_INDEX + (
        {"index": owner.CELL_INDEX, "time": owner.CELL_START, "step": asdict(step)},
    )
    graph = SimpleNamespace(graph={"_t": 1.0, "THOL_BIFURCATION_THRESHOLD": 0.1})
    return SimpleNamespace(
        step=retained,
        retention=SimpleNamespace(
            records=records,
            admitted=SimpleNamespace(graph=graph),
            prediction=SimpleNamespace(
                capacities=capacities,
                edges=edges,
                conductances=weights,
                shifts=(0,) * 53,
            ),
        ),
    )


@pytest.fixture(scope="module")
def prepared(synthetic_source):
    return owner._prepare_child_sample_certificate(synthetic_source)


def _points(certificate):
    # Deliberately manufactured unit-test values, not numerical acquisitions.
    states = tuple(
        tuple(float(value.midpoint) for value in box) for box in certificate.boxes
    )
    pressures = tuple(
        tuple(
            float(value.midpoint)
            for value in owner._field(
                certificate.source, tuple(I(Q.from_float(x)) for x in state)
            ).state.pressure
        )
        for state in states
    )
    return states, pressures


def test_fixed_midpoint_uses_original_jets_and_keeps_both_endpoints(prepared):
    assert prepared.boxes[0] == prepared.source.step.initial
    assert prepared.boxes[2] == prepared.source.step.endpoint
    assert len(prepared.boxes[1]) == 42
    assert prepared.sample_times == tuple(
        Q(1) + value
        for value in (
            owner.CELL_START,
            owner.CELL_START + owner.SAMPLE_STEP,
            owner.CELL_END,
        )
    )
    assert all(
        value.subset_of(bound)
        for value, bound in zip(
            prepared.boxes[1], prepared.source.step.tube, strict=True
        )
    )
    # Synthetic child initially has positive pressure and acceleration; its
    # narrow interior form enclosure lies strictly between both endpoints.
    assert prepared.boxes[0][20].hi < prepared.boxes[1][20].lo
    assert prepared.boxes[1][20].hi < prepared.boxes[2][20].lo


def test_public_prepare_loads_once_and_passes_original_partial_evidence(
    synthetic_source, monkeypatch
):
    loads, partial_calls = [], []
    original = owner.reconstruct_box_taylor_partial

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return synthetic_source

    def partial(*args, **kwargs):
        partial_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "load_child_joint_window", load)
    monkeypatch.setattr(owner, "reconstruct_box_taylor_partial", partial)
    result = owner.prepare_child_sample_certificate("synthetic", repository_root="root")
    assert len(loads) == len(partial_calls) == 1
    args, kwargs = partial_calls[0]
    assert args[0] == synthetic_source.step.initial
    assert args[1] == synthetic_source.step.tube
    assert args[4:] == (2 * owner.SAMPLE_STEP, owner.SAMPLE_STEP)
    assert kwargs == {"order": 4}
    assert result.source is synthetic_source


def test_boundary_bounds_use_farther_endpoint_not_point_inclusion(prepared):
    states, pressures = _points(prepared)
    state = list(states[0])
    state[5] += float(owner.SAMPLE_ERROR_ALLOWANCE / 2)
    assert not prepared.boxes[0][5].contains(Q.from_float(state[5]))
    report = owner._assess_child_boundary(
        prepared,
        index=0,
        state=state,
        pressures=pressures[0],
        sample_time=prepared.sample_times[0],
    )
    assert report.passed
    assert report.state_error_bounds[5] == owner.SAMPLE_ERROR_ALLOWANCE / 2
    for value, box, error in zip(
        report.state, prepared.boxes[0], report.state_error_bounds
    ):
        assert error == max(abs(value - box.lo), abs(value - box.hi))
    assert report.child_error_bound == 0
    assert (
        len(report.exact_principal_margins)
        == len(report.represented_principal_margins)
        == 53
    )


def test_error_at_far_endpoint_is_not_mistaken_for_distance_to_box(prepared):
    states, pressures = _points(prepared)
    report = owner._assess_child_boundary(
        prepared,
        index=1,
        state=states[1],
        pressures=pressures[1],
        sample_time=prepared.sample_times[1],
    )
    assert report.passed
    for value, box, error in zip(
        report.state, prepared.boxes[1], report.state_error_bounds
    ):
        assert error >= box.width / 2
        assert error == max(abs(value - box.lo), abs(value - box.hi))


@pytest.mark.parametrize("coordinate", (0, 20, 21, 41))
def test_every_form_and_phase_coordinate_has_the_full_state_budget(
    prepared, coordinate
):
    states, pressures = _points(prepared)
    state = list(states[0])
    state[coordinate] += float(2 * owner.SAMPLE_ERROR_ALLOWANCE)
    report = owner._assess_child_boundary(
        prepared,
        index=0,
        state=state,
        pressures=pressures[0],
        sample_time=prepared.sample_times[0],
    )
    assert not report.passed
    assert not report.checks["full_state_error_within_allowance"]


@pytest.mark.parametrize("node", (0, 20))
def test_pressure_error_is_independent_of_perfect_coordinate_error(prepared, node):
    states, pressures = _points(prepared)
    pressure = list(pressures[0])
    pressure[node] += float(2 * owner.PRESSURE_ERROR_ALLOWANCE)
    report = owner._assess_child_boundary(
        prepared,
        index=0,
        state=states[0],
        pressures=pressure,
        sample_time=prepared.sample_times[0],
    )
    assert report.max_state_error_bound == 0
    assert not report.checks["exact_flow_pressure_error_within_allowance"]
    assert report.pressure_error_bounds[node] > owner.PRESSURE_ERROR_ALLOWANCE
    assert not report.passed


def test_exact_flow_and_current_point_pressure_defects_are_separate(prepared):
    states, _ = _points(prepared)
    state = list(states[0])
    state[41] += float(owner.SAMPLE_ERROR_ALLOWANCE / 2)
    point = owner._field(prepared.source, tuple(I(Q.from_float(x)) for x in state))
    pressures = tuple(float(value.midpoint) for value in point.state.pressure)
    report = owner._assess_child_boundary(
        prepared,
        index=0,
        state=state,
        pressures=pressures,
        sample_time=prepared.sample_times[0],
    )
    assert report.passed
    assert max(report.point_pressure_error_bounds) < Q(1, 2**46)
    assert report.pressure_error_bounds[20] > Q(1, 2**28)


def test_batch_rebuilds_whole_window_and_uses_actual_child_error_budget(prepared):
    states, pressures = _points(prepared)
    result = owner._assess_child_samples(
        prepared, states=states, pressures=pressures, sample_times=prepared.sample_times
    )
    assert result.passed
    assert all(item.passed for item in result.boundaries)
    assert result.reader_budget.sample_error_bounds == result.child_error_bounds
    errors = result.child_error_bounds
    assert (
        result.reader_budget.acceleration_sample_error_bound
        == (errors[0] + 2 * errors[1] + errors[2]) / owner.SAMPLE_STEP**2
    )
    assert result.reader_budget.exact
    assert result.reader_budget.represented_acceleration_bounds.lo > result.threshold


def test_replaceable_certificate_outputs_cannot_supply_assessment_premises(prepared):
    states, pressures = _points(prepared)
    altered = replace(
        prepared,
        boxes=((I(-100, 100),) * 42,) * 3,
        sample_times=(Q(0), Q(1), Q(2)),
    )
    correct = owner._assess_child_samples(
        prepared, states=states, pressures=pressures, sample_times=prepared.sample_times
    )
    rebuilt = owner._assess_child_samples(
        altered, states=states, pressures=pressures, sample_times=prepared.sample_times
    )
    assert rebuilt == correct


def test_public_assessment_reloads_and_accepts_exact_binary64_fraction_records(
    prepared, monkeypatch
):
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return prepared.source

    monkeypatch.setattr(owner, "load_child_joint_window", load)
    states, pressures = _points(prepared)
    result = owner.assess_child_samples(
        "synthetic",
        repository_root="root",
        states=tuple(tuple(Q.from_float(x) for x in row) for row in states),
        pressures=tuple(tuple(Q.from_float(x) for x in row) for row in pressures),
        sample_times=prepared.sample_times,
    )
    assert len(loads) == 1 and result.passed


@pytest.mark.parametrize(
    "bad", (True, float("nan"), float("inf"), Q(1, 3), Q(1, 2**1200), 1j)
)
@pytest.mark.parametrize("field", ("state", "pressures", "sample_time"))
def test_unadmitted_scalars_cannot_enter_binary64_records(prepared, bad, field):
    states, pressures = _points(prepared)
    arguments = dict(
        index=0,
        state=list(states[0]),
        pressures=list(pressures[0]),
        sample_time=prepared.sample_times[0],
    )
    if field == "sample_time":
        arguments[field] = bad
    else:
        arguments[field][0] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._assess_child_boundary(prepared, **arguments)


@pytest.mark.parametrize("index", (True, Q(0), -1, 3))
def test_boundary_index_does_not_coerce_or_wrap(prepared, index):
    states, pressures = _points(prepared)
    with pytest.raises(ValueError, match="boundary index"):
        owner._assess_child_boundary(
            prepared,
            index=index,
            state=states[0],
            pressures=pressures[0],
            sample_time=prepared.sample_times[0],
        )


def test_wrong_exact_sample_time_and_missing_rows_reject(prepared):
    states, pressures = _points(prepared)
    with pytest.raises(ValueError, match="fixed exact clock"):
        owner._assess_child_boundary(
            prepared,
            index=0,
            state=states[0],
            pressures=pressures[0],
            sample_time=prepared.sample_times[1],
        )
    for field in ("states", "pressures", "sample_times"):
        arguments = dict(
            states=states, pressures=pressures, sample_times=prepared.sample_times
        )
        arguments[field] = arguments[field][:-1]
        with pytest.raises(ValueError, match="exactly three"):
            owner._assess_child_samples(prepared, **arguments)


@pytest.mark.parametrize("field", ("state", "pressures"))
@pytest.mark.parametrize("change", ("short", "long", "unordered"))
def test_coordinate_sequences_are_complete_and_ordered(prepared, field, change):
    states, pressures = _points(prepared)
    arguments = dict(
        index=0,
        state=states[0],
        pressures=pressures[0],
        sample_time=prepared.sample_times[0],
    )
    row = arguments[field]
    arguments[field] = (
        row[:-1]
        if change == "short"
        else row + (0.0,) if change == "long" else set(row)
    )
    with pytest.raises((TypeError, ValueError)):
        owner._assess_child_boundary(prepared, **arguments)


@pytest.mark.parametrize("field", ("time", "duration", "initial_box", "endpoint"))
def test_raw_record_and_admitted_cell_association_cannot_diverge(
    synthetic_source, field
):
    source = deepcopy(synthetic_source)
    raw = source.retention.records[owner.CELL_INDEX]["step"]
    if field in ("time", "duration"):
        raw[field] += owner.SAMPLE_STEP
    else:
        raw[field] = ({"lo": Q(-1), "hi": Q(-1)},) + raw[field][1:]
    with pytest.raises(ValueError):
        owner._prepare_child_sample_certificate(source)


def test_bad_branch_is_not_hidden_by_small_coordinate_error(prepared):
    source = deepcopy(prepared.source)
    source.retention.prediction.shifts = (1,) + (0,) * 52
    altered = replace(prepared, source=source)
    states, pressures = _points(prepared)
    result = owner._assess_child_boundary(
        altered,
        index=0,
        state=states[0],
        pressures=pressures[0],
        sample_time=prepared.sample_times[0],
    )
    assert result.max_state_error_bound == 0
    assert not result.checks["exact_flow_original_branches_retained"]
    assert not result.checks["represented_original_branches_retained"]
    assert not result.passed
