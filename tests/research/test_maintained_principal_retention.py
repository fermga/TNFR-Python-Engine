"""Synthetic fixed-loop and admission controls; no reserved field is run."""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import maintained_principal_retention as owner

EDGES = tuple((i, j) for i in range(21) for j in range(i + 1, 21))[:53]
INITIAL = (I(0),) * 42
SHIFTS = (0,) * 53
ENERGY = (Q(-100),) * 53


@pytest.fixture
def prepared():
    calls, records = [], []

    def no_extra_observation(_):
        pytest.fail("the retention loop must not evaluate an extra field readout")

    def stepper(box, duration, flow, domain, *, order, time):
        assert flow is no_extra_observation
        assert duration == Q(1, 512) and order == 12
        assert time == len(calls) * duration
        assert box[0] == I(time)
        assert domain(box) == (Q(1),)
        endpoint = (box[0] + duration,) + box[1:]
        calls.append((box, endpoint))
        return SimpleNamespace(endpoint=endpoint, tube=box), None, None

    def record(value):
        assert len(calls) == len(records) + 1
        records.append(value)

    return (
        dict(
            box=INITIAL,
            flow=no_extra_observation,
            edges=EDGES,
            shifts=SHIFTS,
            energy_margins=ENERGY,
            record_step=record,
            monotonic_clock=lambda: 0,
            stepper=stepper,
        ),
        calls,
        records,
    )


def test_retention_only_loop_carries_full_state_without_a_velocity_target(prepared):
    inputs, calls, records = prepared
    result = owner.certify_retention(**inputs)
    assert type(result) is owner.RetentionAttempt
    assert result.status == "completed"
    assert result.accepted_steps == result.attempted_steps == len(records) == 256
    assert result.certified_retention_time == Q(1, 2)
    assert result.minimum_retained_margin == pi_interval().lo
    assert result.full_horizon_retention
    assert calls[-1][1][0] == I(Q(1, 2))
    assert set(records[0]) == {
        "index",
        "time",
        "step",
        "tube_principal_margins",
        "combined_principal_margins",
    }
    assert not hasattr(result, "reversal_certified")


def test_source_preparation_can_be_included_without_an_extra_clock_read(prepared):
    inputs, calls, records = prepared
    times = iter((899, 900, 900))
    inputs.update(started=0, monotonic_clock=lambda: next(times))
    result = owner.certify_retention(**inputs)
    assert result.status == "resource_budget_exhausted"
    assert result.attempted_steps == result.accepted_steps == len(calls) == 1
    assert len(records) == 1 and not result.full_horizon_retention
    assert result.certified_retention_time == Q(1, 512)
    assert result.elapsed_seconds == 900


def test_failed_enclosure_is_recorded_and_stops_without_a_retry(prepared):
    inputs, calls, _ = prepared
    records = []

    def fail(box, *args, **kwargs):
        calls.append(None)
        return None, box, "synthetic failed strict inclusion"

    inputs.update(stepper=fail, record_step=records.append)
    result = owner.certify_retention(**inputs)
    assert result.status == "enclosure_unavailable"
    assert result.attempted_steps == len(calls) == len(records) == 1
    assert result.accepted_steps == result.certified_retention_time == 0
    assert records[0]["failed_tube"] == INITIAL
    assert records[0]["reason"] == "synthetic failed strict inclusion"
    assert not result.full_horizon_retention


@pytest.mark.parametrize("global_bound", (Q(-100), Q(1, 3)))
def test_same_law_global_bound_is_distinct_from_a_wide_picard_tube(
    prepared, global_bound
):
    inputs, _, records = prepared
    original = inputs["stepper"]

    def wide(*args, **kwargs):
        step, failed, reason = original(*args, **kwargs)
        step.tube = step.tube[:22] + (I(-10, 10),) + step.tube[23:]
        return step, failed, reason

    inputs.update(stepper=wide, energy_margins=(global_bound,) * 53)
    result = owner.certify_retention(**inputs)
    assert min(records[0]["tube_principal_margins"]) < 0
    if global_bound > 0:
        assert result.full_horizon_retention and result.accepted_steps == 256
        assert result.minimum_retained_margin == global_bound
    else:
        assert result.status == "principal_bound_unresolved"
        assert result.accepted_steps == 0 and len(records) == 1


def test_recording_failure_propagates_before_another_step(prepared):
    inputs, calls, _ = prepared

    def unavailable(_):
        raise OSError("synthetic durable write failure")

    inputs["record_step"] = unavailable
    with pytest.raises(OSError, match="durable write"):
        owner.certify_retention(**inputs)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "key,value",
    (
        ("box", INITIAL[:-1]),
        ("box", (True,) + INITIAL[1:]),
        ("edges", EDGES[:-1]),
        ("edges", (EDGES[0],) + EDGES[:-1]),
        ("edges", ((False, 1),) + EDGES[1:]),
        ("shifts", SHIFTS[:-1]),
        ("shifts", (False,) + SHIFTS[1:]),
        ("energy_margins", ENERGY[:-1]),
        ("energy_margins", (True,) + ENERGY[1:]),
        ("energy_margins", (float("nan"),) + ENERGY[1:]),
        ("energy_margins", (1.0,) + ENERGY[1:]),
        ("flow", None),
        ("record_step", None),
        ("monotonic_clock", None),
        ("stepper", False),
        ("started", True),
        ("started", float("inf")),
    ),
)
def test_invalid_prepared_premises_are_rejected_before_the_stepper(
    prepared, key, value
):
    inputs, calls, _ = prepared
    inputs[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner.certify_retention(**inputs)
    assert not calls


@pytest.mark.parametrize("clock_value", (True, float("nan"), float("inf"), -1))
def test_invalid_resource_clock_cannot_disable_the_fixed_budget(prepared, clock_value):
    inputs, calls, _ = prepared
    inputs.update(started=0, monotonic_clock=lambda: clock_value)
    with pytest.raises((TypeError, ValueError)):
        owner.certify_retention(**inputs)
    assert not calls


def test_initial_branch_is_required_even_if_a_supplied_global_margin_is_positive(
    prepared,
):
    inputs, calls, _ = prepared
    inputs["box"] = INITIAL[:22] + (2 * pi_interval(),) + INITIAL[23:]
    inputs["energy_margins"] = (Q(1),) * 53
    with pytest.raises(ValueError, match="initial principal identity"):
        owner.certify_retention(**inputs)
    assert not calls
