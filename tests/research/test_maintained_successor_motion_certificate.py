"""Observation and frozen-loop wiring controls; no reserved flow is evaluated."""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import maintained_successor_motion_certificate as owner


@pytest.fixture
def wired(monkeypatch):
    # Deliberately synthetic: a clock marker drives a sign-changing readout.
    # This checks orchestration, not the scientific sine law or its certificate.
    edges = tuple((i, j) for i in range(21) for j in range(i + 1, 21))[:53]
    chamber = SimpleNamespace(
        edges=edges, conditional=SimpleNamespace(shifts=(0,) * 53)
    )
    prediction = SimpleNamespace(
        chamber=chamber,
        conditional=SimpleNamespace(horizon_remaining_margins=(I(-100),) * 53),
    )
    initial = (I(0),) * 42

    def flow(state):
        result = [I(0)] * 42
        result[41] = 2 * state[0] - Q(1, 8)
        return tuple(result)

    monkeypatch.setattr(owner, "prepare_motion", lambda _: (prediction, initial, flow))
    calls = []

    def stepper(box, duration, field, domain, *, order, time):
        assert duration == Q(1, 512) and order == 12
        assert time == len(calls) * duration
        assert box[0] == I(time)
        assert domain(box) == (Q(1),)
        endpoint = (box[0] + duration,) + box[1:]
        calls.append((box, endpoint))
        return SimpleNamespace(endpoint=endpoint, tube=box), None, None

    monkeypatch.setattr(owner, "validated_box_taylor_step", stepper)
    return calls, stepper, chamber


def test_fixed_loop_carries_complete_endpoint_and_records_before_next_step(wired):
    calls, _, _ = wired
    records = []

    def record(value):
        assert len(calls) == len(records) + 1
        records.append(value)

    result = owner.certify_motion(None, record_step=record, monotonic_clock=lambda: 0)
    assert len(records) == 256
    assert result.full_horizon_retention and result.joint_target_certified
    assert result.certified_retention_time == Q(1, 2)
    assert result.positive_velocity_times[0] == Q(33, 512)
    assert result.positive_velocity_times[-1] == Q(1, 2)
    assert result.initial_relative_velocity == I(Q(-1, 8))
    assert set(records[0]) == {
        "index",
        "time",
        "step",
        "tube_principal_margins",
        "combined_principal_margins",
        "endpoint_relative_velocity",
    }


def test_failed_enclosure_is_retained_without_retry_or_escape_claim(wired, monkeypatch):
    _, stepper, _ = wired

    def fail(box, *args, **kwargs):
        if kwargs["time"] == Q(2, 512):
            return None, box, "strict Picard inclusion unavailable"
        return stepper(box, *args, **kwargs)

    monkeypatch.setattr(owner, "validated_box_taylor_step", fail)
    records = []
    result = owner.certify_motion(
        None, record_step=records.append, monotonic_clock=lambda: 0
    )
    assert result.status == "enclosure_unavailable"
    assert result.accepted_steps == 2 and result.attempted_steps == 3
    assert result.certified_retention_time == Q(1, 256)
    assert records[-1]["step"] is None and records[-1]["failed_tube"] is not None
    assert not result.full_horizon_retention and not result.joint_target_certified


def test_unresolved_branch_does_not_promote_an_enclosed_step_to_identity(
    wired, monkeypatch
):
    _, stepper, _ = wired

    def unresolved(*args, **kwargs):
        step, _, _ = stepper(*args, **kwargs)
        step.tube = step.tube[:22] + (pi_interval(),) + step.tube[23:]
        return step, None, None

    monkeypatch.setattr(owner, "validated_box_taylor_step", unresolved)
    records = []
    result = owner.certify_motion(
        None, record_step=records.append, monotonic_clock=lambda: 0
    )
    assert result.status == "principal_bound_unresolved"
    assert result.accepted_steps == 0 and result.certified_retention_time == 0
    assert records[0]["step"] is not None


def test_resource_stop_is_a_method_outcome_and_preserves_completed_prefix(wired):
    times = iter((0, 0, 901, 901))
    result = owner.certify_motion(
        None, record_step=lambda _: None, monotonic_clock=lambda: next(times)
    )
    assert result.status == "resource_budget_exhausted"
    assert result.accepted_steps == result.attempted_steps == 1
    assert result.certified_retention_time == Q(1, 512)
    assert not result.full_horizon_retention


def test_initial_resource_stop_retains_unavailable_zero_step_result(wired):
    times = iter((0, 901, 901))
    records = []
    result = owner.certify_motion(
        None, record_step=records.append, monotonic_clock=lambda: next(times)
    )
    assert result.status == "resource_budget_exhausted"
    assert result.accepted_steps == result.attempted_steps == 0
    assert result.certified_retention_time == 0 and records == []


def test_independent_global_energy_bound_can_cover_a_wide_tube(wired, monkeypatch):
    _, stepper, _ = wired
    prediction, initial, flow = owner.prepare_motion(None)
    prediction.conditional.horizon_remaining_margins = (I(Q(1, 3)),) * 53

    def wide(*args, **kwargs):
        step, _, _ = stepper(*args, **kwargs)
        step.tube = step.tube[:22] + (I(-10, 10),) + step.tube[23:]
        return step, None, None

    monkeypatch.setattr(owner, "validated_box_taylor_step", wide)
    records = []
    result = owner.certify_motion(
        None, record_step=records.append, monotonic_clock=lambda: 0
    )
    assert result.full_horizon_retention
    assert min(records[0]["tube_principal_margins"]) < 0
    assert min(records[0]["combined_principal_margins"]) > 0


def test_observation_retains_every_original_branch_not_wrapped_endpoint(wired):
    _, _, chamber = wired
    state = [I(0)] * 42
    state[22] = 2 * pi_interval() + Q(1, 4)
    margins = owner._principal_margins(state, chamber.edges, (0,) * 53)
    assert margins[0] < 0  # Wrapping would have concealed this original-branch loss.
    shifted = owner._principal_margins(state, chamber.edges, (1,) + (0,) * 52)
    assert shifted[0] > 0
    with pytest.raises(ValueError, match="42 rows"):
        owner._principal_margins(state[:-1], chamber.edges, (0,) * 53)


def test_recording_exception_does_not_silently_resume(wired):
    calls, _, _ = wired

    def unavailable(_):
        raise OSError("durable recording unavailable")

    with pytest.raises(OSError, match="durable recording"):
        owner.certify_motion(None, record_step=unavailable, monotonic_clock=lambda: 0)
    assert len(calls) == 1
