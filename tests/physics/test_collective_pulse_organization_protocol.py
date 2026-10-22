"""Exact preparation and shared producer wiring; no research trajectory replay."""

from fractions import Fraction as Q

import pytest

from benchmarks import conservative_regional_organization as owner
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.utils.io import json_loads

DECLARATION = owner.ROOT / "docs/assets/collective_pulse_organization/declaration.json"


def test_pulse_preparation_retains_complete_exact_budget_and_absent_identity(
    monkeypatch,
):
    def forbidden(**kwargs):
        pytest.fail("preparation must not evaluate its reserved trajectory")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    protocol = owner.prepare_protocol(DECLARATION)
    assert (
        protocol["declaration_path"] == DECLARATION.relative_to(owner.ROOT).as_posix()
    )
    prepared = owner._preparation(protocol["declaration"])
    assert len(prepared["initial"]) == 21
    x = tuple(Q(value) for value in protocol["declaration"]["form"])
    assert all(box.lo <= value <= box.hi for box, value in zip(prepared["initial"], x))
    assert all(value.lo == value.hi == 0 for value in prepared["initial"][10:20])
    edges = protocol["declaration"]["edges"]
    storage = sum((x[i] - x[j]) ** 2 / 2 for i, j in edges)
    assert storage == Q(14201, 4000) > Q(7, 2)
    assert sum(len(prepared["neighbors"][i]) * x[i] for i in range(10)) == 0
    assert prepared["end_time"] / prepared["time_step"] == 256
    assert prepared["order"] == 12
    # A nonconstant adjacent signed dipole has no ordinary cycle reflection.
    assert not any(all(x[i] == x[(k - i) % 5] for i in range(5)) for k in range(5))


@pytest.mark.parametrize("bad", (True, float("inf"), "nan"))
def test_preparation_rejects_invalid_state_before_freezing(bad):
    declaration = json_loads(DECLARATION.read_bytes())
    declaration["form"][0] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._preparation(declaration)


def test_preparation_rejects_budget_mismatch():
    declaration = json_loads(DECLARATION.read_bytes())
    declaration["maximum_steps"] = 255
    with pytest.raises(ValueError, match="step budget"):
        owner._preparation(declaration)


@pytest.mark.parametrize(
    "key",
    (
        "observation_time",
        "end_time",
        "time_step",
        "minimum_duration",
        "acute_margin",
        "order",
    ),
)
def test_preparation_rejects_boolean_budgets_before_coercion(key):
    declaration = json_loads(DECLARATION.read_bytes())
    declaration[key] = True
    with pytest.raises((TypeError, ValueError)):
        owner._preparation(declaration)


def test_selected_declaration_is_frozen_and_evaluated_once(monkeypatch, tmp_path):
    calls = []

    def evaluate(declaration):
        calls.append(declaration)
        return {"report": {"outcome": "unresolved"}}

    monkeypatch.setattr(owner, "evaluate", evaluate)
    output = tmp_path / "response.json"
    args = ["--declaration", str(DECLARATION), "--output", str(output)]
    assert owner.main(["--prepare", *args]) == 0
    assert calls == []
    assert owner.main(args) == 1
    record = json_loads(output.read_bytes())
    assert calls == [json_loads(DECLARATION.read_bytes())]
    assert record["evaluation_error"] is None
    assert record["passed"] is False
    assert record["protocol"]["declaration_path"] in record["protocol"]["source_sha256"]
    with pytest.raises(FileExistsError):
        owner.main(args)


def test_selected_declaration_requires_new_output():
    with pytest.raises(SystemExit):
        owner.main(["--prepare", "--declaration", str(DECLARATION)])


def test_retained_pulse_control_excludes_winding_from_complete_primitive_tubes(
    monkeypatch,
):
    """Audit frozen source binding and geometric implications without replay."""

    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence must not replay its scientific producer")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    monkeypatch.setattr(owner, "evaluate", forbidden)
    output = DECLARATION.with_name("response-v1.json")
    forecast, cycle, metadata = owner._load_channel_source(output)
    record = json_loads(output.read_bytes())
    report = record["response"]["report"]
    assert metadata["source_response_sha256"]
    assert (
        record["protocol"]["declaration_path"]
        == DECLARATION.relative_to(owner.ROOT).as_posix()
    )
    assert record["evaluation_error"] is None and record["passed"] is False
    assert forecast.validated_end_time == forecast.end_time == 24
    assert len(forecast.steps) == len(report["steps"]) == 256
    assert cycle == tuple(range(5))
    pi, at = pi_interval(), Q(0)
    box = forecast.initial_box
    for step, observation in zip(forecast.steps, report["steps"]):
        assert step.time == at and step.duration == Q(3, 32)
        # Every lifted receiver gap lies in its principal branch throughout
        # the supplied validated tube. Actual node differences telescope,
        # hence their winding is zero at every time, not only sampled times.
        phases = step.tube[10:20]
        gaps = tuple(
            phases[j] - phases[i] for i, j in zip(cycle, cycle[1:] + cycle[:1])
        )
        assert all(-pi.lo < gap.lo <= gap.hi < pi.lo for gap in gaps)
        assert observation["winding"] == 0
        assert observation["edge_turn_offsets"] == [0] * 5
        assert (
            tuple(map(owner._interval_projection, observation["cycle_raw_gap_bounds"]))
            == gaps
        )
        assert all(
            tube.lo <= initial.lo <= initial.hi <= tube.hi
            for tube, initial in zip(step.tube, box)
        )
        box = step.endpoint
        at += step.duration
    assert at == forecast.end_time and box == forecast.endpoint
    assert report["outcome"] == "acute_winding_excluded_on_horizon"
