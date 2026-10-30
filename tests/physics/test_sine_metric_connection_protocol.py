"""Software-only connection observations and frozen transport controls.

Synthetic forecast fixtures below test wiring and interval observations. They
are deliberately not trajectories and must never become scientific evidence.
The only real engine work in this suite is static rational preparation.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import pytest

from benchmarks import sine_metric_connection as producer
from tests.physics.test_sine_cycle_barrier import _state
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_sine_metric_connection as engine
from tnfr.physics.relational_sine_corridor import prepare_sine_saddle_state
from tnfr.sdk import relational_report_to_dict
from tnfr.utils.io import json_loads


def _box(kind):
    phase = (Q(0),) * 5 if kind == 0 else tuple(kind * Q(5 * i, 4) for i in range(5))
    return (I(0),) * 10 + tuple(map(I, phase * 2))


@dataclass(frozen=True)
class SoftwareStep:
    time: Q
    duration: Q
    endpoint: tuple[I, ...]
    tube: tuple[I, ...]


@dataclass(frozen=True)
class SoftwareForecast:
    """A nonphysical stand-in for the private fresh producer boundary only."""

    initial_center: tuple[Q, ...]
    direction: int
    steps: tuple[SoftwareStep, ...]
    admitted: bool = True
    duration: Q = Q(4)
    validated_duration: Q = Q(4)

    def to_dict(self):
        return {"schema": "test-only-not-a-scientific-forecast", "report": {}}


def _software_pair(source):
    center = source.epi + source.phase
    forward = SoftwareForecast(
        center,
        1,
        (
            SoftwareStep(Q(0), Q(2), _box(1), _box(1)),
            SoftwareStep(Q(2), Q(2), _box(0), _box(0)),
        ),
    )
    backward = SoftwareForecast(
        center,
        -1,
        tuple(SoftwareStep(Q(time), Q(2), _box(1), _box(1)) for time in (0, 2)),
    )
    return forward, backward


def test_observations_select_first_endpoint_and_whole_positive_winding_window():
    forward, backward = _software_pair(_state())
    endpoints, tubes, first, window = engine._connection_observations(
        forward, backward, tuple(range(5))
    )
    assert tuple(row.winding for row in endpoints) == (1, 0)
    assert first.step_index == 1 and first.time == 4
    assert all(row.winding == 1 for row in tubes)
    assert window[:2] == (0, 4) and window[2] > 0
    assert window[1] - window[0] >= pi_interval().hi


@pytest.mark.parametrize("case", ("opposite", "gap", "short", "nonacute", "ambiguous"))
def test_endpoint_twist_or_disjoint_short_tubes_cannot_replace_whole_window(case):
    forward, backward = _software_pair(_state())
    steps = list(backward.steps)
    if case == "opposite":
        steps = [replace(step, tube=_box(-1)) for step in steps]
    elif case == "gap":
        steps[1] = replace(steps[1], time=Q(3))
    elif case == "short":
        steps = steps[:1]
    elif case == "nonacute":
        phase = list(_box(1))
        phase[11] = pi_interval() / 2
        steps = [replace(step, tube=tuple(phase)) for step in steps]
    else:
        steps = [replace(step, tube=(I(0),) * 10 + (I(-4, 4),) * 10) for step in steps]
    _, _, _, window = engine._connection_observations(
        forward, replace(backward, steps=tuple(steps)), tuple(range(5))
    )
    assert window is None


@pytest.mark.parametrize("partial", (False, True))
def test_shared_engine_executes_both_directions_from_same_readmitted_state(
    monkeypatch, partial
):
    source = _state()
    forwards, backwards = _software_pair(source)
    if partial:
        forwards = replace(forwards, admitted=False, duration=Q(6))
    calls = []

    def forecast(actual, **options):
        calls.append((actual, options))
        assert actual.storage != I(-999)
        return forwards if options["direction"] == 1 else backwards

    monkeypatch.setattr(engine, "forecast_sine_saddle_metric", forecast)
    result = engine.forecast_sine_metric_connection(
        replace(source, storage=I(-999), form_rates=()),
        cycle=range(5),
        duration=Q(4),
        time_step=Q(2),
        initial_coordinate_radius=Q(0),
    )
    assert [options["direction"] for _, options in calls] == [1, -1]
    assert calls[0][0] is calls[1][0]
    assert calls[0][1] | {"direction": -1} == calls[1][1]
    assert calls[0][1]["phase_radius"] is None
    assert result.same_orbit_connection_certified
    assert result.declared_horizons_complete is not partial
    assert not result.independent_zero_winding_source_ball_certified
    assert result.outcome == "same_orbit_connection_certified"
    assert result.forward_zero_winding_step_index == 1
    assert result.backward_retention_end == 4
    # Shared export wiring must preserve the independent-source limitation.
    assert (
        relational_report_to_dict(result)["report"][
            "independent_zero_winding_source_ball_certified"
        ]
        is False
    )


@pytest.mark.parametrize(
    "partial,expected",
    ((False, "no_certificate_on_declared_horizons"), (True, "unavailable")),
)
def test_complete_noncertificate_and_partial_unavailability_are_distinct(
    monkeypatch, partial, expected
):
    source = _state()
    forward, backward = _software_pair(source)
    forward = replace(
        forward, steps=tuple(replace(step, endpoint=_box(1)) for step in forward.steps)
    )
    if partial:
        forward = replace(forward, admitted=False, validated_duration=Q(2))
    calls = []

    def forecast(actual, **options):
        calls.append(options["direction"])
        return forward if options["direction"] == 1 else backward

    monkeypatch.setattr(engine, "forecast_sine_saddle_metric", forecast)
    result = engine.forecast_sine_metric_connection(
        source,
        cycle=range(5),
        duration=Q(4),
        time_step=Q(2),
        initial_coordinate_radius=0,
    )
    assert calls == [1, -1]
    assert result.outcome == expected
    assert not result.same_orbit_connection_certified


def test_fresh_producer_cannot_lose_its_common_source_association(monkeypatch):
    source = _state()
    forward, backward = _software_pair(source)
    forward = replace(forward, initial_center=(Q(9),) * 20)
    monkeypatch.setattr(
        engine,
        "forecast_sine_saddle_metric",
        lambda source, **options: forward if options["direction"] == 1 else backward,
    )
    with pytest.raises(ArithmeticError, match="initial-state association"):
        engine.forecast_sine_metric_connection(
            source, cycle=range(5), duration=4, time_step=2, initial_coordinate_radius=0
        )


def test_ordered_growth_generator_is_admitted_once_for_both_directions(monkeypatch):
    source = _state()
    forward, backward = _software_pair(source)
    bounds = []

    def forecast(actual, **options):
        bounds.append(options["growth_rate_bounds"])
        return forward if options["direction"] == 1 else backward

    monkeypatch.setattr(engine, "forecast_sine_saddle_metric", forecast)
    result = engine.forecast_sine_metric_connection(
        source,
        cycle=range(5),
        duration=4,
        time_step=2,
        initial_coordinate_radius=0,
        growth_rate_bounds=(value for value in (Q(0), Q(1))),
    )
    assert result.same_orbit_connection_certified
    assert bounds == [(Q(0), Q(1)), (Q(0), Q(1))]


@pytest.fixture(scope="module")
def prepared():
    return prepare_sine_saddle_state(_state(), cycle=range(5))


@pytest.fixture
def declaration(prepared):
    return {
        "schema": "tnfr.sine-metric-connection-declaration.v1",
        "law": "normalized_sine_reciprocal_e0_w1_beta1",
        "support": "fixed_simple_connected_unit_undirected",
        "forcing": "none",
        "events": "none",
        "clock": "original_structural_t; tau=t/pi",
        "selection_rule": "first_forward_zero_winding_endpoint_and_first_backward_acute_winding_one_window",
        "directions": [1, -1],
        "nodes": list(range(10)),
        "edges": [list(edge) for edge in prepared.prepared_state.edges],
        "cycle_indices": list(range(5)),
        "epsilon": str(prepared.epsilon),
        "form": list(map(str, prepared.prepared_state.epi)),
        "phase": list(map(str, prepared.prepared_state.phase)),
        "capacity": [1] * 10,
        "phase_radius": None,
        "duration": "256",
        "time_step": "1",
        "order": 16,
        "max_steps": 256,
        "growth_rate_bounds": ["0", "1"],
        "growth_bisections": 8,
        "initial_coordinate_radius": "0",
        "retained_scaled_duration": "1",
        "acute_margin": "0",
    }


def test_preflight_recomputes_literal_preparation_without_reserved_flow(
    declaration, prepared, monkeypatch
):
    monkeypatch.setattr(
        producer,
        "forecast_sine_metric_connection",
        lambda *args, **kwargs: pytest.fail(
            "preparation must not run the reserved forecast"
        ),
    )
    source, admission, options = producer._preparation(declaration)
    assert source.epi == prepared.prepared_state.epi
    assert source.phase == prepared.prepared_state.phase
    assert admission.preparation_certified
    assert options["duration"] == 256 and options["time_step"] == 1
    assert options["growth_rate_bounds"] == (0, 1)


def test_benchmark_calls_the_shared_connection_owner_once(declaration, monkeypatch):
    calls = []
    response = {
        "schema": "tnfr.sine-metric-connection.v1",
        "report": {"outcome": "unavailable"},
    }

    class Result:
        def to_dict(self):
            return response

    def call(source, **options):
        calls.append((source, options))
        return Result()

    monkeypatch.setattr(producer, "forecast_sine_metric_connection", call)
    assert producer.evaluate(declaration) is response
    assert len(calls) == 1 and len(calls[0][0].epi + calls[0][0].phase) == 20


@pytest.mark.parametrize(
    "key,value",
    (
        ("forcing", "external"),
        ("events", "allowed"),
        ("clock", "scaled_tau"),
        ("directions", [True, -1]),
        ("directions", [-1, 1]),
        ("phase_radius", "1/1000"),
        ("duration", True),
        ("duration", 0),
        ("duration", "257"),
        ("time_step", False),
        ("time_step", 0),
        ("order", True),
        ("order", 17),
        ("max_steps", True),
        ("max_steps", 255),
        ("growth_bisections", True),
        ("growth_bisections", 17),
        ("growth_rate_bounds", [0, True]),
        ("growth_rate_bounds", [1, 0]),
        ("initial_coordinate_radius", True),
        ("initial_coordinate_radius", -1),
        ("retained_scaled_duration", 0),
        ("acute_margin", "1/1000"),
        ("cycle_indices", [False, 1, 2, 3, 4]),
    ),
)
def test_invalid_declarations_reject_before_evaluation(
    declaration, monkeypatch, key, value
):
    declaration[key] = value
    monkeypatch.setattr(
        producer,
        "forecast_sine_metric_connection",
        lambda *args, **kwargs: pytest.fail("malformed declaration consumed a flow"),
    )
    with pytest.raises((TypeError, ValueError)):
        producer.evaluate(declaration)


def test_literal_rational_association_is_stricter_than_equal_materialized_intervals(
    declaration,
):
    original = Q(declaration["phase"][0])
    changed = original + Q(1, 2**250)
    assert I(original) == I(changed)
    declaration["phase"][0] = str(changed)
    with pytest.raises(ValueError, match="literal state differs"):
        producer._preparation(declaration)


def _transport(declaration, monkeypatch, tmp_path):
    path = tmp_path / "declaration.json"
    path.write_text(producer.evidence._encoded(declaration), encoding="utf-8")
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    monkeypatch.setattr(
        producer,
        "_source_files",
        lambda selected: {
            "declaration.json": selected.read_bytes(),
            "software_fixture.py": b"no scientific flow evaluated",
        },
    )
    output = tmp_path / "response.json"
    return path, output, ["--declaration", str(path), "--output", str(output)]


@pytest.mark.parametrize(
    "certified,complete", ((True, True), (True, False), (False, True), (False, False))
)
def test_freeze_and_evaluate_are_separate_and_every_outcome_is_retained_once(
    declaration, monkeypatch, tmp_path, certified, complete
):
    _, output, args = _transport(declaration, monkeypatch, tmp_path)
    calls = []

    def evaluate(actual):
        calls.append(actual)
        return {
            "schema": "tnfr.sine-metric-connection.v1",
            "report": {
                "same_orbit_connection_certified": certified,
                "declared_horizons_complete": complete,
            },
        }

    monkeypatch.setattr(producer, "evaluate", evaluate)
    assert producer.main(["--prepare", *args]) == 0
    assert not calls
    assert producer.main(args) == (0 if certified and complete else 1)
    record = json_loads(output.read_bytes())
    assert record["passed"] is (certified and complete)
    assert record["evaluation_error"] is None
    assert record["protocol"]["declaration"] == declaration
    assert len(calls) == 1
    retained = output.read_bytes()
    with pytest.raises(FileExistsError):
        producer.main(args)
    assert output.read_bytes() == retained and len(calls) == 1


def test_changed_declaration_cannot_evaluate_a_frozen_protocol(
    declaration, monkeypatch, tmp_path
):
    path, output, args = _transport(declaration, monkeypatch, tmp_path)
    assert producer.main(["--prepare", *args]) == 0
    changed = deepcopy(declaration)
    changed["initial_coordinate_radius"] = "1/1099511627776"
    path.write_text(producer.evidence._encoded(changed), encoding="utf-8")
    monkeypatch.setattr(
        producer,
        "evaluate",
        lambda actual: pytest.fail("changed declaration must not be evaluated"),
    )
    with pytest.raises(
        ValueError, match="frozen source, runtime or declaration changed"
    ):
        producer.main(args)
    assert not output.exists()


def test_unexpected_response_schema_is_retained_as_failure(
    declaration, monkeypatch, tmp_path
):
    _, output, args = _transport(declaration, monkeypatch, tmp_path)
    assert producer.main(["--prepare", *args]) == 0
    monkeypatch.setattr(
        producer,
        "evaluate",
        lambda actual: {"schema": "not-the-metric-connection", "report": {}},
    )
    assert producer.main(args) == 1
    record = json_loads(output.read_bytes())
    assert record["passed"] is False
    assert record["evaluation_error"]["error_type"] == "ValueError"
