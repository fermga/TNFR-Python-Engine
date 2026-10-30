"""Read-only audit of the single frozen child OZ/history acquisition.

The original SHA solution, numerical path, pressure-only event and actual
history have separate evidence. These tests inspect retained data and perform
arithmetic checks; they never replay a producer, integrator or event. Existing
ancestor admission may rebuild its own earlier history theorem; no retained
SHA derivative producer is replayed.
Archive receipts associate bytes, not acquisition chronology or authenticity.
"""

from fractions import Fraction as Q
from math import fsum, pi
from math import sin as scalar_sin
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions import Dissonance, SelfOrganization, Silence
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.research import local_capacity_history
from tnfr.research import maintained_child_history_acquisition as producer
from tnfr.research import (
    maintained_parent_sha_execution,
    maintained_parent_thol_execution,
    maintained_parent_um_execution,
    maintained_principal_retention,
    maintained_sha_motion,
    maintained_successor_motion_certificate,
)
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.maintained_child_oz_timing import _admit_child_oz_timing_source
from tnfr.research.maintained_child_sample_certificate import (
    _assess_child_samples,
    prepare_child_sample_certificate,
)
from tnfr.research.maintained_parent_sha_source import _same
from tnfr.research.maintained_parent_thol import _represented, _signature
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.types import Glyph
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_history"
JOINT = ROOT / "docs/assets/maintained_child_joint_window"
CHILD, CHILD_INDEX = "2_sub_0", 20
STEP, SAMPLE_STEP, EPSILON = Q(1, 2**14), Q(1, 1024), Q(1, 2**24)
MAX_BYTES = 128 * 1024**2


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=MAX_BYTES))
    )


def _receipt(base, record):
    assert type(record) is dict and set(record) == {"path", "bytes", "sha256"}
    name = validate_artifact_path(record["path"])
    path = (base / name).resolve()
    assert path.is_relative_to(base.resolve())
    assert type(record["bytes"]) is int and record["bytes"] >= 0
    assert file_receipt(path, label=name, max_bytes=MAX_BYTES) == record


def _values(graph, aliases):
    return tuple(
        _represented(
            get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value),
            "retained state coordinate",
        )
        for node in graph
    )


def _state(graph):
    return _values(graph, ALIAS_EPI) + _values(graph, ALIAS_THETA)


def _neighbors(prediction):
    rows = [[] for _ in prediction.nodes]
    for (left, right), weight in zip(
        prediction.edges, prediction.conductances, strict=True
    ):
        rows[left].append((right, weight))
        rows[right].append((left, weight))
    return rows


def _pressure_bounds(state, prediction):
    theta = tuple(I.coerce(value) for value in state[21:])
    return tuple(
        sum((sin(theta[j] - theta[i]) for j, _ in row), I(0))
        / (pi_interval() * len(row))
        for i, row in enumerate(_neighbors(prediction))
    )


def _independent_rates(state, prediction):
    """Node-wise evaluation of recorded stage inputs, never a time step."""
    x, theta = tuple(map(float, state[:21])), tuple(map(float, state[21:]))
    rows = _neighbors(prediction)
    mobility = tuple(
        float(nu) / (pi * len(row))
        for nu, row in zip(prediction.capacities, rows, strict=True)
    )
    form = tuple(
        Q(mobility[i] * fsum(scalar_sin(theta[j] - theta[i]) for j, _ in row))
        for i, row in enumerate(rows)
    )
    phase = tuple(
        Q(mobility[i] * fsum(float(weight) * (x[i] - x[j]) for j, weight in row))
        for i, row in enumerate(rows)
    )
    return form + phase


def _rn(value):
    """One independent nearest binary64 rounding of exact supplied operands."""
    return Q(float(value))


def _stage_from_recorded_rate(initial, rates, duration):
    return tuple(
        _rn(x + _rn(duration * rate)) for x, rate in zip(initial, rates, strict=True)
    )


def _endpoint_from_recorded_rates(initial, stages):
    factor = _rn(STEP / 6)
    result = []
    for index, x in enumerate(initial):
        k1, k2, k3, k4 = (stage["rates"][index] for stage in stages)
        weighted = _rn(_rn(_rn(k1 + _rn(2 * k2)) + _rn(2 * k3)) + k4)
        result.append(_rn(x + _rn(factor * weighted)))
    return tuple(result)


@pytest.fixture(scope="module", autouse=True)
def no_acquisition_event_refresh_integrator_or_sha_producer():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "retained evidence must not execute a new acquisition or trajectory"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                producer,
                (
                    "evaluate_child_history",
                    "_acquire_child_history",
                    "_advance",
                    "execute_network_operator_stage",
                    "refresh_pressure",
                    "rk4_coupled_step",
                ),
            ),
            (
                local_capacity_history,
                (
                    "evaluate_local_history",
                    "declared_preparation",
                    "refresh_pressure",
                    "rk4_coupled_step",
                ),
            ),
            (
                maintained_parent_sha_execution,
                ("evaluate_sha_event", "refresh_pressure"),
            ),
            (maintained_parent_um_execution, ("evaluate_um_event", "refresh_pressure")),
            (maintained_parent_thol_execution, ("evaluate_thol_event",)),
            (
                maintained_sha_motion,
                ("certify_sha_retention", "validated_box_taylor_step"),
            ),
            (maintained_principal_retention, ("certify_retention",)),
            (maintained_successor_motion_certificate, ("certify_motion",)),
            (
                _validated_taylor,
                (
                    "validated_box_taylor_step",
                    "validated_taylor_step",
                ),
            ),
            (integrators, ("rk4_coupled_step",)),
            (relational, ("step_relational_exchange",)),
            (word_execution, ("execute_network_operator_stage",)),
            (
                network_stage,
                ("execute_dissonance_stage", "execute_self_organization_stage"),
            ),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    names = (
        "protocol.json",
        "source-manifest.json",
        "freeze.json",
        "attempt.json",
        "result.json",
        "source.json",
        "event_before.json",
        "event_raw.json",
        "event_after.json",
        "sample_0.json",
        "sample_1.json",
        "sample_2.json",
        "response.json",
    )
    documents = {name: _read(name) for name in names}
    certificate = prepare_child_sample_certificate(JOINT, repository_root=ROOT)
    source_graph = restore_recorded_graph_state(
        documents["source.json"]["endpoint"]
    ).graph
    events = (
        restore_recorded_graph_state(documents["event_before.json"]).graph,
        restore_recorded_graph_state(documents["event_raw.json"]).graph,
        restore_recorded_graph_state(documents["event_after.json"]["endpoint"]).graph,
    )
    samples = tuple(documents[f"sample_{index}.json"] for index in range(3))
    graphs = tuple(
        restore_recorded_graph_state(record["endpoint"]).graph for record in samples
    )
    response = documents["response.json"]
    assessment = _assess_child_samples(
        certificate,
        states=response["states"],
        pressures=response["pressures"],
        sample_times=response["sample_times"],
    )
    with ZipFile(EVIDENCE / "steps.zip") as archive:
        expected = tuple(f"step-{index:04d}.json" for index in range(416))
        assert tuple(info.filename for info in archive.infolist()) == expected
        steps = tuple(
            decode_exact_tree(json_loads(archive.read(name))) for name in expected
        )
    protected = (
        (source_graph,)
        + events
        + graphs
        + (certificate.source.retention.admitted.graph,)
    )
    signatures = tuple(map(_signature, protected))
    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths)
    yield dict(
        documents=documents,
        certificate=certificate,
        source_graph=source_graph,
        events=events,
        samples=samples,
        graphs=graphs,
        assessment=assessment,
        steps=steps,
        response=response,
    )
    assert tuple(map(_signature, protected)) == signatures
    assert tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths) == receipts
    assert tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file())) == paths


def test_frozen_source_archive_prior_receipts_and_single_attempt(retained):
    records = retained["documents"]
    freeze, attempt, result = (
        records[name] for name in ("freeze.json", "attempt.json", "result.json")
    )
    assert freeze["schema"] == "tnfr.maintained-child-history-freeze.v1"
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert {row["path"] for row in freeze["artifacts"]} == {
        "protocol.json",
        "source-manifest.json",
        "source.zip",
    }
    for row in freeze["artifacts"]:
        _receipt(EVIDENCE, row)
    for row in freeze["prior_artifacts"]:
        _receipt(ROOT, row)
    members = records["source-manifest.json"]["files"]
    expected = {row["path"]: row["sha256"] for row in members}
    assert len(expected) == len(members)
    assert {
        "src/tnfr/research/maintained_child_history_acquisition.py",
        "src/tnfr/research/maintained_child_sample_certificate.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/dynamics/integrators.py",
        "instrument.py",
        "pyproject.toml",
    } <= expected.keys()
    verify_archive_members(EVIDENCE / "source.zip", expected, max_bytes=MAX_BYTES)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {info.filename: info.file_size for info in archive.infolist()} == {
            row["path"]: row["bytes"] for row in members
        }
    assert attempt["schema"] == "tnfr.maintained-child-history-attempt.v1"
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, attempt["freeze"])
    assert result["schema"] == "tnfr.maintained-child-history-result.v1"
    assert result["status"] == "acquired"
    assert result["attempt"]["path"] == "attempt.json"
    _receipt(EVIDENCE, result["attempt"])
    assert {row["path"] for row in result["artifacts"]} == {
        "source.json",
        "event_before.json",
        "event_raw.json",
        "event_after.json",
        "sample_0.json",
        "sample_1.json",
        "sample_2.json",
        "steps.zip",
        "response.json",
    }
    for row in result["artifacts"]:
        _receipt(EVIDENCE, row)


def test_frozen_protocol_selects_the_audited_law_schedule_and_error_budgets():
    protocol = _read("protocol.json")
    assert protocol["schema"] == "tnfr.maintained-child-history-protocol.v1"
    assert protocol["evidence_kind"] == (
        "new_numerical_acquisition_against_existing_exact_flow_certificate"
    )
    assert protocol["child"] == CHILD and protocol["parent"] == 2
    assert protocol["node_order"] == tuple(range(20)) + (CHILD,)
    assert protocol["state_dimension"] == 42 and protocol["original_edges"] == 53
    assert protocol["held"] == ("support", "conductances", "positive capacities")
    assert protocol["forcing"] == protocol["loss"] == 0
    assert protocol["law"] == "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))"
    assert protocol["source_time"] == Q(3670017, 524288)
    elapsed = (Q(3, 128), Q(25, 1024), Q(13, 512))
    assert protocol["elapsed_sample_times"] == elapsed
    assert protocol["sample_times"] == tuple(
        protocol["source_time"] + time for time in elapsed
    )
    assert protocol["event_time"] == protocol["sample_times"][0]
    assert protocol["step"] == STEP
    assert protocol["maximum_steps"] == 416
    assert protocol["segment_steps"] == (384, 16, 16)
    assert protocol["field_evaluations"] == 1664
    assert protocol["integrator"] == "tnfr.dynamics.integrators.rk4_coupled_step"
    assert protocol["field"] == "tnfr.research.local_capacity_history.numerical_field"
    assert protocol["maximum_reserved_evaluations"] == protocol["maximum_events"] == 1
    assert protocol["execution_wall_budget_seconds"] == 600
    assert protocol["wall_budget_scope"] == (
        "cooperative execution checks; source re-admission excluded"
    )
    assert protocol["context_word"] == ("OZ", "THOL", "SHA")
    assert protocol["executed_prefix"] == ("OZ",)
    assert protocol["oz_noise"] is False
    assert protocol["random_seed"] == 17
    for name in (
        "full_state_error_allowance",
        "child_sample_error_allowance",
        "exact_flow_pressure_error_allowance",
    ):
        assert protocol[name] == EPSILON
    assert protocol["represented_point_pressure_error_allowance"] == Q(1, 2**46)
    assert protocol["timestamp_error"] == protocol["reader_roundoff_allowance"] == 0
    assert protocol["minimum_form"] == Q(0.2)
    assert protocol["threshold"] == Q(0.1)
    assert protocol["source_result"]["path"] == (
        "docs/assets/maintained_child_joint_window/result.json"
    )
    _receipt(ROOT, protocol["source_result"])


def test_actual_source_is_the_admitted_sha_point_and_not_an_interval_midpoint(retained):
    certificate, source = retained["certificate"], retained["source_graph"]
    admitted = certificate.source.retention.admitted.graph
    assert _signature(source) == _signature(admitted)
    assert _state(source) == tuple(
        value.lo for value in certificate.source.retention.initial
    )
    assert all(value.lo == value.hi for value in certificate.source.retention.initial)
    assert tuple(source) == certificate.source.retention.prediction.nodes
    assert len(source) == 21 and source.number_of_edges() == 53
    assert not source.nodes[CHILD].get("epi_time_history")
    assert not source.nodes[CHILD].get("glyph_history")
    assert node_debt(source.nodes[CHILD]) == 0
    record = retained["documents"]["source.json"]
    assert _same(record["source_receipts"], certificate.source.receipts)
    assert _same(
        record["timing_admission"], _admit_child_oz_timing_source(certificate.source)
    )
    assert _same(retained["response"]["source_receipts"], certificate.source.receipts)


def test_oz_preserves_all_continuous_coordinates_support_and_storage_arguments(
    retained,
):
    before, raw, after = retained["events"]
    old_state, old_capacity = _state(before), _values(before, ALIAS_VF)
    old_edges = tuple(before.edges(data=True))
    for graph in (raw, after):
        assert tuple(graph) == tuple(before)
        assert _state(graph) == old_state
        assert _values(graph, ALIAS_VF) == old_capacity
        assert tuple(graph.edges(data=True)) == old_edges
        assert tuple((node, tuple(graph[node])) for node in graph) == tuple(
            (node, tuple(before[node])) for node in before
        )
        assert graph.graph["_t"] == before.graph["_t"]
        assert graph.graph.get("hierarchy") == before.graph.get("hierarchy")
        assert tuple(
            (
                node,
                graph.nodes[node].get("parent_node"),
                graph.nodes[node].get("sub_nodes"),
            )
            for node in graph
        ) == tuple(
            (
                node,
                before.nodes[node].get("parent_node"),
                before.nodes[node].get("sub_nodes"),
            )
            for node in before
        )
        # H depends only on these form differences and cosine arguments.
        for left, right in retained["certificate"].source.retention.prediction.edges:
            assert (old_state[left] - old_state[right]) ** 2 == (
                _state(graph)[left] - _state(graph)[right]
            ) ** 2
            assert (
                old_state[21 + left] - old_state[21 + right]
                == _state(graph)[21 + left] - _state(graph)[21 + right]
            )


def test_raw_oz_is_the_binary64_gain_without_neighbor_propagation_then_refresh(
    retained,
):
    before, raw, after = retained["events"]
    incoming, outgoing = _values(before, ALIAS_DNFR), _values(raw, ALIAS_DNFR)
    gain = resolve_runtime_operator_factors(
        before.graph.get("GLYPH_FACTORS"), Glyph.OZ, before.graph
    )["OZ_dnfr_factor"]
    assert outgoing[CHILD_INDEX] == Q(float(incoming[CHILD_INDEX]) * gain)
    assert outgoing[CHILD_INDEX] > incoming[CHILD_INDEX]
    assert all(
        outgoing[i] == value for i, value in enumerate(incoming) if i != CHILD_INDEX
    )
    previous_events = tuple(before.graph.get("_oz_propagation_events", ()))
    events = tuple(raw.graph.get("_oz_propagation_events", ()))
    assert events[:-1] == previous_events and len(events) == len(previous_events) + 1
    assert events[-1]["source"] == CHILD
    assert Q(events[-1]["magnitude"]) == outgoing[CHILD_INDEX] - incoming[CHILD_INDEX]
    assert events[-1]["affected_nodes"] == []
    assert type(events[-1]["affected_count"]) is int
    assert events[-1]["affected_count"] == 0
    assert all(
        raw.nodes[node].get("_oz_propagation")
        == before.nodes[node].get("_oz_propagation")
        for node in before
    )
    assert _values(after, ALIAS_DNFR) == incoming
    assert tuple(raw.nodes[CHILD]["glyph_history"]) == ("OZ",)
    assert tuple(after.nodes[CHILD]["glyph_history"]) == ("OZ",)
    assert node_debt(after.nodes[CHILD]) == 1
    stage = retained["documents"]["event_after.json"]["stage"]
    assert stage["glyph"] == "OZ" and stage["nodes_processed"] == 1
    assert _same(stage, retained["response"]["stage"])


def test_full_rk4_trace_has_exact_clock_carry_stage_arithmetic_and_residuals(retained):
    prediction = retained["certificate"].source.retention.prediction
    start = Q(retained["source_graph"].graph["_t"])
    previous = _state(retained["source_graph"])
    for index, record in enumerate(retained["steps"]):
        assert type(record["index"]) is int and record["index"] == index
        time = start + index * STEP
        assert record["start"] == time and record["end"] == time + STEP
        stages = record["stages"]
        assert len(stages) == 4
        assert tuple(stage["time"] for stage in stages) == (
            time,
            time + STEP / 2,
            time + STEP / 2,
            time + STEP,
        )
        assert stages[0]["state"] == previous
        for stage in stages:
            assert len(stage["state"]) == len(stage["rates"]) == 42
            assert all(
                Q(float(value)) == value for value in stage["state"] + stage["rates"]
            )
            assert stage["rates"] == _independent_rates(stage["state"], prediction)
        for stage_index, fraction in ((1, Q(1, 2)), (2, Q(1, 2)), (3, Q(1))):
            assert stages[stage_index]["state"] == _stage_from_recorded_rate(
                previous, stages[stage_index - 1]["rates"], STEP * fraction
            )
        assert record["endpoint"] == _endpoint_from_recorded_rates(previous, stages)
        residuals = tuple(
            value
            - old
            - STEP
            * sum(
                weight * stage["rates"][i]
                for weight, stage in zip((1, 2, 2, 1), stages, strict=True)
            )
            / 6
            for i, (old, value) in enumerate(
                zip(previous, record["endpoint"], strict=True)
            )
        )
        assert record["represented_quadrature_residuals"] == residuals
        previous = record["endpoint"]
    assert len(retained["steps"]) == 416
    assert _state(retained["events"][0]) == retained["steps"][383]["endpoint"]
    assert _state(retained["events"][2]) == retained["steps"][384]["stages"][0]["state"]
    for index, step_index in enumerate((383, 399, 415)):
        assert (
            retained["samples"][index]["state"]
            == retained["steps"][step_index]["endpoint"]
        )


def test_all_state_and_pressure_errors_are_rebuilt_from_same_original_solution(
    retained,
):
    certificate, assessment = retained["certificate"], retained["assessment"]
    prediction = certificate.source.retention.prediction
    assert assessment.passed
    for index, (sample, box, boundary) in enumerate(
        zip(retained["samples"], certificate.boxes, assessment.boundaries, strict=True)
    ):
        state, pressures = sample["state"], sample["pressures"]
        errors = tuple(
            max(abs(value - bound.lo), abs(value - bound.hi))
            for value, bound in zip(state, box, strict=True)
        )
        assert errors == boundary.state_error_bounds
        assert max(errors) <= EPSILON
        assert errors[CHILD_INDEX] == boundary.child_error_bound
        truth, point = _pressure_bounds(box, prediction), _pressure_bounds(
            tuple(map(I, state)), prediction
        )
        true_errors = tuple(
            max(abs(value - bound.lo), abs(value - bound.hi))
            for value, bound in zip(pressures, truth, strict=True)
        )
        point_errors = tuple(
            max(abs(value - bound.lo), abs(value - bound.hi))
            for value, bound in zip(pressures, point, strict=True)
        )
        assert max(true_errors) <= EPSILON
        assert max(point_errors) <= Q(1, 2**46)
        for actual, independent in zip(
            boundary.pressure_error_bounds, true_errors, strict=True
        ):
            assert abs(actual - independent) < Q(1, 2**85)
        for phases, actual in (
            (box[21:], boundary.exact_principal_margins),
            (tuple(map(I, state[21:])), boundary.represented_principal_margins),
        ):
            principal = tuple(
                pi_interval().lo
                - (phases[right] - phases[left] - 2 * shift * pi_interval()).abs_max
                for (left, right), shift in zip(
                    prediction.edges, prediction.shifts, strict=True
                )
            )
            assert len(principal) == 53 and principal == actual
            assert min(principal) > 0
        assert _same(sample["association"], boundary)
        assert _same(retained["response"]["boundary_associations"][index], boundary)
    assert _same(retained["response"]["assessment"], assessment)


def test_acquired_history_clock_exact_reader_and_real_prefix_without_thol(retained):
    response, certificate = retained["response"], retained["certificate"]
    assert response["schema"] == "tnfr.maintained-child-history-response.v1"
    assert response["status"] == "acquired"
    assert response["numerical_steps"] == 416 and response["field_evaluations"] == 1664
    start = Q(retained["source_graph"].graph["_t"])
    times = tuple(start + elapsed for elapsed in (Q(3, 128), Q(25, 1024), Q(13, 512)))
    assert certificate.sample_times == response["sample_times"] == times
    assert times[1] - times[0] == times[2] - times[1] == SAMPLE_STEP
    for index, (sample, graph) in enumerate(
        zip(retained["samples"], retained["graphs"], strict=True)
    ):
        assert sample["index"] == index and sample["time"] == times[index]
        assert Q(graph.graph["_t"]) == times[index]
        assert _state(graph) == sample["state"] == response["states"][index]
        assert (
            _values(graph, ALIAS_DNFR)
            == sample["pressures"]
            == response["pressures"][index]
        )
        expected = tuple(
            (times[j], response["states"][j][CHILD_INDEX]) for j in range(index + 1)
        )
        history = tuple(
            tuple(map(Q, row)) for row in graph.nodes[CHILD]["epi_time_history"]
        )
        assert history == expected
        assert history[-1] == (Q(graph.graph["_t"]), _state(graph)[CHILD_INDEX])
        assert tuple(graph.nodes[CHILD]["glyph_history"]) == ("OZ",)
        assert node_debt(graph.nodes[CHILD]) == 1
        assert len(graph) == 21 and graph.number_of_edges() == 53
        assert graph.graph.get("hierarchy") == retained["source_graph"].graph.get(
            "hierarchy"
        )
    endpoint = retained["graphs"][-1]
    for node_index, node in enumerate(endpoint):
        assert tuple(
            tuple(map(Q, row))
            for row in tuple(endpoint.nodes[node]["epi_time_history"])[-3:]
        ) == tuple(
            (time, state[node_index])
            for time, state in zip(times, response["states"], strict=True)
        )
    assert _signature(endpoint) == _signature(
        restore_recorded_graph_state(response["endpoint"]).graph
    )
    observation = observe_structural_acceleration(endpoint, CHILD)
    values = tuple(state[CHILD_INDEX] for state in response["states"])
    stencil = (values[0] - 2 * values[1] + values[2]) / SAMPLE_STEP**2
    assert observation.available and observation.current_endpoint_matches_state
    assert Q(observation.value) == stencil
    assert _same(response["observation"], observation)
    assessment = retained["assessment"]
    errors = assessment.child_error_bounds
    assert (
        assessment.reader_budget.acceleration_sample_error_bound
        == (errors[0] + 2 * errors[1] + errors[2]) / SAMPLE_STEP**2
    )
    assert assessment.reader_budget.acceleration_sample_error_bound <= Q(1, 4)
    assert assessment.reader_budget.arithmetic_error_bound == 0
    assert (
        stencil
        >= assessment.reader_budget.represented_acceleration_bounds.lo
        > assessment.threshold
    )
    word = ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    grammar = validate_candidate(endpoint, CHILD, "THOL", sequence_context=word.step(1))
    assert grammar.allowed and _same(response["grammar"], grammar)
    validate_self_organization_strict(endpoint, CHILD, emit_warnings=False)
    assert response["executed_prefix"] == ("OZ",)
    assert response["outstanding_u2_debt"] == 1
    assert response["complete_word_executed"] is False
    assert response["second_birth_executed"] is False
    assert response["physical_observation"] is False
