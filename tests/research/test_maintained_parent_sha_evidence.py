"""Audit the single retained SHA closure without replaying events or flow.

Archive receipts associate bytes, not acquisition chronology. The actual
capacity product, unchanged storage arguments, complete rows and means are
reconstructed independently from primitive retained state. No earlier motion
certificate is transferred across this capacity-changing event.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import inf, nextafter, pi
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.operators import al_sha_stage_proposals, network_stage, word_execution
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.research import local_capacity_history
from tnfr.research import maintained_parent_sha_execution as owner
from tnfr.research import (
    maintained_parent_thol_execution,
    maintained_parent_um_execution,
    maintained_successor_motion_certificate,
)
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.local_capacity_history import TIMES, encode_result
from tnfr.research.maintained_parent_sha import predict_parent_sha
from tnfr.research.maintained_parent_sha_source import load_um_successor
from tnfr.research.maintained_parent_thol import _represented, _signature
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_sha"
UM = ROOT / "docs/assets/maintained_parent_um_corrected"
PARENT, CHILD = 2, "2_sub_0"


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def _same(left, right):
    def canonical(value):
        return json_dumps(encode_exact_tree(encode_result(value)), sort_keys=True)

    return canonical(left) == canonical(right)


def _receipt(base, row):
    name = validate_artifact_path(row["path"])
    path = (base / name).resolve()
    assert path.is_relative_to(base.resolve())
    assert type(row["bytes"]) is int
    assert file_receipt(path, label=name) == row


def _values(graph, aliases):
    return tuple(
        _represented(
            get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value),
            f"retained SHA coordinate at {node!r}",
        )
        for node in graph
    )


def _interval(record):
    assert type(record) is dict and set(record) == {"lo", "hi"}
    assert type(record["lo"]) is Q and type(record["hi"]) is Q
    return I(record["lo"], record["hi"])


def _tight_overlap(record, independent):
    value = _interval(record)
    assert max(value.lo, independent.lo) <= min(value.hi, independent.hi)
    assert value.hi - value.lo < Q(1, 2**85)
    assert independent.hi - independent.lo < Q(1, 2**85)


def _require_rebuilt_assessment(saved, rebuilt):
    if not _same(saved, rebuilt):
        raise ValueError("retained SHA report differs from its primitive rebuild")


@pytest.fixture(scope="module", autouse=True)
def no_producer_event_refresh_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("SHA evidence audit cannot execute a producer, proposal or flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (owner, "evaluate_sha_event"),
            (owner, "execute_network_operator_stage"),
            (owner, "refresh_pressure"),
            (maintained_parent_um_execution, "evaluate_um_event"),
            (maintained_parent_um_execution, "execute_network_operator_stage"),
            (maintained_parent_um_execution, "refresh_pressure"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_successor_motion_certificate, "certify_motion"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "refresh_pressure"),
            (local_capacity_history, "rk4_coupled_step"),
            (al_sha_stage_proposals, "propose_silence_stage"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    names = (
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "freeze.json",
        "attempt.json",
        "raw-after.json",
        "response.json",
        "report.json",
    )
    # Missing evidence fails explicitly; no producer is imported or invoked.
    records = {name: _read(EVIDENCE / name) for name in names}
    protected = tuple(
        sorted(
            path
            for folder in (EVIDENCE, UM)
            for path in folder.iterdir()
            if path.is_file()
        )
    )
    receipts = tuple(file_receipt(path) for path in protected)
    admitted = load_um_successor(UM, repository_root=ROOT)
    response = records["response.json"]
    graphs = tuple(
        restore_recorded_graph_state(response[key]).graph
        for key in ("before", "raw_after", "endpoint")
    )
    signatures = tuple(map(_signature, graphs))
    prediction = predict_parent_sha(admitted)
    assessment = owner.check_sha_endpoint(
        admitted, graphs[1], graphs[2], prediction=prediction, stage=response["stage"]
    )
    yield dict(
        records=records,
        admitted=admitted,
        graphs=graphs,
        prediction=prediction,
        assessment=assessment,
    )
    assert tuple(map(_signature, graphs)) == signatures
    assert receipts == tuple(file_receipt(path) for path in protected)
    assert protected == tuple(
        sorted(
            path
            for folder in (EVIDENCE, UM)
            for path in folder.iterdir()
            if path.is_file()
        )
    )


def test_freeze_archive_prior_evidence_and_single_attempt_remain_associated(retained):
    records = retained["records"]
    frozen, attempt, report = (
        records[name] for name in ("freeze.json", "attempt.json", "report.json")
    )
    assert frozen["schema"] == "tnfr.maintained-parent-sha-freeze.v1"
    assert frozen["evaluation_status_at_freeze"] == "not_evaluated"
    assert {row["path"] for row in frozen["artifacts"]} == {
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "source.zip",
    }
    for row in frozen["artifacts"]:
        _receipt(EVIDENCE, row)
    for row in frozen["prior_artifacts"]:
        _receipt(ROOT, row)
    members = records["source-manifest.json"]["files"]
    expected = {row["path"]: row["sha256"] for row in members}
    assert len(expected) == len(members)
    assert {
        "src/tnfr/research/maintained_parent_sha.py",
        "src/tnfr/research/maintained_parent_sha_source.py",
        "src/tnfr/research/maintained_parent_sha_execution.py",
        "src/tnfr/operators/al_sha_stage_proposals.py",
        "instrument.py",
        "pyproject.toml",
    } <= expected.keys()
    verify_archive_members(EVIDENCE / "source.zip", expected)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {info.filename: info.file_size for info in archive.infolist()} == {
            row["path"]: row["bytes"] for row in members
        }
    assert attempt["schema"] == "tnfr.maintained-parent-sha-attempt.v1"
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert type(attempt["seed"]) is int and attempt["seed"] == 17
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, attempt["freeze"])
    assert report["schema"] == "tnfr.maintained-parent-sha-report.v1"
    for key, name in (
        ("attempt", "attempt.json"),
        ("response", "response.json"),
        ("raw_after", "raw-after.json"),
    ):
        assert report[key]["path"] == name
        _receipt(EVIDENCE, report[key])
    assert not (EVIDENCE / "failure.json").exists()


def test_frozen_protocol_keeps_one_default_event_and_no_trajectory(retained):
    protocol = retained["records"]["protocol.json"]
    assert protocol["schema"] == "tnfr.maintained-parent-sha-protocol.v1"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert protocol["word_context"] == ("VAL", "THOL", "UM", "SHA")
    assert protocol["target_order"] == (PARENT,)
    assert type(protocol["step_index"]) is int and protocol["step_index"] == 3
    for key, expected in (
        ("maximum_events", 1),
        ("maximum_attempts", 1),
        ("maximum_trajectories", 0),
    ):
        assert type(protocol[key]) is int and protocol[key] == expected
    assert protocol["factor"] == Q(1.0 - 1.0 / (4.0 * pi))
    assert protocol["structural_time"] == TIMES[-1]
    assert protocol["capacity_error_allowance"] == Q(1, 2**40)
    assert protocol["pressure_refresh_error_allowance"] == Q(1, 2**46)
    assert protocol["event_metadata_timestamp"] == "2030-01-02T03:04:08+00:00"


def test_before_state_and_predictions_are_rebuilt_from_actual_committed_um(retained):
    source, response = (
        retained["records"][name] for name in ("source.json", "response.json")
    )
    assert source["schema"] == "tnfr.maintained-parent-sha-source.v1"
    assert response["schema"] == "tnfr.maintained-parent-sha-response.v1"
    assert _signature(retained["graphs"][0]) == _signature(retained["admitted"].graph)
    assert _same(source["before"], response["before"])
    assert _same(response["raw_after"], retained["records"]["raw-after.json"])
    for record in (source, response):
        assert _same(record["prediction"], retained["prediction"])
        assert _same(record["retained_source_receipts"], retained["admitted"].receipts)
    assert retained["admitted"].assessment["all_checks_passed"] is True
    assert _read(UM / "report.json")["assessment"]["all_checks_passed"] is False


def test_saved_assessment_matches_independent_primitive_rebuild(retained):
    assessment = retained["assessment"]
    _require_rebuilt_assessment(
        retained["records"]["report.json"]["assessment"], assessment
    )
    assert assessment["schema"] == "tnfr.maintained-parent-sha-assessment.v1"
    assert len(assessment["checks"]) == 19
    assert all(value is True for value in assessment["checks"].values())
    assert assessment["all_checks_passed"] is True
    assert assessment["complete_word_executed"] is True


def test_exact_unchanged_storage_arguments_give_zero_jump_without_interval_cancellation(
    retained,
):
    before, raw, endpoint = retained["graphs"]
    nodes = tuple(before)
    before_x, before_theta = _values(before, ALIAS_EPI), _values(before, ALIAS_THETA)
    before_edges = tuple(before.edges(data=True))
    for graph in (raw, endpoint):
        assert tuple(graph) == nodes
        assert graph.number_of_edges() == 53
        assert tuple(graph.edges(data=True)) == before_edges
        after_x, after_theta = _values(graph, ALIAS_EPI), _values(graph, ALIAS_THETA)
        assert after_x == before_x and after_theta == before_theta
        for left, right in before.edges:
            i, j = nodes.index(left), nodes.index(right)
            weight = _represented(before[left][right]["weight"], "conductance")
            # Form terms cancel as exact represented rationals. Equal real
            # phase differences give identical cosine arguments algebraically.
            assert (
                weight
                * ((after_x[j] - after_x[i]) ** 2 - (before_x[j] - before_x[i]) ** 2)
                / 2
                == 0
            )
            assert after_theta[j] - after_theta[i] == before_theta[j] - before_theta[i]
    assert type(retained["assessment"]["event_storage_jump"]) is Q
    assert retained["assessment"]["event_storage_jump"] == 0


def test_only_parent_capacity_changes_by_correctly_rounded_default_product(retained):
    before, raw, endpoint = retained["graphs"]
    old, actual = _values(before, ALIAS_VF), _values(endpoint, ALIAS_VF)
    parent = tuple(before).index(PARENT)
    factor = Q(1.0 - 1.0 / (4.0 * pi))
    ideal = old[parent] * factor
    represented = Q(float(old[parent]) * float(factor))
    assert actual == _values(raw, ALIAS_VF)
    assert actual[parent] == represented == Q(float(ideal))
    assert all(actual[i] == old[i] for i in range(len(old)) if i != parent)
    lower = Q(nextafter(float(represented), -inf))
    upper = Q(nextafter(float(represented), inf))
    assert (lower + represented) / 2 < ideal < (represented + upper) / 2
    error = abs(represented - ideal)
    assert 0 < error < Q(1, 2**40)
    assert retained["assessment"]["capacity_absolute_error"] == error
    assert 0 < represented < old[parent]


def test_both_rows_pressure_and_mean_metric_use_actual_new_capacity(retained):
    _, raw, graph = retained["graphs"]
    nodes = tuple(graph)
    x, theta, nu = (
        _values(graph, aliases) for aliases in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
    )
    state = retained["assessment"]["actual_state"]
    weights = tuple(Q(graph.degree(node)) / nu[i] for i, node in enumerate(nodes))
    pi_bounds = pi_interval()
    for i, node in enumerate(nodes):
        neighbors = tuple(nodes.index(other) for other in graph[node])
        sine_sum = sum((sin(I(theta[j] - theta[i])) for j in neighbors), I(0))
        form_gradient = sum(
            (
                _represented(graph[node][nodes[j]]["weight"], "conductance")
                * (x[i] - x[j])
                for j in neighbors
            ),
            Q(0),
        )
        pressure = sine_sum / (pi_bounds * len(neighbors))
        form_rate = nu[i] * pressure
        phase_rate = nu[i] * form_gradient / (pi_bounds * len(neighbors))
        _tight_overlap(state["pressure"][i], pressure)
        _tight_overlap(state["form_rate"][i], form_rate)
        _tight_overlap(state["phase_rate"][i], phase_rate)
        assert _interval(state["mean_weights"][i]).contains(weights[i])
        actual_pressure = _values(graph, ALIAS_DNFR)[i]
        assert max(
            abs(actual_pressure - pressure.lo), abs(actual_pressure - pressure.hi)
        ) <= Q(1, 2**46)
    denominator = sum(weights, Q(0))
    for key, values in (("form_mean", x), ("phase_mean", theta)):
        independent = (
            sum(
                (weight * value for weight, value in zip(weights, values, strict=True)),
                Q(0),
            )
            / denominator
        )
        assert _interval(state[key]).contains(independent)
    assert _values(raw, ALIAS_DNFR) == _values(retained["graphs"][0], ALIAS_DNFR)


def test_latency_closes_actual_word_but_does_not_stop_either_continuous_row(retained):
    before, raw, endpoint = retained["graphs"]
    assert tuple(before.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL", "UM")
    for graph in (raw, endpoint):
        parent = graph.nodes[PARENT]
        assert tuple(parent["glyph_history"]) == ("VAL", "THOL", "UM", "SHA")
        assert type(parent["_operator_step"]) is int and parent["_operator_step"] == 4
        assert (
            type(parent["_grammar_u2_debt"]) is int and parent["_grammar_u2_debt"] == 0
        )
        assert parent["latent"] is True
        assert parent["latency_start_time"] == "2030-01-02T03:04:08+00:00"
        assert parent["silence_duration"] == 0.0
        assert _same(
            parent["epi_time_history"], before.nodes[PARENT]["epi_time_history"]
        )
        assert _same(graph.nodes[CHILD], before.nodes[CHILD])
        assert "epi_time_history" not in graph.nodes[CHILD]
        assert not observe_structural_acceleration(graph, CHILD).available
        assert graph.graph["_t"] == before.graph["_t"]
    index = tuple(endpoint).index(PARENT)
    state = retained["assessment"]["actual_state"]
    for key in ("form_rate", "phase_rate"):
        assert not _interval(state[key][index]).contains(0)


def test_capacity_changed_law_does_not_inherit_um_motion_or_new_child_observations(
    retained,
):
    prediction = retained["prediction"]
    actual_capacity = _values(retained["graphs"][2], ALIAS_VF)[prediction.parent_index]
    old_family = retained["admitted"].prediction.chamber.conditional_capacities[
        prediction.parent_index
    ]
    assert actual_capacity < old_family.lo
    assert prediction.form_mean_after != prediction.form_mean_before
    assert prediction.phase_mean_after != prediction.phase_mean_before
    assert (
        prediction.mean_weights_after[prediction.parent_index]
        > prediction.mean_weights_before[prediction.parent_index]
    )
    for key in (
        "um_motion_certificate_transferred",
        "autonomous_occurrence_derived",
        "numerical_u5_observation_available",
        "cross_birth_u6_observation_available",
    ):
        assert retained["assessment"][key] is False


@pytest.mark.parametrize(
    "case", ("boolean_verdict", "invented_storage", "invented_rate")
)
def test_cached_report_forgery_cannot_replace_rebuilt_result(retained, case):
    changed = deepcopy(retained["records"]["report.json"]["assessment"])
    if case == "boolean_verdict":
        changed["all_checks_passed"] = 1
    elif case == "invented_storage":
        changed["event_storage_jump"] = Q(1)
    else:
        changed["actual_state"]["phase_rate"] = ({"lo": Q(0), "hi": Q(0)},) * 21
    with pytest.raises(ValueError, match="primitive rebuild"):
        _require_rebuilt_assessment(changed, retained["assessment"])


@pytest.mark.parametrize("case", ("capacity_boolean", "form_change", "child_baseline"))
def test_invalid_or_changed_endpoint_is_not_rescued_by_the_passing_saved_report(
    retained, case
):
    response = retained["records"]["response.json"]
    raw, endpoint = (
        restore_recorded_graph_state(response[key]).graph
        for key in ("raw_after", "endpoint")
    )
    if case == "capacity_boolean":
        key = next(key for key in ALIAS_VF if key in endpoint.nodes[PARENT])
        endpoint.nodes[PARENT][key] = True
        with pytest.raises((TypeError, ValueError)):
            owner.check_sha_endpoint(
                retained["admitted"], raw, endpoint, stage=response["stage"]
            )
        return
    if case == "form_change":
        set_attr(endpoint.nodes[PARENT], ALIAS_EPI, 0.8)
    else:
        endpoint.nodes[CHILD]["epi_time_history"] = endpoint.nodes[PARENT][
            "epi_time_history"
        ]
    rebuilt = owner.check_sha_endpoint(
        retained["admitted"], raw, endpoint, stage=response["stage"]
    )
    assert rebuilt["all_checks_passed"] is False
    assert rebuilt["complete_word_executed"] is False
    if case == "form_change":
        assert rebuilt["event_storage_jump"] is None
        assert rebuilt["checks"]["storage_arguments_unchanged"] is False
    else:
        assert rebuilt["checks"]["child_temporal_baseline_unavailable"] is False
