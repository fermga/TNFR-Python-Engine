"""Read-only admission of the separately frozen corrected UM transaction.

The inherited motion has its own retained-step arithmetic audit. This module
checks its source association and actual endpoint inclusion without executing
an event, refreshing pressure or evaluating a trajectory. Archive receipts
associate bytes; they do not independently authenticate acquisition chronology.
"""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_parent_um_execution as owner
from tnfr.research import maintained_successor_motion_certificate
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.local_capacity_history import TIMES, encode_result
from tnfr.research.maintained_parent_thol import _represented, _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_parent_um_work import predict_parent_um_work
from tnfr.research.maintained_successor_motion import predict_successor_motion
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.utils.cache import GRAPH_RUNTIME_CACHE_KEYS
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_um_corrected"
ORIGINAL = ROOT / "docs/assets/maintained_parent_um_execution"
MOTION = ROOT / "docs/assets/successor_nonstationary"
PARENT, CHILD = 2, "2_sub_0"


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def _same(left, right):
    def canonical(value):
        return json_dumps(encode_exact_tree(encode_result(value)), sort_keys=True)

    return canonical(left) == canonical(right)


def _receipt(root, row):
    name = validate_artifact_path(row["path"])
    path = root / name
    assert path.resolve().is_relative_to(root.resolve())
    assert file_receipt(path, label=name) == row


def _values(graph, alias):
    return tuple(
        _represented(
            get_attr(graph.nodes[node], alias, strict=True, conv=lambda value: value),
            f"retained coordinate at {node!r}",
        )
        for node in graph
    )


def _interval(record):
    assert type(record) is dict and set(record) == {"lo", "hi"}
    assert type(record["lo"]) is Q and type(record["hi"]) is Q
    result = I(record["lo"], record["hi"])
    assert result.lo == record["lo"] and result.hi == record["hi"]
    return result


def _overlap_tightly(record, independent):
    value = _interval(record)
    assert max(value.lo, independent.lo) <= min(value.hi, independent.hi)
    assert value.hi - value.lo < Q(1, 2**90)
    assert independent.hi - independent.lo < Q(1, 2**90)


def _require_assessment(saved, rebuilt):
    if not _same(saved, rebuilt):
        raise ValueError("retained assessment differs from its primitive rebuild")


@pytest.fixture(scope="module", autouse=True)
def no_operator_or_trajectory_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("corrected UM evidence audit cannot execute an event or flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (owner, "evaluate_um_event"),
            (owner, "execute_network_operator_stage"),
            (owner, "refresh_pressure"),
            (maintained_successor_motion_certificate, "certify_motion"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "refresh_pressure"),
            (local_capacity_history, "rk4_coupled_step"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
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
    # Missing required evidence fails instead of launching a producer or skip.
    names = (
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "freeze.json",
        "attempt.json",
        "raw-after.json",
        "response.json",
        "report.json",
        "review-protocol.json",
        "review.json",
    )
    records = {name: _read(EVIDENCE / name) for name in names}
    protected = tuple(
        sorted(
            path
            for directory in (EVIDENCE, ORIGINAL)
            for path in directory.rglob("*")
            if path.is_file()
        )
    )
    receipts = tuple(file_receipt(path) for path in protected)
    admitted = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    )
    response = records["response.json"]
    graphs = tuple(
        restore_recorded_graph_state(response[key]).graph
        for key in ("before", "raw_after", "endpoint")
    )
    signatures = tuple(map(_signature, graphs))
    prediction = predict_successor_motion(graphs[0])
    work = predict_parent_um_work(graphs[0])
    assessment = owner.check_um_endpoint(
        *graphs, prediction=prediction, stage=response["stage"]
    )
    yield dict(
        records=records,
        graphs=graphs,
        admitted=admitted,
        prediction=prediction,
        work=work,
        assessment=assessment,
    )
    assert tuple(map(_signature, graphs)) == signatures
    assert protected == tuple(
        sorted(
            path
            for directory in (EVIDENCE, ORIGINAL)
            for path in directory.rglob("*")
            if path.is_file()
        )
    )
    assert receipts == tuple(file_receipt(path) for path in protected)


def test_frozen_archive_and_single_attempt_have_independent_receipts(retained):
    records = retained["records"]
    freeze, attempt, report = (
        records[name] for name in ("freeze.json", "attempt.json", "report.json")
    )
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert freeze["schema"] == "tnfr.maintained-parent-um-corrected-freeze.v1"
    assert attempt["schema"] == "tnfr.maintained-parent-um-corrected-attempt.v1"
    assert report["schema"] == "tnfr.maintained-parent-um-corrected-report.v1"
    assert {row["path"] for row in freeze["artifacts"]} == {
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "source.zip",
    }
    for row in freeze["artifacts"]:
        _receipt(EVIDENCE, row)
    for row in freeze["prior_artifacts"]:
        _receipt(ROOT, row)
    entries = records["source-manifest.json"]["files"]
    expected = {row["path"]: row["sha256"] for row in entries}
    assert len(entries) == len(expected)
    assert {
        "src/tnfr/research/maintained_parent_um_execution.py",
        "src/tnfr/research/maintained_successor_motion.py",
        "src/tnfr/operators/network_stage.py",
        "src/tnfr/operators/_coupling_stage_kernel.py",
        "pyproject.toml",
        "instrument.py",
    } <= expected.keys()
    verify_archive_members(EVIDENCE / "source.zip", expected)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {info.filename: info.file_size for info in archive.infolist()} == {
            row["path"]: row["bytes"] for row in entries
        }
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert type(attempt["seed"]) is int and attempt["seed"] == 17
    _receipt(EVIDENCE, attempt["freeze"])
    assert attempt["freeze"]["path"] == "freeze.json"
    for key, path in (
        ("attempt", "attempt.json"),
        ("response", "response.json"),
        ("raw_after", "raw-after.json"),
    ):
        assert report[key]["path"] == path
        _receipt(EVIDENCE, report[key])
    assert not (EVIDENCE / "failure.json").exists()


def test_new_archive_contains_only_the_documented_guard_source_change(retained):
    old = _read(ORIGINAL / "source-manifest.json")["files"]
    new = retained["records"]["source-manifest.json"]["files"]
    old = {row["path"]: row for row in old if row["path"] != "instrument.py"}
    new = {row["path"]: row for row in new if row["path"] != "instrument.py"}
    assert old.keys() == new.keys()
    changed = {path for path in old if old[path] != new[path]}
    owner_path = "src/tnfr/research/maintained_parent_um_execution.py"
    assert changed == {owner_path}
    correction = _read(ORIGINAL / "correction.json")
    for key in ("failed_attempt", "failure", "diagnosis", "source_archive"):
        _receipt(ORIGINAL, correction[key])
    expected = {row["path"]: row["sha256"] for row in correction["members"]}
    verify_archive_members(ORIGINAL / "correction-source.zip", expected)
    assert new[owner_path]["sha256"] == expected[owner_path]
    assert correction["original_attempt_status"] == "failed_and_rolled_back"
    assert correction["corrected_retained_transaction_executed"] is False
    assert correction["scientific_protocol_changed"] is False
    assert correction["numerical_allowances_changed"] is False
    # The earlier record describes its own historical freeze, not this outcome.
    assert not (ORIGINAL / "response.json").exists()
    assert not (ORIGINAL / "report.json").exists()


def test_scientific_protocol_is_unchanged_and_permits_no_new_motion(retained):
    previous = _read(ORIGINAL / "protocol.json")
    protocol = retained["records"]["protocol.json"]
    assert protocol["schema"] == "tnfr.maintained-parent-um-corrected-protocol.v1"
    for key, value in previous.items():
        if key != "schema":
            assert _same(protocol[key], value), key
    assert protocol["maximum_events"] == protocol["maximum_attempts"] == 1
    assert type(protocol["maximum_trajectories"]) is int
    assert protocol["maximum_trajectories"] == 0
    assert protocol["word_context"] == ("VAL", "THOL", "UM", "SHA")
    assert protocol["structural_time"] == TIMES[-1]
    assert protocol["reset_error_allowance"] == Q(1, 2**40)
    assert protocol["pressure_refresh_error_allowance"] == Q(1, 2**46)


def test_actual_before_state_and_all_predictions_are_rebuilt(retained):
    source = retained["records"]["source.json"]
    response = retained["records"]["response.json"]
    before = retained["graphs"][0]
    old_source = _read(ORIGINAL / "source.json")
    assert source["schema"] == "tnfr.maintained-parent-um-corrected-source.v1"
    assert _signature(before) == _signature(retained["admitted"].graph)
    assert _signature(before) == _signature(
        restore_recorded_graph_state(old_source["before"]).graph
    )
    assert _same(response["before"], source["before"])
    assert _same(response["raw_after"], retained["records"]["raw-after.json"])
    assert _same(source["motion_prediction"], retained["prediction"])
    assert _same(source["work_prediction"], retained["work"])
    assert _same(response["prediction"], retained["prediction"])
    for record in (source, response):
        assert _same(record["retained_source_receipts"], retained["admitted"].receipts)
    assert response["schema"] == "tnfr.maintained-parent-um-response.v1"
    assert {row["path"] for row in source["correction_receipts"]} == {
        f"docs/assets/maintained_parent_um_execution/{name}"
        for name in (
            "freeze.json",
            "protocol.json",
            "source.json",
            "source-manifest.json",
            "source.zip",
            "attempt.json",
            "failure.json",
            "diagnosis.json",
            "correction.json",
            "correction-source.zip",
        )
    }
    for row in source["correction_receipts"]:
        _receipt(ROOT, row)


def test_motion_transfer_matches_complete_archived_family_and_clock(retained):
    source = retained["records"]["source.json"]
    assert {row["path"] for row in source["existing_motion_receipts"]} == {
        f"docs/assets/successor_nonstationary/{name}"
        for name in ("freeze.json", "source.json", "attempt.json", "result.json")
    }
    for row in source["existing_motion_receipts"]:
        _receipt(ROOT, row)
    motion_source, protocol = _read(MOTION / "source.json"), _read(
        MOTION / "protocol.json"
    )
    prediction = retained["prediction"]
    assert _same(motion_source["prediction"], prediction)
    conditional = prediction.conditional
    assert _same(motion_source["initial_box"], conditional.epi + conditional.phases)
    assert protocol["node_order"] == prediction.chamber.reset.nodes
    assert protocol["law"] == "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))"
    assert protocol["prior_clock"] == TIMES[-1]
    assert protocol["horizon"] == Q(1, 2)
    assert protocol["step"] == Q(1, 512)
    assert type(protocol["maximum_steps"]) is int and protocol["maximum_steps"] == 256
    assert type(protocol["order"]) is int and protocol["order"] == 12
    # The separate motion reader rechecks every retained Picard/Taylor row;
    # the result's cached flags are deliberately not used as proof here.


def test_saved_assessment_matches_a_fresh_primitive_endpoint_check(retained):
    rebuilt = retained["assessment"]
    _require_assessment(retained["records"]["review.json"]["assessment"], rebuilt)
    assert rebuilt["all_checks_passed"] is True
    assert all(type(value) is bool and value for value in rebuilt["checks"].values())
    assert rebuilt["schema"] == "tnfr.maintained-parent-um-assessment.v1"


def test_original_negative_assessment_is_preserved_and_its_failures_reproduced(
    retained,
):
    before, raw, endpoint = retained["graphs"]
    original = retained["records"]["report.json"]["assessment"]
    assert original["all_checks_passed"] is False
    assert {key for key, value in original["checks"].items() if value is False} == {
        "old_edge_form_work_zero",
        "declared_graph_context_preserved",
    }
    assert len(original["checks"]) == 27
    assert all(type(value) is bool for value in original["checks"].values())
    nodes, reset = tuple(before), retained["prediction"].chamber.reset
    old_x, actual_x = _values(before, ALIAS_EPI), _values(endpoint, ALIAS_EPI)
    interval_terms = []
    for i, j in reset.old_edges:
        old_weight = _represented(before[nodes[i]][nodes[j]]["weight"], "old weight")
        new_weight = _represented(endpoint[nodes[i]][nodes[j]]["weight"], "held weight")
        old_form = old_weight * (old_x[i] - old_x[j]) ** 2
        new_form = new_weight * (actual_x[i] - actual_x[j]) ** 2
        assert new_form - old_form == 0
        # Reproduce the archived dependency loss: enclosing one product before
        # subtracting the same exact product need not return the point zero.
        interval_terms.append(
            (I(new_weight) * (actual_x[i] - actual_x[j]) ** 2 - old_form) / 2
        )
    assert _same(original["old_edge_form_work"], tuple(interval_terms))
    assert all(value.lo <= 0 <= value.hi for value in interval_terms)
    assert any(value.lo < value.hi for value in interval_terms)
    excluded = GRAPH_RUNTIME_CACHE_KEYS | {
        "_dnfrmax",
        "_dnfrmax_node",
        "_vfmax",
        "_vfmax_node",
        "_edge_version",
        network_stage.STAGE_SCHEDULE_KEY,
        network_stage.STAGE_CONTRACT_KEY,
    }
    old_context = {
        key: value for key, value in before.graph.items() if key not in excluded
    }
    for graph in (raw, endpoint):
        new_context = {
            key: value for key, value in graph.graph.items() if key not in excluded
        }
        assert "_dnfr_prep_dirty" not in old_context
        assert new_context.pop("_dnfr_prep_dirty") is True
        assert owner._same(new_context, old_context)


def test_separate_review_preserves_original_bytes_and_unaffected_conclusions(retained):
    records = retained["records"]
    review, protocol = records["review.json"], records["review-protocol.json"]
    assert review["schema"] == "tnfr.maintained-parent-um-assessment-review.v1"
    assert (
        protocol["schema"] == "tnfr.maintained-parent-um-assessment-review-protocol.v1"
    )
    for row in protocol["inputs"]:
        _receipt(EVIDENCE, row)
    for key, path in (
        ("protocol", "review-protocol.json"),
        ("original_report", "report.json"),
        ("response", "response.json"),
    ):
        assert review[key]["path"] == path
        _receipt(EVIDENCE, review[key])
    _receipt(EVIDENCE, protocol["source_archive"])
    assert protocol["source_archive"]["path"] == "review-source.zip"
    members = {row["path"]: row["sha256"] for row in protocol["members"]}
    assert set(members) == {
        "src/tnfr/research/maintained_parent_um_execution.py",
        "tests/research/test_maintained_parent_um_execution.py",
        "review.py",
    }
    verify_archive_members(EVIDENCE / "review-source.zip", members)
    for key in ("maximum_events", "maximum_trajectories"):
        assert type(protocol[key]) is int and protocol[key] == 0
    for key in ("new_events_executed", "new_trajectories_executed"):
        assert type(review[key]) is int and review[key] == 0
    assert review["original_report_preserved"] is True
    original, corrected = records["report.json"]["assessment"], review["assessment"]
    assert set(original) == set(corrected)
    permitted = {
        "old_edge_form_work",
        "actual_total_work",
        "checks",
        "all_checks_passed",
    }
    assert all(
        _same(value, corrected[key])
        for key, value in original.items()
        if key not in permitted
    )
    assert set(original["checks"]) == set(corrected["checks"])
    assert {
        key
        for key in original["checks"]
        if not _same(original["checks"][key], corrected["checks"][key])
    } == {
        "old_edge_form_work_zero",
        "declared_graph_context_preserved",
    }


def test_actual_endpoint_enters_every_state_and_held_parameter_interval(retained):
    before, raw, endpoint = retained["graphs"]
    prediction = retained["prediction"]
    chamber, conditional = prediction.chamber, prediction.conditional
    reset, nodes = chamber.reset, chamber.reset.nodes
    assert tuple(before) == tuple(raw) == tuple(endpoint) == nodes
    assert _values(endpoint, ALIAS_EPI) == _values(raw, ALIAS_EPI) == reset.epi
    epsilon = Q(1, 2**40)
    for alias, bounds, ideals, changed in (
        (
            ALIAS_THETA,
            conditional.phases,
            reset.phase_after,
            reset.phase_changed_indices,
        ),
        (
            ALIAS_VF,
            conditional.capacities,
            reset.capacity_after,
            reset.capacity_changed_indices,
        ),
    ):
        values = _values(endpoint, alias)
        assert values == _values(raw, alias)
        assert all(
            I(value).subset_of(bound)
            for value, bound in zip(values, bounds, strict=True)
        )
        assert all(
            max(abs(values[i] - ideals[i].lo), abs(values[i] - ideals[i].hi)) <= epsilon
            for i in changed
        )
        assert all(
            values[i] == _values(before, alias)[i]
            for i in range(21)
            if i not in changed
        )
    assert _values(endpoint, ALIAS_THETA)[10] < 0
    assert _values(endpoint, ALIAS_THETA)[11] < 0
    rank = {node: index for index, node in enumerate(nodes)}
    actual_edges = tuple(
        sorted(tuple(sorted((rank[a], rank[b]))) for a, b in endpoint.edges)
    )
    assert actual_edges == chamber.edges
    assert (
        len(reset.old_edges) == 45
        and len(reset.new_edges) == 8
        and len(actual_edges) == 53
    )
    for edge, bound in zip(chamber.edges, conditional.conductances, strict=True):
        a, b = (nodes[i] for i in edge)
        weight = _represented(endpoint[a][b]["weight"], "conductance")
        assert I(weight).subset_of(bound)
        if edge in reset.old_edges:
            assert _same(endpoint[a][b], before[a][b])
        else:
            index = reset.new_edges.index(edge)
            ideal = reset.new_conductances[index]
            assert max(abs(weight - ideal.lo), abs(weight - ideal.hi)) <= epsilon


def test_all_fifty_three_original_branch_labels_are_retained(retained):
    endpoint = retained["graphs"][2]
    chamber = retained["prediction"].chamber
    theta, pi = _values(endpoint, ALIAS_THETA), pi_interval()
    assert len(chamber.edges) == len(chamber.ideal.shifts) == 53
    assert chamber.ideal.shifts == chamber.conditional.shifts
    assert (
        tuple(retained["assessment"]["actual_chamber"]["shifts"])
        == chamber.ideal.shifts
    )
    assert all(
        (I(theta[j]) - theta[i] - 2 * shift * pi).abs_max < pi.lo
        for (i, j), shift in zip(chamber.edges, chamber.ideal.shifts, strict=True)
    )


def test_actual_both_rows_and_means_follow_endpoint_not_stored_relief(retained):
    before, raw, endpoint = retained["graphs"]
    x, theta, nu = (
        _values(endpoint, alias) for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
    )
    nodes = tuple(endpoint)
    index = {node: i for i, node in enumerate(nodes)}
    state = retained["assessment"]["actual_state"]
    pi, rho = pi_interval(), []
    refreshed, raw_pressure = _values(endpoint, ALIAS_DNFR), _values(raw, ALIAS_DNFR)
    assert refreshed != raw_pressure
    for i, node in enumerate(nodes):
        neighbors = tuple(sorted(endpoint[node], key=index.__getitem__))
        degree = len(neighbors)
        assert state["degrees"][i] == degree > 0
        rho.append(Q(degree) / nu[i])
        sine_sum = sum((sin(I(theta[index[j]]) - theta[i]) for j in neighbors), I(0))
        pressure = sine_sum / (pi * degree)
        gradient = sum(
            _represented(endpoint[node][j]["weight"], "conductance")
            * (x[i] - x[index[j]])
            for j in neighbors
        )
        _overlap_tightly(state["pressure"][i], pressure)
        _overlap_tightly(state["form_rate"][i], nu[i] * pressure)
        _overlap_tightly(state["phase_rate"][i], I(nu[i] * gradient) / (pi * degree))
        assert max(
            abs(refreshed[i] - pressure.lo), abs(refreshed[i] - pressure.hi)
        ) <= Q(1, 2**46)
        _overlap_tightly(state["mean_weights"][i], I(rho[-1]))
    for key, values, weights in (
        ("form_mean", x, rho),
        ("phase_mean", theta, rho),
        ("old_marked_form_mean", x[:20], rho[:20]),
        ("old_marked_phase_mean", theta[:20], rho[:20]),
    ):
        exact = sum(
            value * weight for value, weight in zip(values, weights, strict=True)
        ) / sum(weights)
        _overlap_tightly(state[key], I(exact))
    assert _values(before, ALIAS_DNFR)[20] == raw_pressure[20]


def test_actual_event_work_keeps_old_phase_and_both_new_edge_costs(retained):
    before, _, endpoint = retained["graphs"]
    x, theta = _values(endpoint, ALIAS_EPI), _values(endpoint, ALIAS_THETA)
    old_theta, nodes = _values(before, ALIAS_THETA), tuple(endpoint)
    reset = retained["prediction"].chamber.reset
    old_phase = sum(
        (
            cos(I(old_theta[j]) - old_theta[i]) - cos(I(theta[j]) - theta[i])
            for i, j in reset.old_edges
        ),
        I(0),
    )
    new_form = sum(
        (
            I(_represented(endpoint[nodes[i]][nodes[j]]["weight"], "conductance"))
            * (x[j] - x[i]) ** 2
            / 2
            for i, j in reset.new_edges
        ),
        I(0),
    )
    new_phase = sum((1 - cos(I(theta[j]) - theta[i]) for i, j in reset.new_edges), I(0))
    assessment = retained["assessment"]
    for key, value in (
        ("old_phase_total", old_phase),
        ("new_form_total", new_form),
        ("new_phase_total", new_phase),
    ):
        _overlap_tightly(assessment["actual_work"][key], value)
    total = old_phase + new_form + new_phase
    _overlap_tightly(assessment["actual_total_work"], total)
    assert Q(318, 100) < total.lo <= total.hi < Q(319, 100)
    assert all(
        _interval(value).lo == _interval(value).hi == 0
        for value in assessment["old_edge_form_work"]
    )
    assert _interval(assessment["actual_old_radius_squared"]).lo > Q(1, 256)


def test_committed_prefix_and_child_availability_are_not_fabricated(retained):
    before, raw, endpoint = retained["graphs"]
    assert _signature(endpoint) != _signature(before)
    assert _signature(endpoint) != _signature(raw)
    assert before.degree(CHILD) == 0 and endpoint.degree(CHILD) == 1
    assert tuple(before.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL")
    assert tuple(endpoint.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL", "UM")
    assert type(endpoint.nodes[PARENT]["_operator_step"]) is int
    assert endpoint.nodes[PARENT]["_operator_step"] == 3
    assert node_debt(endpoint.nodes[PARENT]) == 0
    assert owner._same(
        before.nodes[PARENT]["epi_time_history"],
        endpoint.nodes[PARENT]["epi_time_history"],
    )
    assert owner._same(before.graph["hierarchy"], endpoint.graph["hierarchy"])
    assert _represented(endpoint.graph["_t"], "structural clock") == TIMES[-1]
    assert endpoint.nodes[CHILD]["glyph_history"] == []
    assert "epi_time_history" not in endpoint.nodes[CHILD]
    assert not observe_structural_acceleration(endpoint, CHILD).available
    for key in (
        "complete_word_executed",
        "autonomous_occurrence_derived",
        "numerical_u5_observation_available",
        "cross_birth_u6_observation_available",
    ):
        assert retained["assessment"][key] is False


@pytest.mark.parametrize("mutation", ("flag_type", "work", "row"))
def test_cached_assessment_cannot_substitute_for_primitive_rebuild(retained, mutation):
    altered = deepcopy(retained["records"]["review.json"]["assessment"])
    if mutation == "flag_type":
        altered["all_checks_passed"] = 1
    elif mutation == "work":
        altered["actual_total_work"]["lo"] += Q(1, 100)
    else:
        altered["actual_state"]["phase_rate"][20]["hi"] += 1
    with pytest.raises(ValueError, match="primitive rebuild"):
        _require_assessment(altered, retained["assessment"])


def test_uncommitted_raw_record_cannot_replace_the_accepted_endpoint(retained):
    before, raw, _ = retained["graphs"]
    result = owner.check_um_endpoint(
        before,
        raw,
        raw,
        prediction=retained["prediction"],
        stage=retained["records"]["response.json"]["stage"],
    )
    assert result["all_checks_passed"] is False
    assert result["checks"]["final_schedule_metadata"] is False
    assert result["checks"]["refreshed_pressure_independently_bounded"] is False


def test_modified_held_capacity_cannot_borrow_the_motion_certificate(retained):
    before, raw, endpoint = deepcopy(retained["graphs"])
    changed = float(_values(endpoint, ALIAS_VF)[PARENT] + Q(1, 2**30))
    for graph in (raw, endpoint):
        set_attr(graph.nodes[PARENT], ALIAS_VF, changed)
    result = owner.check_um_endpoint(
        before,
        raw,
        endpoint,
        prediction=retained["prediction"],
        stage=retained["records"]["response.json"]["stage"],
    )
    assert result["all_checks_passed"] is False
    assert result["checks"]["changed_outputs_within_original_allowance"] is False
    assert (
        result["checks"]["conditional_full_state_and_held_parameters_inclusion"]
        is False
    )
