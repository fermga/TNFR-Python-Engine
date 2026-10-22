"""Independent read-only audit of the recorded retrospective child prefix.

The retained SHA reader re-admits complete cells without regenerating jets.
This audit reconstructs only the reported prefix from the degree-one child
equations. It never invokes the selection instrument or another trajectory.
"""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import al_sha_stage_proposals, network_stage, word_execution
from tnfr.operators._thol_config import (
    resolve_thol_bifurcation_threshold,
    resolve_thol_min_epi,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.research import (
    local_capacity_history,
    maintained_child_joint_window,
    maintained_parent_sha_execution,
    maintained_parent_thol_execution,
    maintained_parent_um_execution,
    maintained_principal_retention,
    maintained_sha_motion,
    maintained_successor_motion_certificate,
    maintained_temporal_window,
)
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.research.maintained_parent_sha_source import _same
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_sha_retention_evidence import (
    _interval,
    _receipt,
    _receipt_fields,
    load_sha_retention,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_joint_window"
RETENTION = ROOT / "docs/assets/maintained_sha_retention"


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=32 * 1024**2))
    )


def _tight_agreement(actual, independent):
    assert max(actual.lo, independent.lo) <= min(actual.hi, independent.hi)
    assert abs(actual.lo - independent.lo) < Q(1, 2**85)
    assert abs(actual.hi - independent.hi) < Q(1, 2**85)


def _independent_cell(step, prediction, minimum_form, threshold):
    """Reconstruct the child equation directly from its complete tube."""
    parent = prediction.nodes.index(2)
    child = prediction.nodes.index("2_sub_0")
    x, theta, pi = step.tube[:21], step.tube[21:], pi_interval()
    neighbors = [[] for _ in prediction.nodes]
    for (left, right), weight in zip(
        prediction.edges, prediction.conductances, strict=True
    ):
        neighbors[left].append((right, weight))
        neighbors[right].append((left, weight))
    assert len(neighbors[child]) == 1 and neighbors[child][0][0] == parent

    def omega(node):
        row = neighbors[node]
        contrast = sum((weight * (x[node] - x[other]) for other, weight in row), I(0))
        return prediction.capacities[node] * contrast / (pi * len(row))

    edge_index = prediction.edges.index((parent, child))
    delta = theta[child] - theta[parent] - 2 * prediction.shifts[edge_index] * pi
    pressure = -sin(delta) / pi
    mobility = prediction.capacities[child] / pi
    acceleration = -mobility * cos(delta) * (omega(child) - omega(parent))
    direct = tuple(
        pi.lo - (theta[right] - theta[left] - 2 * shift * pi).abs_max
        for (left, right), shift in zip(
            prediction.edges, prediction.shifts, strict=True
        )
    )
    margins = tuple(
        max(value, energy.lo)
        for value, energy in zip(
            direct, prediction.bounds.horizon_remaining_margins, strict=True
        )
    )
    form = x[child]
    statuses = dict(
        form=(
            "admitted"
            if form.lo >= minimum_form
            else "excluded" if form.hi < minimum_form else "unresolved"
        ),
        pressure=(
            "admitted"
            if pressure.lo > 0
            else "excluded" if pressure.hi <= 0 else "unresolved"
        ),
        acceleration=(
            "admitted"
            if acceleration.lo > threshold or acceleration.hi < -threshold
            else "excluded" if acceleration.abs_max <= threshold else "unresolved"
        ),
        identity="admitted" if min(margins) > 0 else "unresolved",
    )
    statuses["joint"] = all(value == "admitted" for value in statuses.values())
    return form, pressure, acceleration, min(margins), statuses


def _check_cell(record, step, prediction, minimum_form, threshold):
    assert type(record["index"]) is int and record["index"] == step.index
    assert exact_record(record["elapsed_start"]) == step.time
    assert exact_record(record["elapsed_end"]) == step.time + step.duration
    form, pressure, acceleration, margin, statuses = _independent_cell(
        step, prediction, minimum_form, threshold
    )
    assert _interval(record["form"]) == form
    _tight_agreement(_interval(record["pressure"]), pressure)
    _tight_agreement(_interval(record["acceleration"]), acceleration)
    assert exact_record(record["minimum_principal_margin"]) == margin
    assert _same(record["conditions"], statuses)
    return statuses["joint"]


@pytest.fixture(scope="module", autouse=True)
def no_new_selection_event_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("evidence audit cannot execute a new observation or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_child_joint_window, "inspect_child_joint_window"),
            (maintained_child_joint_window, "bound_node_history_box"),
            (maintained_temporal_window, "bound_node_history_box"),
            (_validated_taylor, "validated_box_taylor_step"),
            (maintained_sha_motion, "certify_sha_retention"),
            (maintained_sha_motion, "validated_box_taylor_step"),
            (maintained_principal_retention, "certify_retention"),
            (maintained_successor_motion_certificate, "certify_motion"),
            (maintained_parent_sha_execution, "evaluate_sha_event"),
            (maintained_parent_sha_execution, "refresh_pressure"),
            (maintained_parent_um_execution, "evaluate_um_event"),
            (maintained_parent_um_execution, "refresh_pressure"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
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
def evidence():
    before = {
        path.name: file_receipt(path) for path in EVIDENCE.iterdir() if path.is_file()
    }
    data = {
        name: _read(f"{name}.json")
        for name in ("protocol", "freeze", "attempt", "result", "source-manifest")
    }
    retained = load_sha_retention(RETENTION, repository_root=ROOT)
    signature = _signature(retained.admitted.graph)
    data["retained"] = retained
    yield data
    assert _signature(retained.admitted.graph) == signature
    assert before == {
        path.name: file_receipt(path) for path in EVIDENCE.iterdir() if path.is_file()
    }


def test_retrospective_freeze_attempt_archive_and_prior_bytes_are_bound(evidence):
    freeze, attempt, result = (evidence[key] for key in ("freeze", "attempt", "result"))
    assert freeze["schema"] == "tnfr.maintained-child-joint-window-freeze.v1"
    assert freeze["observation_status"] == "not_observed_by_this_instrument"
    assert freeze["evidence_kind"] == (
        "retrospective observation; source trajectory already evaluated"
    )
    assert {_receipt_fields(row) for row in freeze["artifacts"]} == {
        "protocol.json",
        "source-manifest.json",
        "source.zip",
    }
    assert len(freeze["artifacts"]) == 3
    for row in freeze["artifacts"]:
        _receipt(EVIDENCE, row)
    prior = freeze["prior_artifacts"]
    prior_names = {_receipt_fields(row) for row in prior}
    assert len(prior) == len(prior_names)
    assert {
        f"docs/assets/maintained_sha_retention/{row['path']}"
        for row in evidence["retained"].receipts
    } <= prior_names
    for row in prior:
        _receipt(ROOT, row)
    manifest = evidence["source-manifest"]
    assert manifest["schema"] == "tnfr.source-manifest.v1"
    members = manifest["files"]
    assert len(members) == len({_receipt_fields(row) for row in members})
    hashes = {row["path"]: row["sha256"] for row in members}
    assert {
        "src/tnfr/research/maintained_child_joint_window.py",
        "src/tnfr/research/maintained_temporal_window.py",
        "src/tnfr/research/maintained_sha_retention_evidence.py",
        "src/tnfr/operators/_thol_config.py",
        "instrument.py",
        "pyproject.toml",
    } <= hashes.keys()
    verify_archive_members(EVIDENCE / "source.zip", hashes)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert _same(
            {item.filename: item.file_size for item in archive.infolist()},
            {row["path"]: row["bytes"] for row in members},
        )
    assert attempt["schema"] == "tnfr.maintained-child-joint-window-attempt.v1"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert attempt["evidence_kind"] == "retrospective observation"
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, attempt["freeze"])
    assert result["schema"] == "tnfr.maintained-child-joint-window-result.v1"
    assert result["attempt"]["path"] == "attempt.json"
    _receipt(EVIDENCE, result["attempt"])


def test_protocol_preserves_actual_source_law_clock_and_unmodified_policy(evidence):
    protocol, retained = evidence["protocol"], evidence["retained"]
    graph, prediction = retained.admitted.graph, retained.prediction
    report = evidence["result"]["report"]
    assert protocol["schema"] == "tnfr.maintained-child-joint-window-protocol.v1"
    expected = {
        "evidence_kind": "new_retrospective_observation_of_retained_evidence",
        "not_a_prospective_reserved_prediction": True,
        "child": "2_sub_0",
        "parent": 2,
        "node_order": prediction.nodes,
        "state_dimension": 42,
        "original_edges": 53,
        "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
        "held": ("support", "conductances", "positive capacities"),
        "forcing": 0,
        "loss": 0,
        "clock": "original structural clock",
        "source_time": Q(graph.graph["_t"]),
        "elapsed_horizon": Q(1, 2),
        "cell_width": Q(1, 512),
        "maximum_cells": 256,
        "minimum_form": Q(resolve_thol_min_epi(graph.graph)),
        "threshold": Q(resolve_thol_bifurcation_threshold(graph.graph)),
        "sample_acquisition": False,
        "runtime_history_installation": False,
        "maximum_new_trajectories": 0,
        "maximum_events": 0,
        "mesh_refinement": False,
        "parameter_retuning": False,
    }
    for key, value in expected.items():
        assert _same(protocol[key], value), key
    assert protocol["source_result"]["path"] == (
        "docs/assets/maintained_sha_retention/result.json"
    )
    _receipt(ROOT, protocol["source_result"])
    assert _same(report["source_receipts"], retained.receipts)
    for key in ("child", "parent", "source_time", "minimum_form", "threshold"):
        assert _same(report[key], expected[key]), key
    assert retained.summary["status"] == "completed"
    assert retained.summary["full_horizon_retention"] is True
    assert type(report["retained_cells"]) is int
    assert report["retained_cells"] == len(retained.steps) == 256


def test_recorded_prefix_ranges_follow_independent_degree_one_child_equations(evidence):
    retained, report = evidence["retained"], evidence["result"]["report"]
    records = report["inspected_cells"]
    minimum = exact_record(report["minimum_form"])
    threshold = exact_record(report["threshold"])
    outcomes = []
    for index, record in enumerate(records):
        assert record["index"] == index
        outcomes.append(
            _check_cell(
                record, retained.steps[index], retained.prediction, minimum, threshold
            )
        )
    assert len(records) == 13
    assert all(value is False for value in outcomes[:-1])
    assert outcomes[-1] is True
    assert _same(report["selected_cell"], records[-1])
    assert report["status"] == "joint_window_admitted"


def test_selected_whole_cell_has_strict_joint_margins_without_claiming_first_entry(
    evidence,
):
    retained, report = evidence["retained"], evidence["result"]["report"]
    selected = report["selected_cell"]
    index = selected["index"]
    form, pressure, acceleration, margin, conditions = _independent_cell(
        retained.steps[index],
        retained.prediction,
        exact_record(report["minimum_form"]),
        exact_record(report["threshold"]),
    )
    assert index == 12
    assert exact_record(selected["elapsed_start"]) == Q(3, 128)
    assert exact_record(selected["elapsed_end"]) == Q(13, 512)
    assert form.lo > Q(251, 1250)  # 0.2008, above unchanged represented 0.2.
    assert pressure.lo > Q(17, 500)
    assert acceleration.lo > 111
    assert margin > Q(79, 50)
    assert conditions["joint"] is True
    assert "first_actual_entry_time" not in report
    assert "numerical_history" not in report


def test_true_window_does_not_supply_executed_context_or_acquired_history(evidence):
    graph = evidence["retained"].admitted.graph
    report = evidence["result"]["report"]
    child = graph.nodes["2_sub_0"]
    assert child["parent_node"] == 2
    assert child["glyph_history"] == []
    assert "epi_time_history" not in child
    observation = observe_structural_acceleration(graph, "2_sub_0")
    assert observation.history_length == 1 and not observation.available
    grammar = validate_candidate(graph, "2_sub_0", "THOL")
    errors = tuple(item.rule for item in grammar.violations if item.severity == "error")
    assert report["source_grammar_error_rules"] == errors == ("U4b",)
    assert report["retrospective_observation"] is True
    for key in (
        "acquired_history",
        "acquired_u4b_context",
        "executed_event",
        "full_thol_admission_established",
    ):
        assert report[key] is False


@pytest.mark.parametrize(
    "mutation", ("form", "pressure", "acceleration", "margin", "condition")
)
def test_changed_report_ranges_or_cached_verdict_cannot_pass_independent_audit(
    evidence, mutation
):
    retained, report = evidence["retained"], evidence["result"]["report"]
    record = deepcopy(report["selected_cell"])
    if mutation in ("form", "pressure", "acceleration"):
        record[mutation]["lo"] += Q(1, 1024)
    elif mutation == "margin":
        record["minimum_principal_margin"] += Q(1, 1024)
    else:
        record["conditions"]["joint"] = 1
    with pytest.raises(AssertionError):
        _check_cell(
            record,
            retained.steps[record["index"]],
            retained.prediction,
            exact_record(report["minimum_form"]),
            exact_record(report["threshold"]),
        )
