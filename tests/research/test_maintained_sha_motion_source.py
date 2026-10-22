"""Read the actual SHA endpoint without replaying a producer or continuation."""

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.operators import al_sha_stage_proposals, network_stage, word_execution
from tnfr.research import (
    local_capacity_history,
    maintained_parent_sha_execution,
    maintained_parent_thol_execution,
    maintained_parent_um_execution,
)
from tnfr.research import maintained_sha_motion_source as owner
from tnfr.research import maintained_successor_motion_certificate
from tnfr.research.artifact_io import decode_exact_tree, encode_exact_tree, file_receipt
from tnfr.research.maintained_parent_sha_source import load_um_successor
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_sha"


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


@pytest.fixture(scope="module", autouse=True)
def no_producer_event_refresh_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("SHA source reader cannot execute a producer, event or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_parent_sha_execution, "evaluate_sha_event"),
            (maintained_parent_sha_execution, "execute_network_operator_stage"),
            (maintained_parent_sha_execution, "refresh_pressure"),
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
    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    admitted = owner.load_sha_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(admitted.graph)
    um = load_um_successor(
        ROOT / "docs/assets/maintained_parent_um_corrected", repository_root=ROOT
    )
    yield (
        admitted,
        um,
        _read(EVIDENCE / "response.json"),
        _read(EVIDENCE / "report.json")["assessment"],
    )
    assert _signature(admitted.graph) == signature
    assert tuple(file_receipt(path) for path in paths) == receipts
    assert tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file())) == paths


def test_committed_complete_word_preserves_child_temporal_boundary(retained):
    admitted, um, _, _ = retained
    graph = admitted.graph
    assert graph.number_of_nodes() == 21 and graph.number_of_edges() == 53
    assert tuple(graph) == tuple(um.graph)
    assert tuple(graph.nodes[2]["glyph_history"]) == ("VAL", "THOL", "UM", "SHA")
    assert graph.nodes[2]["_grammar_u2_debt"] == 0
    assert graph.nodes[2]["latent"] is True
    assert graph.nodes["2_sub_0"]["glyph_history"] == []
    assert "epi_time_history" not in graph.nodes["2_sub_0"]
    assert _signature(admitted.before) == _signature(um.graph)
    assert len(admitted.assessment["checks"]) == 19
    assert all(value is True for value in admitted.assessment["checks"].values())
    assert admitted.assessment["complete_word_executed"] is True
    assert admitted.assessment["um_motion_certificate_transferred"] is False
    assert admitted.assessment["autonomous_occurrence_derived"] is False
    assert (
        admitted.prediction.represented_parent_capacity
        < admitted.prediction.capacity_before[2]
    )
    assert {row["path"] for row in admitted.receipts} == {
        path.name for path in EVIDENCE.iterdir() if path.is_file()
    }


def test_live_validation_rebuilds_cached_fields_in_a_fresh_graph(retained):
    admitted, _, _, _ = retained
    changed = replace(
        admitted,
        prediction=object(),
        assessment={"all_checks_passed": False},
        sequence=None,
        receipts=(),
    )
    fresh = owner.validate_live_sha_successor(changed)
    assert fresh.graph is not admitted.graph
    assert _signature(fresh.graph) == _signature(admitted.graph)
    assert owner._same(fresh.assessment, admitted.assessment)
    assert fresh.prediction == admitted.prediction
    assert fresh.sequence is not None and fresh.receipts


def test_changed_live_endpoint_cannot_use_cached_passing_assessment(retained):
    admitted, _, _, _ = retained
    graph = restore_recorded_graph_state(record_graph_state(admitted.graph)).graph
    set_attr(graph.nodes[2], ALIAS_THETA, 0.75)
    with pytest.raises(ValueError, match="live SHA endpoint differs"):
        owner.validate_live_sha_successor(replace(admitted, graph=graph))


def test_unassociated_admission_cannot_provide_file_backed_live_validation(retained):
    admitted, _, _, _ = retained
    with pytest.raises(ValueError, match="file-backed"):
        owner.validate_live_sha_successor(replace(admitted, evidence_dir=None))


@pytest.mark.parametrize("case", ("history", "seed_type", "zero_capacity"))
def test_before_state_must_match_independently_admitted_um(retained, case):
    _, um, response, report = retained
    changed = deepcopy(response)
    graph = restore_recorded_graph_state(changed["before"]).graph
    if case == "history":
        graph.nodes[2]["epi_time_history"] = ()
    elif case == "seed_type":
        graph.graph["RANDOM_SEED"] = 17.0
    else:
        set_attr(graph.nodes[2], ALIAS_VF, 0.0)
    changed["before"] = record_graph_state(graph)
    with pytest.raises(ValueError, match="independently admitted UM endpoint"):
        owner.admit_sha_successor(changed, um_source=um, retained_assessment=report)


@pytest.mark.parametrize(
    "case", ("form", "false_capacity", "child_history", "scalar_debt")
)
def test_cached_passing_report_cannot_replace_endpoint_primitives(retained, case):
    _, um, response, report = retained
    changed = deepcopy(response)
    graph = restore_recorded_graph_state(changed["endpoint"]).graph
    if case == "form":
        set_attr(graph.nodes[4], ALIAS_EPI, 0.8)
    elif case == "false_capacity":
        set_attr(graph.nodes[2], ALIAS_VF, False)
    elif case == "child_history":
        graph.nodes["2_sub_0"]["epi_time_history"] = graph.nodes[2]["epi_time_history"]
    else:
        graph.nodes[2]["_grammar_u2_debt"] = False
    changed["endpoint"] = record_graph_state(graph)
    with pytest.raises((ValueError, TypeError)):
        owner.admit_sha_successor(changed, um_source=um, retained_assessment=report)


def test_python_boolean_integer_equality_does_not_admit_cached_assessment(retained):
    _, um, response, report = retained
    changed = deepcopy(report)
    changed["all_checks_passed"] = 1
    assert changed["all_checks_passed"] == report["all_checks_passed"]
    with pytest.raises(ValueError, match="assessment differs"):
        owner.admit_sha_successor(response, um_source=um, retained_assessment=changed)


@pytest.mark.parametrize("key", ("maximum_events", "seed", "factor", "structural_time"))
def test_protocol_rejects_type_substitution_and_changed_clock(retained, key):
    admitted, _, _, _ = retained
    protocol = _read(EVIDENCE / "protocol.json")
    protocol[key] = {
        "maximum_events": True,
        "seed": 17.0,
        "factor": True,
        "structural_time": protocol["structural_time"] + 1,
    }[key]
    with pytest.raises(ValueError, match="protocol differs"):
        owner._validate_protocol(protocol, admitted.prediction)


@pytest.mark.parametrize("name", ("response.json", "source.zip", "raw-after.json"))
def test_artifact_receipt_tampering_rejects_without_replay(tmp_path, name):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    with (copied / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        owner.load_sha_successor(copied, repository_root=ROOT)


def test_unreceipted_report_still_requires_primitive_rebuild(tmp_path):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / "report.json"
    changed = _read(path)
    changed["assessment"]["all_checks_passed"] = 1
    path.write_text(json_dumps(encode_exact_tree(changed)), encoding="utf-8")
    with pytest.raises(ValueError, match="assessment differs"):
        owner.load_sha_successor(copied, repository_root=ROOT)


def test_prior_artifact_path_cannot_escape_declared_root(tmp_path):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / "freeze.json"
    changed = _read(path)
    changed["prior_artifacts"][0]["path"] = "../outside.json"
    path.write_text(json_dumps(encode_exact_tree(changed)), encoding="utf-8")
    with pytest.raises(ValueError):
        owner.load_sha_successor(copied, repository_root=ROOT)
