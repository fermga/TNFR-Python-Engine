"""Re-admit the committed UM source without any event or trajectory replay."""

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.operators import (
    _coupling_stage_kernel,
    network_stage,
    preconditions,
    word_execution,
)
from tnfr.research import local_capacity_history
from tnfr.research import maintained_parent_sha_source as owner
from tnfr.research import (
    maintained_parent_thol_execution,
    maintained_parent_um_execution,
    maintained_successor_motion_certificate,
)
from tnfr.research.artifact_io import decode_exact_tree, encode_exact_tree, file_receipt
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_um_corrected"


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


@pytest.fixture(scope="module", autouse=True)
def no_producers_or_sha_eligibility():
    def forbidden(*args, **kwargs):
        pytest.fail("UM source reader cannot execute or decide SHA eligibility")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_parent_um_execution, "evaluate_um_event"),
            (maintained_parent_um_execution, "execute_network_operator_stage"),
            (maintained_parent_um_execution, "refresh_pressure"),
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
            (preconditions, "validate_silence"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    admission = owner.load_um_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(admission.graph)
    thol = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    )
    yield (
        admission,
        thol,
        _read(EVIDENCE / "response.json"),
        _read(EVIDENCE / "review.json")["assessment"],
    )
    assert _signature(admission.graph) == signature
    assert tuple(file_receipt(path) for path in paths) == receipts
    assert tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file())) == paths


def test_actual_committed_source_preserves_prefix_and_child_history_boundary(retained):
    admitted, thol, _, _ = retained
    graph = admitted.graph
    assert tuple(graph) == tuple(thol.graph)
    assert graph.number_of_nodes() == 21 and graph.number_of_edges() == 53
    assert tuple(graph.nodes[2]["glyph_history"]) == ("VAL", "THOL", "UM")
    assert graph.nodes[2]["_grammar_u2_debt"] == 0
    assert graph.nodes["2_sub_0"]["glyph_history"] == []
    assert "epi_time_history" not in graph.nodes["2_sub_0"]
    assert _signature(admitted.before) == _signature(thol.graph)
    assert all(value is True for value in admitted.assessment["checks"].values())
    assert len(admitted.assessment["checks"]) == 27
    assert admitted.assessment["complete_word_executed"] is False
    assert admitted.assessment["autonomous_occurrence_derived"] is False
    assert {row["path"] for row in admitted.receipts} == {
        path.name for path in EVIDENCE.iterdir() if path.is_file()
    }


def test_live_validation_rebuilds_cached_derived_fields_and_returns_fresh_graph(
    retained,
):
    admitted, _, _, _ = retained
    changed = replace(
        admitted, prediction=object(), assessment={"passing": False}, sequence=None
    )
    fresh = owner.validate_live_um_successor(changed)
    assert fresh.graph is not admitted.graph
    assert _signature(fresh.graph) == _signature(admitted.graph)
    assert owner._same(fresh.assessment, admitted.assessment)
    assert fresh.prediction == admitted.prediction
    assert fresh.sequence is not None


@pytest.mark.parametrize("case", ("history", "phase"))
def test_live_mutable_endpoint_must_still_equal_actual_committed_source(retained, case):
    admitted, _, _, _ = retained
    graph = restore_recorded_graph_state(record_graph_state(admitted.graph)).graph
    if case == "history":
        graph.nodes[2]["glyph_history"].append("SHA")
    else:
        set_attr(graph.nodes[2], ALIAS_THETA, 0.75)
    with pytest.raises(ValueError, match="live UM endpoint differs"):
        owner.validate_live_um_successor(replace(admitted, graph=graph))


def test_unassociated_pure_admission_does_not_claim_file_backed_live_validation(
    retained,
):
    _, thol, response, review = retained
    admitted = owner.admit_um_successor(
        response, thol_source=thol, review_assessment=review
    )
    with pytest.raises(ValueError, match="file-backed"):
        owner.validate_live_um_successor(admitted)


@pytest.mark.parametrize("case", ("history", "seed_type", "zero_capacity"))
def test_before_state_cannot_substitute_for_independently_admitted_history(
    retained, case
):
    _, thol, response, review = retained
    changed = deepcopy(response)
    graph = restore_recorded_graph_state(changed["before"]).graph
    if case == "history":
        graph.nodes[2]["epi_time_history"] = ()
    elif case == "seed_type":
        graph.graph["RANDOM_SEED"] = 17.0
    else:
        set_attr(graph.nodes[2], ALIAS_VF, 0.0)
    changed["before"] = record_graph_state(graph)
    with pytest.raises(ValueError, match="independently admitted THOL history"):
        owner.admit_um_successor(changed, thol_source=thol, review_assessment=review)


@pytest.mark.parametrize(
    "case", ("form", "false_capacity", "child_history", "scalar_debt")
)
def test_forged_passing_review_cannot_replace_endpoint_primitives(retained, case):
    _, thol, response, review = retained
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
        owner.admit_um_successor(changed, thol_source=thol, review_assessment=review)


def test_boolean_cached_review_value_cannot_match_integer_by_python_equality(retained):
    _, thol, response, review = retained
    changed = deepcopy(review)
    changed["all_checks_passed"] = 1
    assert changed["all_checks_passed"] == review["all_checks_passed"]
    with pytest.raises(ValueError, match="review differs"):
        owner.admit_um_successor(response, thol_source=thol, review_assessment=changed)


def test_historical_negative_assessment_is_not_accepted_as_the_corrected_review(
    retained,
):
    _, thol, response, _ = retained
    original = _read(EVIDENCE / "report.json")["assessment"]
    with pytest.raises(ValueError, match="review differs"):
        owner.admit_um_successor(response, thol_source=thol, review_assessment=original)


@pytest.mark.parametrize(
    "name", ("response.json", "review.json", "review-source.zip", "review.py")
)
def test_artifact_tampering_rejects_without_running_a_producer(tmp_path, name):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / name
    if name == "review.json":
        changed = _read(path)
        changed["assessment"]["all_checks_passed"] = 1
        path.write_text(json_dumps(encode_exact_tree(changed)), encoding="utf-8")
    else:
        with path.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError):
        owner.load_um_successor(copied, repository_root=ROOT)


def test_frozen_artifact_path_cannot_escape_declared_root(tmp_path):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / "freeze.json"
    changed = _read(path)
    changed["prior_artifacts"][0]["path"] = "../outside.json"
    path.write_text(json_dumps(encode_exact_tree(changed)), encoding="utf-8")
    with pytest.raises(ValueError):
        owner.load_um_successor(copied, repository_root=ROOT)
