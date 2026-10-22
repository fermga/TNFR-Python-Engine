"""Causal C8 checks of candidate evidence versus explicit public dispatch."""

import json
from fractions import Fraction

import pytest

from benchmarks.thol_eligibility_dispatch import run_study
from benchmarks.thol_pressure_feedback import _payload


@pytest.fixture(scope="module")
def study():
    return run_study()


def test_same_untuned_physical_preparation_feeds_both_declared_policies(study):
    preparation = study["preparation"]
    assert preparation["initial"]["epi"] == (2.0, 0.5) * 4
    assert preparation["initial"]["capacity"] == (1.0,) * 8
    assert preparation["physical_steps"] == (0.25, 0.25)
    assert preparation["default_birth_threshold"] == 0.1
    assert preparation["segment_methods"] == ("euler", "euler")
    assert preparation["clipping_applied"] == (False, False)
    assert preparation["checkerboard_reference"]["exact_model_threshold_margin"] > 0
    for branch in (study["configured_policy"], study["enabled_gate_counterfactual"]):
        assert branch["before"] == preparation["before_birth"]
        assert branch["observer_state_projection_unchanged"]
        assert branch["observer_node_attributes_unchanged"]
    assert study["configured_policy"]["policy_change"] is None
    assert study["enabled_gate_counterfactual"]["policy_change"] is not None


def test_all_eight_cross_threshold_but_only_actual_mark_passes_grammar(study):
    branch = study["configured_policy"]
    observation = branch["eligibility"]
    candidates = observation["candidates"]
    assert tuple(row["node"] for row in candidates) == tuple(range(8))
    assert all(row["threshold_crossed"] and row["birth_proposed"] for row in candidates)
    assert all(row["proposal_valid"] for row in candidates)
    assert all(row["application_preconditions_passed"] for row in candidates)
    assert all(not row["optional_gate_enabled"] for row in candidates)
    assert tuple(row["node"] for row in candidates if row["grammar_allowed"]) == (0,)
    assert tuple(row["node"] for row in candidates if row["eligible"]) == (0,)
    assert observation["eligible_nodes"] == (0,)
    assert observation["joint_stage_viable"] is True
    assert observation["joint_error"] is None
    assert branch["before"]["glyph_history"][0] == ("IL", "OZ")
    assert all(not branch["before"]["glyph_history"][n] for n in range(1, 8))


def test_dispatch_rederives_evidence_and_commits_one_isolated_child(study):
    branch = study["configured_policy"]
    assert branch["dispatch_eligibility"] == branch["eligibility"]
    assert branch["stage_result"]["nodes_processed"] == 1
    assert branch["stage_result"]["schedule"] == "two_phase_jacobi"
    assert len(branch["parent_children"]) == 1
    parent, child = branch["parent_children"][0]
    assert parent == 0
    assert set(branch["after"]["nodes"]) - set(branch["before"]["nodes"]) == {child}
    assert branch["after"]["children"][parent] == (child,)
    assert branch["after"]["hierarchy"][parent] == [child]
    assert branch["children"][0]["degree"] == 0
    assert branch["children"][0]["node_data"]["parent_node"] == parent
    assert branch["after"]["glyph_history"][0] == ("IL", "OZ", "THOL")
    assert all(branch["after"]["glyph_history"][n] == () for n in range(1, 8))
    assert branch["after"]["edges"] == branch["before"]["edges"]
    assert branch["after"]["time"] == branch["before"]["time"] == 0.5
    for channel in ("epi", "capacity", "phase"):
        assert branch["after"][channel][:8] == branch["before"][channel]
    assert branch["after"]["pressure"][0] > branch["before"]["pressure"][0]
    assert branch["after"]["pressure"][1:8] == branch["before"]["pressure"][1:]
    assert branch["after"]["pressure"][-1] == 0.0
    assert branch["after"]["capacity"][-1] == 0.95
    assert Fraction.from_float(branch["after"]["epi"][-1]) > 0


def test_enabled_optional_gate_does_not_silently_substitute_a_parent(study):
    branch = study["enabled_gate_counterfactual"]
    rows = branch["eligibility"]["candidates"]
    assert all(row["optional_gate_enabled"] for row in rows)
    assert rows[0]["grammar_allowed"]
    assert branch["before"]["pressure"][0] < 0
    assert not rows[0]["application_preconditions_passed"]
    assert all(not row["grammar_allowed"] for row in rows[1:])
    assert branch["eligibility"]["eligible_nodes"] == ()
    assert branch["dispatch_eligibility"] == branch["eligibility"]
    assert branch["stage_result"] is None
    assert branch["parent_children"] == ()
    assert branch["children"] == ()
    assert branch["after"] == branch["before"]


def test_report_is_finite_json_and_preserves_explicit_scope(study):
    payload = json.loads(json.dumps(_payload(study), allow_nan=False))
    assert payload["configured_policy"]["eligibility"]["eligible_nodes"] == [0]
    assert payload["enabled_gate_counterfactual"]["eligibility"]["eligible_nodes"] == []
    limitations = " ".join(payload["limitations"])
    assert "explicit execution policy" in limitations
    assert "no UM, refresh or flow" in limitations
    assert "No autonomous persistence" in limitations
