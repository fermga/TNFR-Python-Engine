"""Actual child coupling does not redefine success by moving its reference."""

from fractions import Fraction

import pytest

from benchmarks.child_coupling_feedback import (
    CASES, CHILD_WORD, SEGMENT_COUNT, STEP, run_child_feedback_case,
)
from tnfr.constants.canonical import COUPLING_GENTLE


@pytest.fixture(scope="module")
def cases():
    return {name: run_child_feedback_case(name) for name in CASES}


def test_both_cases_continue_the_same_retained_pre_sha_state(cases):
    action, control = (cases[name] for name in CASES)
    assert action["checkpoint"] == control["checkpoint"]
    assert action["event"]["before"] == control["event"]["before"]
    assert action["original_reference"] == control["original_reference"]
    for case in (action, control):
        checkpoint = case["checkpoint"]
        assert checkpoint["executed_prefix_segment_count"] == 24
        assert checkpoint["prefix_elapsed_time"] == 6.0
        assert checkpoint["prefix_mean_identity_residuals"] == (0,) * 24
        assert checkpoint["retained_endpoint"]["time"] == 6.5
        for field in (
            "strengths", "metric_weights", "relative_profile", "forcing",
            "epi_weight",
        ):
            assert case["original_reference"][field] == checkpoint[
                "prefix_reference"
            ][field]
        assert case["event"]["before"] == checkpoint["retained_endpoint"]
        assert case["event"]["before"]["glyph_history"][0] == (
            "IL", "OZ", "THOL", "UM",
        )
        assert case["event"]["before"]["glyph_history"]["0_sub_0"] == ()
        assert case["postevent_elapsed_time"] == 3.0
        assert case["final_before_closure"]["time"] == 9.5


def test_child_live_gate_and_own_word_admit_real_auxiliary_writes(cases):
    event = cases["child_coupling"]["event"]
    assert event["child_word"] == CHILD_WORD == ("coupling", "silence")
    assert event["child_word_both_validators_passed"]
    assert event["actual_admission"] == {
        "target": "0_sub_0", "candidate": "UM", "allowed": True,
    }
    assert event["actual_candidate_sample"] == event["before"]["nodes"]
    before, raw = event["before"], event["raw_after_event"]
    assert raw["epi"] == before["epi"]
    assert raw["phase"] == before["phase"]
    assert raw["capacity"][-1] == 0.95 + COUPLING_GENTLE * (1.0 - 0.95)
    assert raw["capacity"][:-1] == before["capacity"][:-1]
    assert raw["glyph_history"]["0_sub_0"] == ("UM",)
    assert raw["pressure"][-1] != before["pressure"][-1]
    assert event["after_refresh"]["pressure"][-1] != raw["pressure"][-1]


def test_actual_child_commit_matches_pure_kernel_edges_and_node_channels(cases):
    event = cases["child_coupling"]["event"]
    proposal = event["readonly_kernel_proposal"]
    assert proposal["targets"] == ("0_sub_0",)
    assert proposal["target_proposals"][0]["compatible_neighbors"] == (0,)
    expected_edges = tuple(
        (edge["left"], edge["right"], {"weight": edge["weight"]})
        for edge in proposal["edges"]
    )
    assert event["actual_new_edges"] == expected_edges
    assert tuple((u, v) for u, v, _ in expected_edges) == (
        (1, "0_sub_0"), (7, "0_sub_0"),
    )
    assert expected_edges[0][2]["weight"] != expected_edges[1][2]["weight"]
    target = proposal["target_proposals"][0]
    assert all(data["weight"] >= target["compatibility_threshold"]
               for _, _, data in expected_edges)
    assert all(value <= target["effective_phase_limit"]
               for value in event["new_edge_phase_separations"])
    before, raw = event["before"], event["raw_after_event"]
    index = {node: i for i, node in enumerate(before["nodes"])}
    expected = {key: list(before[key]) for key in ("phase", "capacity", "pressure")}
    for update in proposal["node_updates"]:
        for field, key in (
            ("theta_after", "phase"), ("vf_after", "capacity"),
            ("dnfr_after", "pressure"),
        ):
            if update[field] is not None:
                expected[key][index[update["node"]]] = update[field]
    assert all(raw[key] == tuple(values) for key, values in expected.items())


def test_reference_reset_does_not_change_fixed_original_pattern_error(cases):
    case = cases["child_coupling"]
    event = case["event"]
    assert event["original_pattern_before"] == event["original_pattern_after"]
    reset = event["same_epi_reference_reset"]
    assert reset["before"]["snapshot"]["epi"] == reset["after"]["snapshot"]["epi"]
    assert reset["after"]["error_variance"] > reset["before"]["error_variance"]
    assert reset["profile_shift"] != (0,) * 9
    assert reset["error_shift"] != (0,) * 9
    for name in ("variance_budget", "dirichlet_budget"):
        budget = reset[name]
        assert budget["identity_residual"] == 0
        assert budget["energy_change"] == (
            budget["metric_term"] + budget["reference_cross_term"]
            + budget["reference_quadratic_term"]
        )
    assert reset["raw_support_reset"]["identity_residual"] == 0
    assert reset["error_support_reset"]["identity_residual"] == 0


def test_no_extra_event_control_keeps_reference_and_pattern_continuous(cases):
    case = cases["no_extra_event"]
    event = case["event"]
    assert event["actual_admission"] is None
    assert event["readonly_kernel_proposal"] is None
    assert event["actual_new_edges"] == ()
    assert event["before"] == event["raw_after_event"] == event["after_refresh"]
    assert event["original_pattern_before"] == event["original_pattern_after"]
    assert case["original_reference"] == case["postevent_reference"]
    reset = event["same_epi_reference_reset"]
    assert reset["variance_budget"]["energy_change"] == 0
    assert reset["dirichlet_budget"]["energy_change"] == 0
    assert reset["drift_change"] == reset["mean_reweighting"] == 0


def test_compatibility_channels_explain_the_measured_drift_change(cases):
    case = cases["child_coupling"]
    event = case["event"]
    before = event["before_compatibility_channels"]
    after = event["after_compatibility_channels"]
    assert before["exact_sum_residual"] == after["exact_sum_residual"] == 0
    assert after["weighted_forcing_by_channel"]["vf"] < before[
        "weighted_forcing_by_channel"
    ]["vf"]
    assert after["weighted_forcing_by_channel"]["phase"] > 0
    assert after["weighted_forcing_by_channel"]["topo"] == 0
    old = case["original_reference"]["mean_drift"]
    new = case["postevent_reference"]["mean_drift"]
    assert old < new < 0
    assert case["event"]["same_epi_reference_reset"]["drift_change"] == new - old


def test_smaller_drift_does_not_recover_the_fixed_original_pattern(cases):
    action = cases["child_coupling"]
    control = cases["no_extra_event"]
    initial = action["event"]["original_pattern_before"]["error_variance"]
    assert abs(action["postevent_reference"]["mean_drift"]) < abs(
        control["postevent_reference"]["mean_drift"]
    )
    assert action["final_original_pattern"]["error_variance"] > initial
    assert control["final_original_pattern"]["error_variance"] < initial


@pytest.mark.parametrize("name", CASES)
def test_postevent_flow_has_bound_default_integrator_and_fixed_regime(cases, name):
    case = cases[name]
    previous = case["event"]["after_refresh"]
    previous_pattern = case["event"]["original_pattern_after"]
    assert len(case["segments"]) == SEGMENT_COUNT == 12
    for segment in case["segments"]:
        assert segment["before"] == previous
        assert segment["original_pattern_before"] == previous_pattern
        assert segment["duration"] == STEP == 0.25
        assert all(segment["frozen_input_checks"].values())
        evidence = segment["executor_evidence"]
        assert evidence["integrator_provenance_certified"]
        assert evidence["resolved_method"] == "euler"
        assert evidence["resolved_substeps"] == 1
        assert evidence["gamma_is_none"]
        assert not evidence["extended_dynamics_requested"]
        assert not evidence["clipping_applied"]
        assert all(evidence["left_binding"].values())
        assert all(evidence["right_binding"].values())
        assert evidence["duration"] == 0.25
        budget = segment["regime_step_budget"]
        assert budget["convex_step_admissible"]
        assert budget["mean_identity_residual"] == 0
        assert budget["relative_recurrence_residual"] == (0,) * 9
        assert budget["relative_energy_budget"]["identity_residual"] == 0
        assert max(map(abs, budget["support_budget"]["state_defect"])) < 1e-14
        previous = segment["after_refresh"]
        previous_pattern = segment["original_pattern_after"]
    assert previous == case["final_before_closure"]
    assert previous_pattern == case["final_original_pattern"]


@pytest.mark.parametrize("name", CASES)
def test_original_pattern_uses_the_fixed_old_metric_and_profile(cases, name):
    case = cases[name]
    reference = case["original_reference"]
    pattern = case["final_original_pattern"]
    metric = reference["metric_weights"]
    epi = tuple(Fraction.from_float(value) for value in case[
        "final_before_closure"
    ]["epi"])
    mean = sum(h * x for h, x in zip(metric, epi)) / sum(metric)
    error = tuple(x - mean - z for x, z in zip(epi, reference["relative_profile"]))
    assert pattern["mean"] == mean
    assert pattern["relative_error"] == error
    assert pattern["error_variance"] == sum(
        h * value**2 for h, value in zip(metric, error)
    ) / 2
    assert case["final_regime_state"]["error_variance"] < case[
        "event"
    ]["same_epi_reference_reset"]["after"]["error_variance"]


def test_both_pending_words_close_only_after_measured_flow(cases):
    for name, case in cases.items():
        closures = case["closures_after_measurement"]
        expected_targets = ("0_sub_0", 0) if name == "child_coupling" else (0,)
        assert tuple(
            item["admission"]["target"] for item in closures
        ) == expected_targets
        for item in closures:
            assert item["admission"]["allowed"]
            assert item["after"]["epi"] == case["final_before_closure"]["epi"]
            assert item["after"]["time"] == 9.5
        assert closures[-1]["after"]["glyph_history"][0][-1] == "SHA"
    assert cases["child_coupling"]["closures_after_measurement"][-1]["after"][
        "glyph_history"
    ]["0_sub_0"] == ("UM", "SHA")
