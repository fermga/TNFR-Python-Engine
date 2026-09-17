"""Finite canonical P2 words, runtime defects and the binary64 gap obstruction."""

from dataclasses import replace
from fractions import Fraction
import json
import math

import pytest

from benchmarks import capacity_feedback as campaign
from tnfr.operators.definitions import Coupling, Silence
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics.forcing_realization import capture_non_epi_forcing

F = Fraction


@pytest.fixture(scope="module")
def cases():
    return {case: campaign.run_capacity_feedback_case(case) for case in campaign.CASES}


def test_branches_have_identical_actual_preparations_and_target(cases):
    coupled, held = (cases[key] for key in campaign.CASES)
    assert coupled["prepared"] == held["prepared"]
    assert coupled["preparation"] == held["preparation"]
    assert coupled["original_target"] == held["original_target"]
    assert coupled["original_target"]["relative_profile"] == (0, 0)
    assert coupled["initial_original_pattern"] == held["initial_original_pattern"]
    for record in cases.values():
        assert record["status"] == "measured"
        assert record["word"]["string_validator_passed"]
        assert record["word"]["instance_validator_passed"]
        assert len(record["cycles"]) == 16
        assert record["final_before_closure"]["state"]["time"] == 4.0


def test_uninterrupted_repeated_um_is_rejected_before_execution(cases):
    for record in cases.values():
        control = record["direct_repeated_um_control"]
        assert not control["string_validator_passed"]
        assert control["instance_validator_passed"]
        assert control["names"][2:4] == ("coupling", "coupling")


def test_actual_events_keep_default_factors_and_record_every_admission(cases):
    defaults = canonical_glyph_factor_defaults()
    record = cases["coupled"]
    for index, cycle in enumerate(record["cycles"]):
        event, separator = cycle["event"], cycle["separator"]
        assert event["status"] == separator["status"] == "executed"
        assert event["admission"]["allowed"] and separator["admission"]["allowed"]
        assert event["requested_glyph"] == "UM"
        assert separator["requested_glyph"] == "IL"
        assert event["sequence_index"] == 2 + 2 * index
        assert separator["sequence_index"] == 3 + 2 * index
        assert event["resolved_factors"]["UM_vf_sync"] == defaults["UM_vf_sync"]
        assert separator["resolved_factors"]["IL_dnfr_factor"] == defaults["IL_dnfr_factor"]
        before = event["before"]["state"]
        after = event["after_refresh"]["state"]
        assert after["capacity"][1] == before["capacity"][1] == 1
        assert 1 < after["capacity"][0] < before["capacity"][0]
        assert after["epi"] == before["epi"]
        assert after["phase"] == before["phase"] == (0, 0)
        assert after["edges"] == before["edges"]
    assert all(cycle["event"] is cycle["separator"] is None
               for cycle in cases["held_capacity"]["cycles"])


def test_il_separator_has_a_raw_pressure_effect_but_no_refreshed_primary_effect(cases):
    for cycle in cases["coupled"]["cycles"]:
        separator = cycle["separator"]
        before = separator["before_forcing_capture"]
        raw = separator["raw_forcing_capture"]
        after = separator["after_forcing_capture"]
        assert before["snapshot"] == after["snapshot"]
        assert before["forcing"] == after["forcing"]
        assert before["phase"] == after["phase"] == (0, 0)
        assert raw["stored_pressure_residual"][0] != 0
        assert after["stored_pressure_residual"] == (0, 0)


def test_finite_domain_and_execution_evidence_are_checked_at_each_boundary(cases):
    for record in cases.values():
        assert record["initial_domain"]["inside"]
        assert record["coupled_model"]["strict_disagreement_contraction"]
        for cycle in record["cycles"]:
            assert all(all(checks.values()) for checks in cycle["family_checks"])
            for key in ("domain_before", "domain_after_event", "domain_after_flow"):
                assert cycle[key]["inside"]
                assert min(cycle[key]["lower_margins"] + cycle[key]["upper_margins"]) >= 0
            evidence = cycle["flow"]["executor_evidence"]
            assert evidence["integrator_provenance_certified"]
            assert evidence["resolved_method"] == "euler"
            # Default internal substeps hold the same pressure/capacity; their
            # exact endpoint is the one-h map, with rounding retained separately.
            assert evidence["resolved_substeps"] == 4
            assert evidence["gamma_is_none"]
            assert not evidence["extended_dynamics_requested"]
            assert not evidence["clipping_applied"]
            assert all(evidence["left_binding"].values())
            assert all(evidence["right_binding"].values())
            assert F(evidence["duration"]) == F(1, 4)
            assert cycle["flow"]["regime_step_budget"]["convex_step_admissible"]


def test_three_error_sources_reconstruct_actual_endpoints_without_zeroing_rounding(cases):
    nonzero = {"capacity": False, "pressure": False, "execution": False}
    for cycle in cases["coupled"]["cycles"]:
        defect = cycle["defects"]
        assert defect["identity_residual"] == (0, 0)
        for i in range(2):
            assert defect["epi_defect"][i] == sum(defect[name][i] for name in (
                "capacity_transition_epi_effect", "pressure_realization_epi_effect",
                "execution_epi_effect",
            ))
            actual = F(cycle["flow"]["after_refresh"]["epi"][i])
            assert actual == defect["ideal_epi"][i] + defect["epi_defect"][i]
        nonzero["capacity"] |= defect["capacity_gap_defect"] != 0
        nonzero["pressure"] |= any(defect["pressure_realization_epi_effect"])
        nonzero["execution"] |= any(defect["execution_epi_effect"])
        mean = defect["mean_budget"]
        assert mean["mean_model_change"] == mean["mean_identity_residual"] == 0
        assert mean["mean_change"] == mean["mean_pressure_defect"] + mean["mean_step_defect"]
    assert all(nonzero.values())


def test_fixed_uniform_target_and_frozen_profile_floor_are_separate(cases):
    coupled, held = (cases[key] for key in campaign.CASES)
    assert (0 < coupled["final_original_pattern"]["error_variance"]
            < held["final_original_pattern"]["error_variance"])
    for record in cases.values():
        snapshot = record["final_capture"]["snapshot"]
        x0, x1 = snapshot["epi"]
        assert record["final_original_pattern"]["error_variance"] == (x0 - x1)**2 / 4
        gap = snapshot["capacity"][0] - snapshot["capacity"][1]
        k = record["coupled_model"]["forcing_ratio"]
        assert record["conditional_frozen_limit"]["error_variance"] == k**2 * gap**2 / 4
        assert record["conditional_frozen_limit"]["error_variance"] > 0
        original = record["prepared"]["state"]["epi"]
        assert x0 + x1 != sum(F(value) for value in original)
    assert coupled["model_bound"] is not None
    assert held["model_bound"] is None
    # A better final response is not a monotone original-target trajectory.
    last = coupled["cycles"][-1]["flow"]
    assert (last["original_pattern_after"]["error_variance"]
            > last["original_pattern_before"]["error_variance"])


def test_sha_is_admitted_after_measurement_and_outside_the_cycle_reference(cases):
    for record in cases.values():
        closure = record["closure_after_measurement"]
        assert closure["status"] == "executed"
        assert closure["admission"]["allowed"]
        assert closure["requested_glyph"] == "SHA"
        before = record["final_before_closure"]["state"]
        assert closure["before"]["state"] == before
        assert closure["after_refresh"]["state"]["epi"] == before["epi"]
        assert closure["after_refresh"]["state"]["capacity"][0] < before["capacity"][0]


def test_production_default_um_has_a_nonzero_binary64_capacity_gap_fixed_point():
    graph = campaign.prepare_p2(capacity=(math.nextafter(1.0, math.inf), 1.0))
    before = capture_non_epi_forcing(graph)
    target = campaign._reference(before)
    ops = (Coupling(), Silence())
    word, _ = campaign._word(ops, initialized=True)
    after, event = campaign._event(graph, 0, ops[0], word.step(0), target, before, refresh=True)
    gamma = F(event["resolved_factors"]["UM_vf_sync"])
    gap = before.snapshot.capacity[0] - 1
    assert 0 < gamma < F(1, 4)
    assert gap == F(1, 2**52)
    assert event["status"] == "executed"
    assert after.snapshot.capacity == before.snapshot.capacity
    assert after.snapshot.epi == before.snapshot.epi
    assert gap - (1 - gamma) * gap == gamma * gap > 0
    assert campaign._reference(after).relative_profile != (0, 0)
    _, closure = campaign._event(graph, 0, ops[1], word.step(1), target, after, refresh=True)
    assert closure["status"] == "executed"


@pytest.mark.parametrize("field,value", (
    ("phase", (F(0), F(1))),
    ("stored_pressure_residual", (F(1), F(0))),
    ("normalized_weights", (("phase", F(1)), ("epi", F(1)), ("vf", F(1)), ("topo", F(1)))),
))
def test_changed_runtime_hypotheses_cannot_reuse_the_family(field, value):
    capture = capture_non_epi_forcing(campaign.prepare_p2())
    with pytest.raises(ValueError, match="hypotheses failed"):
        campaign._family(replace(capture, **{field: value}), capture, base_capacity=1)


def test_benchmark_rejects_unrequested_case():
    with pytest.raises(ValueError, match="case must"):
        campaign.run_capacity_feedback_case("seek_recovery")


def test_strict_artifact_preserves_the_infinite_closure_metric(cases):
    payload = campaign._artifact_payload({"cases": list(cases.values())})
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    raw = cases["coupled"]["closure_after_measurement"]["actual_operator_metrics"][0]
    assert raw["time_to_collapse"] == math.inf
    metric = decoded["cases"][0]["closure_after_measurement"]["actual_operator_metrics"][0]
    assert metric["time_to_collapse"] == {"numeric_kind": "positive_infinity"}
    assert decoded["cases"][0]["cycles"][0]["defects"]["identity_residual"] == ["0", "0"]


def test_artifact_does_not_hide_nonfinite_structural_state(cases):
    payload = campaign._artifact_payload({"cases": list(cases.values())})
    payload["cases"][0]["final_capture"]["snapshot"]["epi"][0] = math.nan
    with pytest.raises(ValueError, match="JSON compliant"):
        json.dumps(payload, allow_nan=False)
