"""The midpoint comparison preserves historical inputs and finite experiment scope."""

from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path

import pytest

from benchmarks import c6_winding_phase_kernel as campaign

INPUT = Path(__file__).resolve().parents[2] / "artifacts/research/c6_winding_joint_domain.json"


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("retained B16 local artifact is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def comparison(parent):
    original = deepcopy(parent)
    result = campaign.compare_c6_phase_kernel(parent)
    assert parent == original
    return result


def test_same_three_cases_two_cycles_and_shared_eligible_results(comparison):
    assert comparison["runtime_executed"] is True
    assert comparison["cycle_count_per_case"] == 2
    assert tuple(case["mode"] for case in comparison["current_cases"]) == ("null", "k1", "k3")
    assert sum(case["eligible_il_rows"] for case in comparison["current"]) == 36
    for summary in comparison["current"]:
        assert summary["different_il_deltas"] == 0
        assert summary["different_il_means"] == 0
        assert summary["different_phase_gradients"] == 0
        assert all(row["recorded_method"] == "exact_two_neighbor_midpoint"
                   for row in summary["il_midpoint_rows"])


def test_historical_midpoint_tie_is_wrong_even_before_subtraction(comparison):
    row = next(row for row in comparison["retained"][2]["il_midpoint_rows"]
               if row["ordinal"] == 1 and row["node"] == 3)
    midpoint = sum(row["neighbors"], F(0)) / 2
    assert midpoint == F(14149309535295891, 2**52)
    assert row["recorded_mean"] == midpoint - F(1, 2**52)
    assert row["midpoint"]["mean"] == float(midpoint)
    assert row["midpoint"]["mean"] == math.nextafter(float(row["recorded_mean"]), math.inf)
    assert row["mean_matches_certified_midpoint"] is False
    assert row["midpoint"]["delta"] == float(midpoint - row["il_center"])


def test_comparison_retains_signed_pressure_and_integration_mean_terms(comparison):
    for period in ("retained", "current"):
        for case in comparison[period]:
            total = F(0)
            for budget in case["flow_mean_budgets"]:
                assert budget["actual_mean_change"] == (
                    budget["held_pressure_mean_effect"] + budget["integrator_rounding_mean_effect"]
                )
                assert budget["identity_residual"] == 0
                total += budget["actual_mean_change"]
            assert total == case["total_epi_mean_change"]
            assert total == case["final_epi_mean"] - case["initial_epi_mean"]


def test_improved_kernel_does_not_claim_future_mean_control(comparison):
    assert comparison["future_mean_bound_verified"] is False
    assert comparison["production_invariant_class_certified"] is False
    assert comparison["empirical_correspondence_tested"] is False
    # The fixed current k1 control still has a nonzero mean defect.
    assert comparison["current"][1]["total_epi_mean_change"] != 0
    assert "future" in comparison["scope"]


def test_shared_serializer_preserves_only_named_sha_infinity(comparison):
    encoded = json.dumps(campaign._payload(comparison), allow_nan=False)
    decoded = json.loads(encoded)
    for case in decoded["current_cases"]:
        metrics = case["closure_after_measurement"]["actual_operator_metrics"]
        assert any(metric.get("time_to_collapse") == {"numeric_kind": "positive_infinity"}
                   for metric in metrics)


def test_current_cases_keep_preparation_coefficients_word_and_closure(parent, comparison):
    for old, current in zip(parent["cases"], comparison["current_cases"], strict=True):
        assert current["joint_domain_reference"] == old["joint_domain_reference"]
        assert current["initial_capture"]["phase"] == old["initial_capture"]["phase"]
        for name in ("configured_controls", "random_provenance"):
            assert current["initial"][name] == old["initial"][name]
        assert current["initial"]["state"]["time"] == old["initial"]["state"]["time"]
        for name in ("epi", "capacity", "conductance", "support_neighbors"):
            assert current["initial_capture"]["snapshot"][name] == old["initial_capture"]["snapshot"][name]
        assert current["word"]["names"] == old["word"]["names"]
        assert current["word"]["string_validator_passed"]
        assert current["word"]["instance_validator_passed"]
        assert current["final_before_closure"]["state"]["time"] == .5
        closure = current["closure_after_measurement"]["after"]["state"]
        assert closure["epi"] == current["final_before_closure"]["state"]["epi"]
        assert all(history == ["UM", "IL", "UM", "IL", "SHA"] for history in closure["glyph_history"].values())


def test_invalid_retained_source_is_rejected_before_any_new_execution(parent, monkeypatch):
    changed = deepcopy(parent)
    changed["cases"][0]["cycles"][0]["after_capture"]["snapshot"]["epi"][0] = "3/4"

    def forbidden(*args):
        raise AssertionError("invalid historical input must fail before running a graph")

    monkeypatch.setattr(campaign, "run_c6_joint_case", forbidden)
    with pytest.raises(ValueError):
        campaign.compare_c6_phase_kernel(changed)


def test_detached_il_comparison_checks_proposal_endpoint_binding(parent):
    case = deepcopy(parent["cases"][2])
    case["cycles"][0]["il"]["independent_prediction"][3]["phase"]["theta_after"] = 0.0
    with pytest.raises(ValueError, match="source and endpoint"):
        campaign._midpoint_rows(case)
