"""Frozen support separates relative relaxation, mean drift and hard clipping."""

from fractions import Fraction

import pytest

from benchmarks.forced_support_balance import (
    CASES,
    SEGMENT_COUNTS,
    STEP,
    run_forced_support_case,
)


@pytest.fixture(scope="module")
def cases():
    return {case: run_forced_support_case(case) for case in CASES}


def test_causal_campaign_starts_at_actual_birth_and_attachment_endpoint(cases):
    case = cases["causal_attached"]
    source = case["source_preparation"]
    assert source["preparation"]["children"] == ("0_sub_0",)
    assert source["coupling"]["actual_new_edges"][0][:2] == (0, "0_sub_0")
    assert case["initial"] == source["coupling"]["after_refresh"]
    assert case["initial"]["time"] == 0.5
    assert case["initial"]["glyph_history"][0] == ("IL", "OZ", "THOL", "UM")
    assert case["physical_elapsed_time"] == 6.0
    assert case["final"]["time"] == 6.5
    closed = case["closure_after_measurement"]["after"]
    assert closed["epi"] == case["final"]["epi"]
    assert closed["glyph_history"][0][-1] == "SHA"
    assert closed["capacity"][0] < case["final"]["capacity"][0]


def test_controls_are_independent_initial_data_without_inherited_live_history(cases):
    causal = cases["causal_attached"]["initial"]
    for name in ("prepared_compatible", "prepared_clipping"):
        case = cases[name]
        initial = case["initial"]
        assert initial["time"] == 0.0
        assert initial["edges"] == causal["edges"]
        assert all(not history for history in initial["glyph_history"].values())
        assert all(not history for history in initial["physical_epi_history"].values())
        assert case["closure_after_measurement"] is None
    compatible = cases["prepared_compatible"]["initial"]
    assert compatible["epi"] == causal["epi"]
    assert compatible["capacity"] == (1.0,) * 9
    assert compatible["phase"] == (0.0,) * 9
    clipping = cases["prepared_clipping"]["initial"]
    assert clipping["epi"] == (3.99,) * 9
    assert clipping["phase"] == causal["phase"]
    assert clipping["capacity"] == causal["capacity"]


@pytest.mark.parametrize("name", CASES)
def test_forcing_and_auxiliary_inputs_remain_frozen_at_actual_boundaries(cases, name):
    case = cases[name]
    initial_capture = case["initial_forcing_capture"]
    previous = case["initial"]
    assert len(case["segments"]) == SEGMENT_COUNTS[name]
    for segment in case["segments"]:
        assert segment["before"] == previous
        assert segment["duration"] == STEP == 0.25
        assert segment["method"] == "euler"
        assert all(segment["frozen_input_checks"].values())
        after = segment["after_refresh"]
        for key in ("phase", "capacity", "nodes", "edges"):
            assert after[key] == case["initial"][key]
        captured = segment["forcing_capture"]
        assert captured["forcing"] == initial_capture["forcing"]
        assert captured["normalized_weights"] == initial_capture["normalized_weights"]
        assert max(map(abs, captured["stored_pressure_residual"])) < 1e-14
        assert max(map(abs, captured["kernel_pressure_defect"])) < 1e-14
        previous = after
    assert previous == case["final"]


def test_actual_capacity_phase_forcing_has_nonzero_derived_mean_drift(cases):
    case = cases["causal_attached"]
    reference = case["reference"]
    metric = reference["metric_weights"]
    assert reference["compatibility_residual"] < 0
    assert reference["mean_drift"] < 0
    assert reference["mean_drift"] * sum(metric) == reference["compatibility_residual"]
    assert (
        sum(
            weight * value
            for weight, value in zip(
                metric,
                reference["relative_profile"],
                strict=True,
            )
        )
        == 0
    )
    assert reference["profile_residual"] == (0,) * 9
    assert reference["max_convex_step"] >= Fraction(1, 4)
    initial = case["initial_relative_state"]
    final = case["final_relative_state"]
    assert final["mean"] < initial["mean"]
    assert final["error_variance"] < initial["error_variance"]
    assert final["error_variance"] > 0


def test_compatible_control_has_zero_forcing_drift_and_centered_profile(cases):
    case = cases["prepared_compatible"]
    reference = case["reference"]
    assert reference["forcing"] == (0,) * 9
    assert reference["compatibility_residual"] == reference["mean_drift"] == 0
    assert reference["relative_profile"] == (0,) * 9
    initial = case["initial_relative_state"]
    final = case["final_relative_state"]
    assert abs(final["mean"] - initial["mean"]) < 1e-14
    assert final["error_variance"] < initial["error_variance"]


@pytest.mark.parametrize("name", CASES)
def test_mean_and_relative_error_budgets_telescope_over_executed_steps(cases, name):
    case = cases[name]
    observations = [item["exact_step_observation"] for item in case["segments"]]
    for observation in observations:
        assert observation["convex_step_admissible"]
        assert observation["mean_identity_residual"] == 0
        assert observation["relative_recurrence_residual"] == (0,) * 9
        assert observation["support_budget"]["identity_residual"] == 0
        assert observation["relative_energy_budget"]["identity_residual"] == 0
        assert observation["mean_change"] == (
            observation["mean_model_change"]
            + observation["mean_pressure_defect"]
            + observation["mean_step_defect"]
        )
    actual_mean_change = (
        case["final_relative_state"]["mean"] - case["initial_relative_state"]["mean"]
    )
    assert sum(row["mean_change"] for row in observations) == actual_mean_change
    assert sum(row["mean_model_change"] for row in observations) == (
        case["reference"]["mean_drift"]
        * Fraction.from_float(case["physical_elapsed_time"])
    )


@pytest.mark.parametrize("name", ["causal_attached", "prepared_compatible"])
def test_unclipped_euler_separates_pressure_realization_from_step_rounding(cases, name):
    for segment in cases[name]["segments"]:
        assert not segment["clipping_applied"]
        observation = segment["exact_step_observation"]
        assert max(map(abs, observation["support_budget"]["state_defect"])) < 1e-14
        assert abs(observation["mean_pressure_defect"]) < 1e-14
        assert abs(observation["mean_step_defect"]) < 1e-14


def test_local_hard_clipping_changes_mean_budget_despite_negative_model_drift(cases):
    case = cases["prepared_clipping"]
    assert case["reference"]["mean_drift"] < 0
    assert all(item["clipping_applied"] for item in case["segments"])
    observations = [item["exact_step_observation"] for item in case["segments"]]
    assert any(
        max(map(abs, row["support_budget"]["state_defect"])) > 1e-3
        for row in observations
    )
    assert all(row["mean_step_defect"] < 0 for row in observations)
    assert all(abs(row["mean_pressure_defect"]) < 1e-14 for row in observations)
    assert max(case["final"]["epi"]) == 4.0
    assert min(case["final"]["epi"]) > -4.0
