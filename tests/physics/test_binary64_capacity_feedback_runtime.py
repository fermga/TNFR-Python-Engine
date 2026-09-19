"""Actual capacity-boundary events and EPI stasis despite nonzero pressure."""

import json
from fractions import Fraction

import pytest

from benchmarks import binary64_capacity_feedback as campaign
from tnfr import _binary64
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import binary64_remesh_relative_defect as remesh

F = Fraction


@pytest.fixture(scope="module")
def cases():
    return {
        index: campaign.run_binary64_capacity_case(index) for index in campaign.INDICES
    }


def test_one_finite_admitted_word_and_flow_per_declared_preparation(cases):
    comparison = cases[0]["uniform_comparison_reference"]
    for index, record in cases.items():
        assert record["status"] == "measured"
        assert record["word"]["string_validator_passed"]
        assert record["word"]["instance_validator_passed"]
        assert record["word"]["names"] == ("coupling", "silence")
        assert record["uniform_comparison_reference"] == comparison
        assert comparison["relative_profile"] == (0, 0)
        assert comparison["metric_weights"] == (1, 1)
        assert record["initial_capture"]["snapshot"]["capacity"] == (
            1 + F(index, 2**52),
            1,
        )
        assert record["initial"]["state"]["time"] == 0.0
        assert record["final_before_closure"]["state"]["time"] == 0.25
        assert record["event"]["status"] == "executed"
        assert record["event"]["admission"]["allowed"]
        assert record["event"]["requested_glyph"] == "UM"
        assert record["final_before_closure"]["state"]["glyph_history"][0] == ("UM",)


def test_actual_default_kernel_endpoints_match_independent_lattice_boundaries(cases):
    expected = {0: 0, 1: 1, 6: 6, 7: 6, 2**52: 4145214556169080}
    gamma = canonical_glyph_factor_defaults()["UM_vf_sync"]
    for index, record in cases.items():
        observation = record["lattice_observation"]
        assert observation["index_before"] == index
        assert observation["index_after"] == expected[index]
        assert observation["eventual_index"] == min(index, 6)
        assert observation["termination_horizon_from_before"] == (
            0 if index <= 6 else 512
        )
        assert record["event"]["resolved_factors"]["UM_vf_sync"] == gamma
        assert record["post_um_capture"]["snapshot"]["capacity"] == (
            1 + F(expected[index], 2**52),
            1,
        )


def test_actual_family_and_executor_inputs_are_retained(cases):
    for record in cases.values():
        assert all(all(checks.values()) for checks in record["family_checks"])
        evidence = record["flow"]["executor_evidence"]
        assert evidence["integrator_provenance_certified"]
        assert evidence["resolved_method"] == "euler"
        assert evidence["resolved_substeps"] == 4
        assert evidence["gamma_is_none"]
        assert not evidence["extended_dynamics_requested"]
        assert not evidence["clipping_applied"]
        assert evidence["duration"] == F(1, 4)
        assert all(evidence["left_binding"].values())
        assert all(evidence["right_binding"].values())
        assert record["event"]["after_forcing_capture"] == record["post_um_capture"]
        assert (
            record["post_um_capture"]["snapshot"]["capacity"]
            == record["final_capture"]["snapshot"]["capacity"]
        )


@pytest.mark.parametrize("index", (1, 6, 7))
def test_uniform_represented_epi_coexists_with_nonzero_frozen_profile(cases, index):
    record = cases[index]
    after = record["final_capture"]["snapshot"]
    assert after["epi"] == (F(1, 2), F(1, 2))
    assert record["represented_epi_change"] == (0, 0)
    assert record["actual_uniform_target_error"]["error_variance"] == 0
    assert record["current_conditional_profile_mismatch"]["error_variance"] > 0
    assert record["current_exact_frozen_profile"] != (0, 0)
    assert all(value != 0 for value in after["stored_pressure"])
    assert record["exact_model_epi_change"][0] < 0 < record["exact_model_epi_change"][1]
    assert all(value != 0 for value in record["execution_epi_effect"])
    for modeled, pressure, execution in zip(
        record["exact_model_epi_change"],
        record["pressure_realization_epi_effect"],
        record["execution_epi_effect"],
        strict=True,
    ):
        assert modeled + pressure + execution == 0


def test_zero_gap_control_and_upper_endpoint_do_not_get_classified_as_the_same_stall(
    cases,
):
    zero, upper = cases[0], cases[2**52]
    assert zero["current_conditional_profile_mismatch"]["error_variance"] == 0
    assert zero["exact_model_epi_change"] == zero["represented_epi_change"] == (0, 0)
    assert upper["actual_uniform_target_error"]["error_variance"] > 0
    assert any(upper["represented_epi_change"])
    assert upper["lattice_observation"]["index_decrease"] > 0


def test_exact_pressure_execution_budget_closes_at_every_actual_endpoint(cases):
    for record in cases.values():
        assert record["epi_identity_residual"] == (0, 0)
        for actual, modeled, pressure, execution in zip(
            record["represented_epi_change"],
            record["exact_model_epi_change"],
            record["pressure_realization_epi_effect"],
            record["execution_epi_effect"],
            strict=True,
        ):
            assert actual == modeled + pressure + execution
        k = (
            dict(record["post_um_capture"]["normalized_weights"])["vf"]
            / record["post_um_capture"]["epi_weight"]
        )
        gap = record["post_um_capture"]["snapshot"]["capacity"][0] - 1
        assert (
            record["current_conditional_profile_mismatch"]["error_variance"]
            == k**2 * gap**2 / 4
        )


def test_closure_is_recorded_after_measurement_without_full_state_stasis(cases):
    for record in cases.values():
        closure = record["closure_after_measurement"]
        assert closure["status"] == "executed" and closure["admission"]["allowed"]
        assert closure["before"]["state"] == record["final_before_closure"]["state"]
        assert (
            closure["after_refresh"]["state"]["epi"]
            == record["final_before_closure"]["state"]["epi"]
        )
        assert closure["after_refresh"]["state"]["glyph_history"][0] == ("UM", "SHA")
    # The finite zero-EPI-increment case still advances physical history/time.
    assert cases[6]["initial"]["state"] != cases[6]["final_before_closure"]["state"]


def test_strict_json_report_retains_small_nonzero_exact_values(cases):
    payload = campaign._artifact_payload({"cases": list(cases.values())})
    restored = json.loads(json.dumps(payload, allow_nan=False))
    six = next(record for record in restored["cases"] if record["case"] == 6)
    assert F(six["current_conditional_profile_mismatch"]["error_variance"]) > 0
    assert six["represented_epi_change"] == ["0", "0"]
    assert six["lattice_observation"]["eventual_index"] == 6


def test_remesh_and_capacity_use_the_same_platform_probe():
    assert (
        remesh._runtime_uses_required_binary64_rounding_model
        is _binary64.uses_ieee_binary64_rounding
    )
    assert _binary64.uses_ieee_binary64_rounding()


@pytest.mark.parametrize("index", (2, -1, True, 1.0))
def test_campaign_does_not_turn_a_boundary_control_into_an_implicit_sweep(index):
    with pytest.raises(ValueError, match="declared boundary controls"):
        campaign.run_binary64_capacity_case(index)
