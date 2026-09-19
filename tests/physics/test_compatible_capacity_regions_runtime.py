"""Prepared U3 components and actual simultaneous capacity mixing stay distinct."""

import math
from fractions import Fraction

import networkx as nx
import pytest

from benchmarks import compatible_capacity_regions as campaign
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults

F = Fraction


@pytest.fixture(scope="module")
def cases():
    return {
        name: campaign.run_compatible_capacity_case(name) for name in campaign.CASES
    }


def _graph(snapshot):
    graph = nx.Graph()
    graph.add_nodes_from(snapshot["nodes"])
    graph.add_edges_from(
        (i, j) for i, row in enumerate(snapshot["support_neighbors"]) for j in row
    )
    return graph


def test_full_pressure_support_is_connected_in_both_preparations(cases):
    for record in cases.values():
        initial = record["initial_capture"]
        graph = _graph(initial["snapshot"])
        assert nx.is_connected(graph)
        assert graph.number_of_nodes() == 6 and graph.number_of_edges() == 7
        assert initial["snapshot"]["epi"] == (F(1, 2),) * 6
        assert initial["snapshot"]["capacity"] == (1, 1, 1, 2, 2, 2)
        assert initial["phase_gradient"] == (0,) * 6
        assert record["full_support_capacity_energy_before"] == F(1, 2)
        assert record["word"]["string_validator_passed"]
        assert record["word"]["instance_validator_passed"]
        assert record["word"]["names"] == ("coupling", "silence")
        assert record["initial"]["state"]["time"] == 0.0
        assert record["initial"]["random_provenance"]["resolved_base_seed"] == 17
    assert (
        cases["split_phase"]["initial_capture"]["phase"]
        == (F(0),) * 3 + (F(math.pi),) * 3
    )
    assert cases["aligned_phase"]["initial_capture"]["phase"] == (0,) * 6


def test_split_u3_components_have_zero_capacity_energy_despite_the_pressure_bridge(
    cases,
):
    before = cases["split_phase"]["before_support"]
    after = cases["split_phase"]["after_support"]
    assert before["components"] == after["components"] == ((0, 1, 2), (3, 4, 5))
    assert before["excluded_edges"] == after["excluded_edges"] == ((2, 3),)
    assert before["blocked_targets"] == after["blocked_targets"] == ()
    assert all(len(row) == 2 for row in before["compatible_neighbors"])
    assert before["effective_phase_limit"] < F(math.pi)
    balance = before["balance"]
    assert balance["component_constant"] and balance["is_fixed"]
    assert balance["capacity_gradient"] == (0,) * 6
    assert balance["capacity_after"] == (1, 1, 1, 2, 2, 2)
    assert (
        balance["component_means_before"] == balance["component_means_after"] == (1, 2)
    )
    assert balance["energy_before"] == balance["energy_after"] == 0
    assert balance["identity_residual"] == 0


@pytest.mark.parametrize("name", campaign.CASES)
def test_actual_all_target_stage_keeps_prediction_distinct_from_execution(cases, name):
    event = cases[name]["event"]
    assert event["targets"] == tuple(range(6))
    assert len(event["admissions"]) == 6
    assert all(
        item["allowed"] and item["candidate"] == "UM" for item in event["admissions"]
    )
    assert event["stage_result"]["schedule"] == "two_phase_jacobi"
    assert event["stage_result"]["nodes_processed"] == 6
    assert event["stage_result"]["glyph"] == "UM"
    assert all(event["prediction_matches_actual"].values())
    assert "not executor-retained proposals" in event["prediction_scope"]
    assert len(event["actual_operator_metrics"]) == 6
    assert (
        event["resolved_factors"]["UM_vf_sync"]
        == canonical_glyph_factor_defaults()["UM_vf_sync"]
    )
    for node in range(6):
        assert event["raw_state"]["state"]["glyph_history"][node] == ("UM",)
    budget = event["budget"]
    assert budget["epi_jump"] == (0,) * 6
    assert budget["mean_identity_residual"] == 0
    assert budget["error_identity_residual"] == (0,) * 6
    assert (
        budget["variance_identity_residual"]
        == budget["dirichlet_identity_residual"]
        == 0
    )


def test_split_primary_preservation_does_not_erase_the_raw_pressure_write(cases):
    record = cases["split_phase"]
    initial, raw, refreshed = (
        record["initial_capture"],
        record["event"]["raw_capture"],
        record["post_um_capture"],
    )
    for field in ("nodes", "epi", "capacity", "conductance", "support_neighbors"):
        assert (
            initial["snapshot"][field]
            == raw["snapshot"][field]
            == refreshed["snapshot"][field]
        )
    assert initial["phase"] == raw["phase"] == refreshed["phase"]
    assert initial["forcing"] == raw["forcing"] == refreshed["forcing"]
    assert raw["snapshot"]["stored_pressure"] != initial["snapshot"]["stored_pressure"]
    assert any(raw["stored_pressure_residual"])
    assert initial["snapshot"] == refreshed["snapshot"]
    assert refreshed["stored_pressure_residual"] == (0,) * 6
    assert record["event"]["new_edges"] == ()
    assert record["event"]["capacity_exact_model_defect"] == (0,) * 6
    assert record["original_reference"] == record["post_um_reference"]
    for proposal in record["event"]["prediction"]["target_proposals"]:
        assert len(proposal["compatible_neighbors"]) == 2
        assert proposal["link_candidates"] == ()
        assert proposal["vf_before"] == proposal["vf_after"]
        assert all(
            item["theta_before"] == item["theta_proposed"]
            for item in proposal["phase_proposals"]
        )
    assert record["event"]["budget"]["variance_change"] == 0
    assert record["event"]["budget"]["dirichlet_change"] == 0


def test_split_forced_profile_has_independently_derived_metric_and_center(cases):
    record = cases["split_phase"]
    capture, reference = record["post_um_capture"], record["post_um_reference"]
    weights = dict(capture["normalized_weights"])
    k = weights["vf"] / capture["epi_weight"]
    assert reference["metric_weights"] == (2, 2, 3, F(3, 2), 1, 1)
    assert sum(reference["metric_weights"]) == F(21, 2)
    profile = (k / 3,) * 3 + (-2 * k / 3,) * 3
    assert reference["relative_profile"] == profile
    assert reference["mean_drift"] == reference["compatibility_residual"] == 0
    assert reference["profile_residual"] == (0,) * 6
    assert reference["profile_center_residual"] == 0
    prediction = record["fixed_profile_prediction"]
    assert all(prediction["checks"].values())
    assert prediction["capacity_mean"] == F(4, 3)
    assert prediction["relative_profile"] == profile
    assert prediction["conditional_exact_limit"] == tuple(
        F(1, 2) + value for value in profile
    )
    assert record["initial_fixed_target_error"]["error_variance"] == 7 * k**2 / 6
    assert record["initial_fixed_target_error"]["error_dirichlet_energy"] == k**2 / 2


def test_aligned_control_activates_bridge_and_keeps_default_functional_links(cases):
    record = cases["aligned_phase"]
    before, after = record["before_support"], record["after_support"]
    assert before["components"] == after["components"] == (tuple(range(6)),)
    assert before["excluded_edges"] == after["excluded_edges"] == ()
    assert before["blocked_targets"] == after["blocked_targets"] == ()
    assert before["balance"]["component_means_before"] == (F(3, 2),)
    assert before["balance"]["component_means_after"] == (F(3, 2),)
    assert not before["balance"]["component_constant"]
    gamma = before["coupling_factor"]
    assert before["balance"]["energy_change"] == -2 * gamma / 3 + 4 * gamma**2 / 9
    assert before["balance"]["energy_change"] < 0
    assert len(record["event"]["new_edges"]) == 8
    assert len(record["event"]["prediction"]["edges"]) == 8
    graph = _graph(record["post_um_capture"]["snapshot"])
    assert graph.number_of_edges() == 15
    assert all(degree == 5 for _, degree in graph.degree())
    assert all(data["weight"] == 1 for _, _, data in record["event"]["new_edges"])


def test_aligned_capacity_update_uses_old_neighbors_not_new_k6_edges(cases):
    record = cases["aligned_phase"]
    gamma = record["event"]["resolved_factors"]["UM_vf_sync"]
    expected = (
        1,
        1,
        F(1.0 + gamma * (4.0 / 3.0 - 1.0)),
        F(2.0 + gamma * (5.0 / 3.0 - 2.0)),
        2,
        2,
    )
    actual = record["post_um_capture"]["snapshot"]["capacity"]
    assert actual == expected
    exact_gamma = F(gamma)
    ideal = (1, 1, 1 + exact_gamma / 3, 2 - exact_gamma / 3, 2, 2)
    assert record["before_support"]["balance"]["capacity_after"] == ideal
    assert record["event"]["capacity_exact_model_defect"] == tuple(
        a - b for a, b in zip(actual, ideal)
    )
    assert actual[2] != ideal[2] and actual[3] != ideal[3]
    assert record["post_um_capture"]["phase"] == (0,) * 6
    assert record["post_um_capture"]["snapshot"]["epi"] == (F(1, 2),) * 6


@pytest.mark.parametrize("name", campaign.CASES)
def test_one_held_nodal_interval_has_actual_endpoint_and_defect_evidence(cases, name):
    record = cases[name]
    flow = record["flow"]
    evidence, budget = flow["executor_evidence"], flow["regime_step_budget"]
    assert flow["duration"] == F(1, 4)
    assert flow["before"] == record["event"]["after_refresh"]["state"]
    assert flow["after_refresh"] == record["final_before_closure"]["state"]
    assert all(flow["frozen_input_checks"].values())
    assert evidence["integrator_provenance_certified"]
    assert evidence["resolved_method"] == "euler" and evidence["resolved_substeps"] == 4
    assert evidence["gamma_is_none"] and not evidence["extended_dynamics_requested"]
    assert not evidence["clipping_applied"]
    assert all(evidence["left_binding"].values()) and all(
        evidence["right_binding"].values()
    )
    assert budget["before"]["snapshot"] == record["post_um_capture"]["snapshot"]
    assert budget["after"]["snapshot"] == record["final_capture"]["snapshot"]
    assert budget["mean_identity_residual"] == 0
    assert budget["relative_recurrence_residual"] == (0,) * 6
    assert budget["support_budget"]["identity_residual"] == 0
    assert budget["relative_energy_budget"]["identity_residual"] == 0
    final = record["final_before_closure"]["state"]
    assert final["time"] == 0.25
    for node, value in zip(final["nodes"], final["epi"], strict=True):
        assert final["physical_epi_history"][node][-1] == (0.25, value)


def test_split_capacity_preservation_is_not_nodal_stationarity(cases):
    record = cases["split_phase"]
    before, after = (
        record["post_um_capture"]["snapshot"],
        record["final_capture"]["snapshot"],
    )
    assert any(before["stored_pressure"])
    assert before["epi"] != after["epi"]
    assert after["epi"][2] > F(1, 2) > after["epi"][3]
    assert all(after["epi"][node] == F(1, 2) for node in (0, 1, 4, 5))
    assert before["capacity"] == after["capacity"]
    assert before["conductance"] == after["conductance"]
    assert (
        record["final_fixed_target_error"]["error_variance"]
        < record["initial_fixed_target_error"]["error_variance"]
    )


def test_sha_closes_every_target_only_after_preservation_and_flow_measurements(cases):
    for record in cases.values():
        closure = record["closure_after_measurement"]
        assert closure["stage_result"]["glyph"] == "SHA"
        assert closure["stage_result"]["schedule"] == "two_phase_jacobi"
        assert closure["stage_result"]["nodes_processed"] == 6
        assert (
            len(closure["admissions"]) == len(closure["actual_operator_metrics"]) == 6
        )
        assert all(item["allowed"] for item in closure["admissions"])
        before, after = (
            record["final_before_closure"]["state"],
            closure["after"]["state"],
        )
        assert after["epi"] == before["epi"] and after["time"] == before["time"] == 0.25
        assert all(a < b for a, b in zip(after["capacity"], before["capacity"]))
        assert all(after["glyph_history"][node] == ("UM", "SHA") for node in range(6))
        assert "autonomous region formation remain open" in record["scope"]
        assert "not a fixed-support ablation" in record["scope"]


def test_only_the_two_declared_preparations_are_admitted():
    with pytest.raises(ValueError, match="case must"):
        campaign.prepare_capacity_regions("seek_formation")
    with pytest.raises(ValueError, match="case must"):
        campaign.run_compatible_capacity_case("seek_formation")
