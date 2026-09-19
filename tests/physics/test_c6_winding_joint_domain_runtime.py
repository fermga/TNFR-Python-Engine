"""Two fixed C6 cycles retain admission, nodal flow and conditional-model limits."""

import math
from fractions import Fraction as F

import networkx as nx
import pytest

from benchmarks import c6_winding_joint_domain as campaign
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.preconditions import validate_coupling
from tnfr.operators.preconditions.coherence import validate_coherence_strict


@pytest.fixture(scope="module")
def cases():
    return {case: campaign.run_c6_joint_case(*case) for case in campaign.CASES}


def _recorded_graph(materialized):
    graph = nx.Graph()
    graph.graph.update(materialized["configured_controls"])
    for node, data in materialized["node_attributes"].items():
        graph.add_node(node, **data)
    graph.add_edges_from(materialized["state"]["edges"])
    return graph


def test_exactly_three_inherited_preparations_and_two_default_cycles(cases):
    assert campaign.CASES == (("null", 0.0), ("k1", 2.0**-12), ("k3", 2.0**-12))
    assert campaign.CYCLE_COUNT == 2 and campaign.STEP == 0.25
    defaults = canonical_glyph_factor_defaults()
    for case, record in cases.items():
        assert record["mode"] == case[0] and record["epsilon"] == F(case[1])
        assert len(record["cycles"]) == record["cycle_count"] == 2
        initial = record["initial_capture"]["snapshot"]
        assert initial["epi"] == (F(1, 2),) * 6
        assert initial["capacity"] == (1,) * 6
        assert record["initial"]["random_provenance"]["resolved_base_seed"] == 17
        controls = record["initial"]["configured_controls"]
        assert controls.get("UM_BIDIRECTIONAL", True)
        assert controls.get("UM_FUNCTIONAL_LINKS", True)
        assert controls.get("UM_SYNC_VF", True)
        assert controls.get("UM_STABILIZE_DNFR", True)
        assert controls["CLIP_MODE"] == "hard"
        for cycle in record["cycles"]:
            assert (
                cycle["um"]["resolved_factors"]["UM_theta_push"]
                == defaults["UM_theta_push"]
            )
            assert (
                cycle["il"]["resolved_factors"]["IL_dnfr_factor"]
                == defaults["IL_dnfr_factor"]
            )


def test_joint_band_uses_the_actual_um_floor_and_distinct_strict_il_minimum(cases):
    for record in cases.values():
        band = record["admission_band"]
        assert band["positive_epi_lower"] == band["um_min_epi_magnitude"] == F(0.05)
        assert band["um_min_capacity"] == F(0.01)
        assert band["il_min_epi"] == band["il_min_capacity"] == 0
        assert band["configured_epi_min"] == -1
        assert band["epi_upper"] == band["configured_epi_max"] == 1
        assert not band["configured_strict_preconditions_enabled"]
        assert (
            record["joint_domain_reference"]["epi_lower"] == band["positive_epi_lower"]
        )
        assert record["joint_domain_reference"]["epi_upper"] == band["epi_upper"]


@pytest.mark.parametrize("case", campaign.CASES)
def test_actual_words_and_per_target_history_span_both_cycles(cases, case):
    record = cases[case]
    word = record["word"]
    assert word["names"] == (
        "coupling",
        "coherence",
        "coupling",
        "coherence",
        "silence",
    )
    assert word["string_validator_passed"] and word["instance_validator_passed"]
    assert word["context"]["initial_epi_nonzero"]
    expected_history = ()
    for ordinal, cycle in enumerate(record["cycles"], start=1):
        assert cycle["ordinal"] == ordinal
        for key, glyph in (("um", "UM"), ("il", "IL")):
            event = cycle[key]
            assert all(
                event["before"]["state"]["glyph_history"][node] == expected_history
                for node in range(6)
            )
            expected_history += (glyph,)
            assert all(
                event["raw_state"]["state"]["glyph_history"][node] == expected_history
                for node in range(6)
            )
            assert event["stage_result"]["schedule"] == "two_phase_jacobi"
            assert event["stage_result"]["nodes_processed"] == 6
            assert all(
                a["allowed"] and a["candidate"] == glyph for a in event["admissions"]
            )
            assert all(m["glyph"] == glyph for m in event["actual_operator_metrics"])
    closure = record["closure_after_measurement"]
    assert all(item["allowed"] for item in closure["admissions"])
    assert all(
        closure["after"]["state"]["glyph_history"][node] == expected_history + ("SHA",)
        for node in range(6)
    )


def test_configured_um_and_strict_il_readiness_can_be_rechecked_from_actual_inputs(
    cases,
):
    for record in cases.values():
        for cycle in record["cycles"]:
            um_graph = _recorded_graph(cycle["um"]["before"])
            il_graph = _recorded_graph(cycle["il"]["before"])
            for item in cycle["independent_strict_um_readiness"]:
                assert item["passed"]
                assert validate_coupling(um_graph, item["node"]) is None
            for item in cycle["il"]["independent_strict_il_readiness"]:
                assert item["passed"]
                assert (
                    validate_coherence_strict(
                        il_graph, item["node"], emit_warnings=False
                    )
                    is None
                )
            for graph in (um_graph, il_graph):
                assert all(graph.nodes[node][ALIAS_EPI[0]] > 0 for node in graph)
                assert all(graph.nodes[node][ALIAS_VF[0]] == 1 for node in graph)
                assert all(
                    graph.nodes[node].get("_grammar_u2_debt", 0) == 0 for node in graph
                )


@pytest.mark.parametrize("case", campaign.CASES)
def test_phase_events_preserve_epi_and_unit_capacity_without_creating_edges(
    cases, case
):
    initial = cases[case]["initial_capture"]["snapshot"]
    for cycle in cases[case]["cycles"]:
        for key in ("um", "il"):
            event = cycle[key]
            assert all(event["prediction_matches_actual"].values())
            before = event["before_capture"]["snapshot"]
            for capture in (event["raw_capture"], event["after_capture"]):
                assert capture["snapshot"]["epi"] == before["epi"]
                assert capture["snapshot"]["capacity"] == (1,) * 6
                assert capture["snapshot"]["conductance"] == initial["conductance"]
                assert (
                    capture["snapshot"]["support_neighbors"]
                    == initial["support_neighbors"]
                )
            assert event["event_budget"]["epi_jump"] == (0,) * 6
            assert event["event_budget"]["variance_identity_residual"] == 0
            assert event["event_budget"]["dirichlet_identity_residual"] == 0
            assert event["raw_capture"]["phase"] == event["after_capture"]["phase"]
            assert event["after_capture"]["stored_pressure_residual"] == (0,) * 6
        assert cycle["um"]["independent_prediction"]["edges"] == ()


@pytest.mark.parametrize("case", campaign.CASES)
def test_each_cycle_uses_one_bound_default_interval_with_four_held_pressure_substeps(
    cases, case
):
    record = cases[case]
    previous_capture = record["initial_capture"]
    for index, cycle in enumerate(record["cycles"]):
        assert cycle["before_capture"] == previous_capture
        flow = cycle["flow"]
        assert flow["before"]["time"] == index * 0.25
        assert flow["raw_after_integrator"]["time"] == (index + 1) * 0.25
        evidence = flow["executor_evidence"]
        assert evidence["resolved_method"] == "euler"
        assert evidence["resolved_substeps"] == 4
        assert evidence["integrator_provenance_certified"] and evidence["gamma_is_none"]
        assert not evidence["extended_dynamics_requested"]
        assert not evidence["clipping_applied"]
        assert all(evidence["left_binding"].values()) and all(
            evidence["right_binding"].values()
        )
        assert evidence["duration"] == flow["duration"] == 0.25
        assert (
            evidence["captured_left"]["pressure"]
            == evidence["captured_right"]["pressure"]
        )
        assert all(flow["frozen_input_checks"].values())
        assert cycle["post_il_capture"]["phase"] == cycle["after_capture"]["phase"]
        exact = flow["regime_step_budget"]
        before = cycle["post_il_capture"]["snapshot"]
        held_prediction = tuple(
            x + F(1, 4) * nu * p
            for x, nu, p in zip(
                before["epi"],
                before["capacity"],
                before["stored_pressure"],
                strict=True,
            )
        )
        assert exact["support_budget"]["expected_epi"] == held_prediction
        assert exact["support_budget"]["state_defect"] == tuple(
            x - y
            for x, y in zip(
                cycle["after_capture"]["snapshot"]["epi"], held_prediction, strict=True
            )
        )
        assert (
            exact["mean_identity_residual"]
            == exact["relative_energy_budget"]["identity_residual"]
            == 0
        )
        previous_capture = cycle["after_capture"]
    assert record["final_capture"] == previous_capture
    assert record["final_before_closure"]["state"]["time"] == 0.5


def test_readonly_repeat_control_preserves_the_validator_disagreement(cases):
    for record in cases.values():
        control = record["repeated_um_control"]
        assert control["word"]["names"] == ("coupling", "coupling", "silence")
        assert not control["word"]["string_validator_passed"]
        assert control["word"]["instance_validator_passed"]
        assert control["materialized_state_preserved"]
        assert all(
            history == ("UM",) for history in control["history_at_probe"].values()
        )
        graph = _recorded_graph(record["cycles"][0]["um"]["after_refresh"])
        for node, recorded in enumerate(control["live_candidates_after_one_um"]):
            actual = validate_candidate(graph, node, "UM")
            assert recorded["candidate"] == actual.candidate == "UM"
            assert recorded["allowed"] == actual.allowed is True


def test_measured_joint_readouts_keep_binary64_pi_and_exact_reserve_accounting(cases):
    for record in cases.values():
        weight = record["joint_domain_reference"]["epi_phase_budget_weight"]
        readouts = [(record["initial_capture"], record["initial_joint_readout"])]
        for cycle in record["cycles"]:
            readouts.extend(
                (
                    (cycle["post_il_capture"], cycle["post_il_joint_readout"]),
                    (cycle["after_capture"], cycle["after_joint_readout"]),
                )
            )
        for capture, readout in readouts:
            assert readout["represented_pi_scale"] == F(math.pi)
            scaled = tuple(
                value / F(math.pi) for value in readout["phase"]["represented_lift"]
            )
            assert readout["phase_pi_represented"] == scaled
            diameter = max(scaled) - min(scaled)
            assert readout["phase_oscillation_pi"] == diameter
            epi = capture["snapshot"]["epi"]
            assert readout["lower_reserve"] == min(epi) - weight * diameter
            assert readout["upper_reserve"] == max(epi) + weight * diameter
            assert readout["inside_represented_prepared_phase_box"]
            assert readout["within_represented_reserve_band"]
            assert readout["unit_capacity"]
            assert min(readout["epi_lower_margins"]) > 0
            assert min(readout["epi_upper_margins"]) > 0
            assert min(readout["strict_il_epi_margins"]) > 0
            assert min(readout["strict_il_capacity_margins"]) > 0


def test_nonzero_controls_bind_the_conditional_nodal_step_with_signed_runtime_defects(
    cases,
):
    for key in campaign.CASES[1:]:
        record = cases[key]
        model = record["joint_domain_reference"]
        for cycle in record["cycles"]:
            conditional = cycle["conditional_transition"]
            assert conditional["status"] == "observed_conditional_transition"
            assert conditional["observed_phase_contraction_residual"] <= 0
            x, phase = (
                conditional["epi_before"],
                conditional["phase_after_pi_represented"],
            )
            h, c, we, wp = (
                model[name]
                for name in ("timestep", "capacity", "epi_weight", "phase_weight")
            )
            expected = tuple(
                x[i]
                + h
                * c
                * (
                    we * ((x[(i - 1) % 6] + x[(i + 1) % 6]) / 2 - x[i])
                    + wp * ((phase[(i - 1) % 6] + phase[(i + 1) % 6]) / 2 - phase[i])
                )
                for i in range(6)
            )
            assert conditional["observation"]["epi_after"] == expected
            actual = cycle["after_capture"]["snapshot"]["epi"]
            assert conditional["runtime_minus_conditional_epi"] == tuple(
                a - b for a, b in zip(actual, expected, strict=True)
            )
            assert conditional["defect_identity_residual"] == (0,) * 6
            assert conditional["runtime_minus_conditional_epi"] == tuple(
                a + b
                for a, b in zip(
                    conditional["conditional_pressure_realization_effect"],
                    conditional["actual_integrator_endpoint_defect"],
                    strict=True,
                )
            )


def test_null_numeric_phase_residue_is_not_promoted_to_an_exact_transition(cases):
    null = cases[("null", 0.0)]
    first = null["cycles"][0]
    assert first["before_joint_readout"]["phase_oscillation_pi"] == 0
    assert first["post_il_joint_readout"]["phase_oscillation_pi"] > 0
    conditional = first["conditional_transition"]
    assert conditional["status"] == "outside_conditional_transition"
    assert conditional["observed_phase_contraction_residual"] > 0
    assert "reason" in conditional
    for cycle in null["cycles"]:
        rejected = cycle["conditional_transition"]
        assert rejected["status"] == "outside_conditional_transition"
        assert rejected["observed_phase_contraction_residual"] > 0
        assert min(rejected["observed_phase_nesting_slacks"]) < 0
        assert "nesting or nonlinear contraction" in rejected["reason"]
    assert null["final_capture"]["snapshot"]["epi"] == (F(1, 2),) * 6


@pytest.mark.parametrize("case", campaign.CASES)
def test_terminal_sha_follows_measurement_and_leaves_the_unit_capacity_domain(
    cases, case
):
    record = cases[case]
    before, closure = (
        record["final_before_closure"]["state"],
        record["closure_after_measurement"],
    )
    after = closure["after"]["state"]
    assert all(item["passed"] for item in closure["independent_strict_sha_readiness"])
    assert before["time"] == after["time"] == 0.5
    assert before["epi"] == after["epi"]
    assert before["capacity"] == (1.0,) * 6
    expected_capacity = canonical_glyph_factor_defaults()["SHA_vf_factor"]
    assert after["capacity"] == (expected_capacity,) * 6
    assert 0 < expected_capacity < 1
    assert all(
        after["glyph_history"][node] == ("UM", "IL", "UM", "IL", "SHA")
        for node in range(6)
    )


@pytest.mark.parametrize(
    "case",
    (("k1", -(2.0**-12)), ("k3", -(2.0**-12)), ("null", 1.0), ("k2", 0.0), ("null", 0)),
)
def test_campaign_rejects_unregistered_preparations_before_execution(case):
    with pytest.raises(ValueError, match="three inherited"):
        campaign.run_c6_joint_case(*case)
