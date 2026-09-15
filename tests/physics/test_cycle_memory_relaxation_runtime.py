"""Delayed memory is distinct from the Recursivity advisory and frozen flow."""

from fractions import Fraction

import pytest

from benchmarks.cycle_memory_relaxation import (
    CYCLE_COUNT, INITIAL_EPI, WORD, run_cycle_memory_case,
)
from tnfr.utils.numeric import angle_diff


@pytest.fixture(scope="module")
def cases():
    # The bounded experiment has sixteen Euler segments, shared by these checks.
    return {memory: run_cycle_memory_case(memory=memory) for memory in (True, False)}


@pytest.mark.parametrize("memory", [True, False])
def test_admitted_words_keep_capacity_positive_and_clock_advancing(cases, memory):
    case = cases[memory]
    assert case["word"] == WORD == ("coupling", "coherence", "recursivity")
    assert case["initial"]["epi"] == INITIAL_EPI
    assert case["physical_elapsed_time"] == 2.0
    assert case["node_reorganization_clocks"] == (Fraction(2),) * 8
    assert case["edge_attributes_before"] == case["edge_attributes_after"]
    assert case["actual_glyph_history"] == (("UM", "IL", "REMESH") * 4,) * 8
    previous = case["initial"]
    for index, cycle in enumerate(case["cycles"]):
        assert cycle["before"] == previous
        assert cycle["before"]["time"] == index / 2
        assert cycle["after"]["time"] == (index + 1) / 2
        for state in (cycle["before"], cycle["pre_remesh"], cycle["after"]):
            assert state["capacity"] == (1.0,) * 8
            readout = state["phase_readout"]
            assert readout["winding"]["winding"] == 1
            assert readout["winding"]["minimum_u3_margin"] > 0.7
            assert readout["maximum_wrapped_drift_from_regular_twist"] < 1e-13
        previous = cycle["after"]
    assert len(case["cycles"]) == CYCLE_COUNT == 4


@pytest.mark.parametrize("memory", [True, False])
def test_physical_flow_has_explicit_channel_and_binary64_defects(cases, memory):
    for cycle in cases[memory]["cycles"]:
        flow = cycle["flow"]
        assert len(flow["segments"]) == 2
        assert len(flow["boundaries"]) == flow["physical_pressure_refresh_calls"] == 3
        for boundary in flow["boundaries"]:
            assert boundary["pressure_only_refresh"]
            assert boundary["phase_preserved_during_refresh"]
            assert boundary["capacity_preserved_during_refresh"]
            error = boundary["exact_runtime_minus_regular_twist_pressure"]
            assert max(map(abs, error)) < Fraction(1, 10**13)
        for segment in flow["segments"]:
            assert segment["method"] == "euler"
            assert segment["gamma_is_none"]
            assert not segment["clipping_applied"]
            assert not segment["extended_dynamics_requested"]
            assert segment["capacity_held"]
            assert segment["reference"].convex_step_condition
            assert segment["reference"].identity_residual == 0
            assert segment["exact_endpoint_decomposition_residual"] == (0,) * 8
            assert max(map(abs, segment["exact_held_input_euler_residual"])) < 1e-13


def test_advisory_alone_neither_echoes_epi_nor_advances_delayed_history(cases):
    control = cases[False]
    assert control["final_history"] == control["initial_history"]
    for cycle in control["cycles"]:
        assert cycle["whole_call_graph_state_atomic"]
        assert cycle["remesh"] is None
        for event, name in zip(cycle["flow"]["events"], WORD, strict=True):
            assert event["operator"] == name
            assert event["epi_before"] == event["epi_after"]
            assert event["capacity_before"] == event["capacity_after"] == (1.0,) * 8
    assert control["final"]["epi"] != control["initial"]["epi"]


def test_memory_executor_samples_pre_map_once_and_reads_previous_sample(cases):
    case = cases[True]
    prior_history = case["initial_history"]
    for index, cycle in enumerate(case["cycles"]):
        history, remesh = cycle["history"], cycle["remesh"]
        assert history["canonical_transition_certified"]
        assert history["schedule_left_history_unchanged"]
        assert history["incoming"] == prior_history
        assert history["outgoing"] == prior_history + (history["appended"],)
        assert len(history["outgoing"]) == index + 2
        assert history["appended"] == tuple(
            Fraction.from_float(x) for x in cycle["pre_remesh"]["epi"]
        )
        assert history["selected_local"] == history["selected_global"]
        assert history["selected_local"] == prior_history[-1]
        assert history["active_newest_first"] == tuple(
            reversed(history["outgoing"][-2:])
        )
        assert remesh["applied"]
        assert remesh["alpha"] == 0.5
        assert remesh["tau_local"] == remesh["tau_global"] == 1
        assert remesh["same_time_epi_boundary_recorded"]
        assert remesh["post_map_pressure_refresh_calls"] == 1
        assert remesh["capacity_preserved"] and remesh["phase_preserved"]
        output = tuple(Fraction.from_float(x) for x in cycle["after"]["epi"])
        assert output != history["appended"]
        prior_history = history["outgoing"]
    assert case["final_history"] == prior_history
    assert case["pre_remesh_companion_transitions_observed"] == 3


def test_each_actual_echo_is_bound_to_the_shared_exact_history_map(cases):
    rounding_seen = []
    for cycle in cases[True]["cycles"]:
        remesh = cycle["remesh"]
        exact = remesh["exact_reference_transition"]
        assert exact.transition_observation_certified
        assert exact.exact_dissipation_identity_certified
        assert exact.exact_history == cycle["history"]["active_newest_first"]
        assert remesh["exact_residual_decomposition"] == (0,) * 8
        assert remesh["exact_clipping_residual"] == (0,) * 8
        assert not remesh["clipping_intervened"]
        rounding_seen.extend(remesh["exact_rounding_residual"])
        assert max(map(abs, remesh["exact_total_residual"])) < Fraction(1, 10**13)
        assert cycle["after"]["epi"][0] > cycle["pre_remesh"]["epi"][0]
    assert any(error != 0 for error in rounding_seen)


def test_finite_memory_echo_slows_this_bump_without_stopping_the_nodal_clock(cases):
    memory, control = cases[True], cases[False]
    assert memory["initial"] == control["initial"]
    assert memory["physical_elapsed_time"] == control["physical_elapsed_time"]
    assert memory["node_reorganization_clocks"] == control["node_reorganization_clocks"]
    assert control["final"]["epi"][0] < memory["final"]["epi"][0] < INITIAL_EPI[0]
    initial_energy = memory["initial"]["exact_disagreement_energy"]
    memory_energy = memory["final"]["exact_disagreement_energy"]
    control_energy = control["final"]["exact_disagreement_energy"]
    assert control_energy < memory_energy < initial_energy
    for case in cases.values():
        assert abs(case["final"]["exact_mean"] - case["initial"]["exact_mean"]) < 1e-13
        assert max(
            abs(angle_diff(a, b)) for a, b in zip(
                case["initial"]["phase_readout"]["phase"],
                case["final"]["phase_readout"]["phase"], strict=True,
            )
        ) < 1e-13


def test_causal_execution_and_detached_contraction_theorem_have_separate_scope(cases):
    case = cases[True]
    provenance = case["execution_provenance"]
    assert provenance["causal_order_certified"]
    assert provenance["same_graph_provenance_certified"]
    assert provenance["whole_sequence_graph_state_atomic"]
    assert provenance["exact_recorded_boundary_continuity_certified"]
    assert provenance["runtime_telescope"] is None
    assert not provenance["runtime_global_gain_certified"]
    assert not provenance["future_stability_certified"]
    model = case["exact_reference"]
    assert model.epi_weight == Fraction.from_float(
        case["normalized_channel_weights"]["epi"]
    )
    assert 0 < model.euler_energy_gain_upper_bound < 1
    assert 0 < model.continuous_energy_gain_upper_bound < 1
    assert model.euler_policy.exact_uniform_normalized_block_margin_lower_bound > 0
    assert model.euler_policy.universal_block_horizon == 2


def test_three_observed_companion_transitions_include_two_complete_blocks(cases):
    observed = cases[True]["finite_envelope_observations"]
    assert len(observed["adjacent_transitions"]) == 3
    assert len(observed["complete_blocks"]) == 2
    assert observed["block_horizon"] == 2
    assert not observed["runtime_contraction_certified"]
    assert observed["exact_reference_q"] == (
        cases[True]["exact_reference"].euler_energy_gain_upper_bound
    )
    for transition in observed["adjacent_transitions"]:
        assert all(value >= 0 for value in transition["signed_observed_envelope_slack"])
    for block in observed["complete_blocks"]:
        assert block["signed_observed_block_slack"] > 0
    assert cases[False]["finite_envelope_observations"] is None
