"""Finite original-preparation bridge scope and retained live phase observations."""

from fractions import Fraction as F

import pytest

from benchmarks.c6_winding_phase_tail_runtime import (
    TAIL_ENTRY_CYCLE,
    _declared_schedule,
    run_c6_phase_tail_runtime_bridge,
)


@pytest.fixture(scope="module")
def prefix():
    return run_c6_phase_tail_runtime_bridge(cycles=2)


@pytest.mark.parametrize("cycles", (True, 0, -1, 90, 2.0, "89"))
def test_runtime_budget_cannot_silently_change_its_declared_source_scope(cycles):
    with pytest.raises(ValueError, match="finite range"):
        _declared_schedule(cycles)


def test_full_phase_tail_budget_is_one_complete_word_with_only_terminal_silence():
    schedule, partitions = _declared_schedule(TAIL_ENTRY_CYCLE)
    assert schedule.operator_names == ("coupling", "coherence") * 89 + ("silence",)
    assert len(schedule.events) == 179 and len(partitions) == 89
    assert schedule.end_time == 22.25
    for cycle, partition in enumerate(partitions, 1):
        assert partition.parent_interval.index == 2 * cycle
        assert (
            tuple(segment.duration for segment in partition.segments) == (0.0625,) * 4
        )
        assert partition.parent_interval.exact_duration == F(1, 4)
    assert schedule.intervals[-1].duration == 0.0


def test_two_cycle_runtime_prefix_retains_actual_phase_and_unreset_carry(prefix):
    assert prefix["runtime_provenance_certified_at_capture"]
    assert prefix["phase_observations_retained_in_executor_seal"]
    assert prefix["event_count"] == 5 and prefix["flow_count"] == 8
    assert prefix["cycle_count"] == 2
    assert prefix["source"]["initial_state"]["epi"] == (0.5,) * 6
    assert prefix["source"]["initial_state"]["remainder"] == (F(0),) * 6
    assert not prefix["source"]["carry_imported_or_reset"]
    assert prefix["tail_entry_time"] == 0.25 and prefix["endpoint_time"] == 0.5
    assert not prefix["declared_phase_tail_reached"]
    assert not prefix["phase_projection_fixed_at_endpoint"]
    assert any(prefix["endpoint_before_SHA"]["remainder"])
    flows = tuple(flow for cycle in prefix["cycles"] for flow in cycle["flows"])
    for first, second in zip(flows, flows[1:]):
        assert first["step"]["after"] == second["step"]["before"]
    assert prefix["exact_nodal_area"] == prefix["exact_reconstructed_change"]
    assert prefix["exact_nodal_balance_residual"] == (F(0),) * 6


def test_unit_capacity_claim_stops_before_the_actual_terminal_sha(prefix):
    assert prefix["all_measured_flows_have_unit_capacity"]
    assert not prefix["unit_capacity_after_terminal_SHA"]
    terminal = prefix["terminal_silence"]
    assert terminal["capacity_before"] == (1.0,) * 6
    assert all(value < 1.0 for value in terminal["capacity_after"])
    assert (
        terminal["state_before"]
        == terminal["state_after"]
        == prefix["endpoint_before_SHA"]
    )
    assert (
        terminal["phase_before"] == terminal["phase_after"] == prefix["endpoint_phase"]
    )
    assert not prefix["historical_detached_endpoint_reachability_certified"]
    assert (
        not prefix["future_runtime_certified"]
        and not prefix["indefinite_trapping_certified"]
    )


def test_each_pressure_is_bound_to_its_captured_live_phase(prefix):
    for cycle in prefix["cycles"]:
        um, il, model = cycle["coupling"], cycle["coherence"], cycle["phase_projection"]
        assert um["phase_before"] == model["phase_before"]
        assert um["phase_after"] == il["phase_before"] == model["phase_after_coupling"]
        assert il["phase_after"] == model["phase_after_coherence"]
        for flow in cycle["flows"]:
            assert flow["phase_before"] == flow["phase_after"] == il["phase_after"]
            assert flow["fresh_pressure_matches"]
            assert flow["step"]["capacity"] == (1.0,) * 6


def test_tail_entry_profile_uses_the_actual_reconstructed_preflow_state(prefix):
    from tnfr.physics._cycle_algebra import laplacian_action

    state, profile, bound = (
        prefix["tail_entry_after_last_IL"],
        prefix["tail_entry_profile"],
        prefix["tail_entry_closure"],
    )
    exact = tuple(
        F(value) + carry
        for value, carry in zip(state["epi"], state["remainder"], strict=True)
    )
    mean = sum(exact) / 6
    errors = tuple(
        value - mean - z
        for value, z in zip(exact, profile["relative_profile"], strict=True)
    )
    assert sum(value * value for value in errors) <= bound["energy_bound"]
    assert tuple(
        profile["epi_weight"] * value
        for value in laplacian_action(profile["relative_profile"])
    ) == tuple(value - profile["mean_drift"] for value in profile["forcing"])
