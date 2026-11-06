"""Changing canonical support, refreshed physical flow and exact budget defects."""

from fractions import Fraction

import pytest

from benchmarks.cycle_support_dynamics import CASES, run_cycle_support_case
from tnfr.utils.numeric import angle_diff


@pytest.fixture(scope="module")
def cases():
    # Reuse one bounded 24-segment run across the independent assertions.
    return {case: run_cycle_support_case(case) for case in CASES}


@pytest.mark.parametrize("name", CASES)
def test_executor_calls_preserve_scope_and_live_continuity(cases, name):
    case = cases[name]
    previous = case["initial"]
    assert case["physical_elapsed_time"] == 2.0
    for index, cycle in enumerate(case["cycles"]):
        assert cycle["before"] == previous
        assert cycle["before"]["time"] == index / 2
        assert cycle["after"]["time"] == (index + 1) / 2
        assert cycle["whole_call_graph_state_atomic"]
        assert not cycle["future_execution_certified"]
        assert not cycle["intermediate_phase_observation"]
        assert len(cycle["segments"]) == 2
        assert len(cycle["boundaries"]) == cycle["physical_pressure_refresh_calls"] == 3
        for state in (cycle["before"], cycle["after"]):
            assert state["winding"]["winding"] == 1
            assert state["winding"]["minimum_u3_margin"] > 0.0
            assert state["strict_semicircle_margin"] > 0.0
            assert state["edge_attributes"] == case["initial"]["edge_attributes"]
            assert set(state["normalized_channel_weights"]) == {
                "epi",
                "vf",
                "phase",
                "topo",
            }
        for boundary in cycle["boundaries"]:
            assert boundary["pressure_only_refresh"]
            assert boundary["phase_preserved_during_refresh"]
            assert boundary["capacity_preserved_during_refresh"]
            assert boundary["node_support_preserved"]
            assert boundary["edge_state_preserved"]
        previous = cycle["after"]


@pytest.mark.parametrize("name", CASES)
def test_actual_euler_defects_and_support_energy_budget_are_not_zeroed(cases, name):
    case = cases[name]
    assert case["exact_multicall_telescope_residual"] == 0
    observed_rounding = []
    for cycle in case["cycles"]:
        assert cycle["exact_energy_telescope_residual"] == 0
        for segment in cycle["segments"]:
            assert segment["exact_duration"] == Fraction(1, 4)
            assert segment["method"] == "euler"
            assert segment["gamma_is_none"]
            assert not segment["clipping_applied"]
            assert not segment["extended_dynamics_requested"]
            assert segment["capacity_held_during_flow"]
            assert segment["reference"]["identity_residual"] == 0
            assert segment["reference"]["convex_step_condition"]
            assert set(segment["exact_endpoint_decomposition_residual"]) == {
                Fraction(0)
            }
            pressure_error = segment["exact_pressure_realization_residual"]
            held_error = segment["exact_held_input_euler_residual"]
            assert max(map(abs, pressure_error)) < Fraction(1, 10**12)
            assert max(map(abs, held_error)) < Fraction(1, 10**13)
            observed_rounding.extend(segment["exact_held_input_euler_residual"])
            assert (
                segment["actual_energy_change"]
                == segment["reference"]["energy_change"]
                + segment["exact_energy_realization_residual"]
            )
    assert any(value != 0 for value in observed_rounding)


def test_actual_um_sha_resets_change_both_support_channels_before_epi_flow(cases):
    case = cases["evolving_support"]
    preparation = case["capacity_preparation"]
    assert preparation["actual_operator_history"] == ("SHA",)
    assert preparation["before"]["epi"] == preparation["after"]["epi"] == (0.5,) * 8
    assert preparation["after"]["capacity"][0] < 1.0
    assert preparation["after"]["capacity"][1:] == (1.0,) * 7
    assert case["final"]["phase"] != case["initial"]["phase"]
    assert case["final"]["capacity"] != case["initial"]["capacity"]
    assert case["final"]["capacity_range"] < case["initial"]["capacity_range"]
    assert case["final"]["epi"] != case["initial"]["epi"]
    assert case["final"]["gap_spread"] < case["initial"]["gap_spread"]
    for cycle in case["cycles"]:
        assert cycle["schedule_word"] == ("coupling", "silence")
        coupling, silence = cycle["events"]
        assert coupling["operator"] == "coupling"
        assert silence["operator"] == "silence"
        assert (
            coupling["left"]["epi"]
            == coupling["right"]["epi"]
            == silence["right"]["epi"]
        )
        assert coupling["left"]["capacity"] != coupling["right"]["capacity"]
        assert silence["left"]["capacity"] != silence["right"]["capacity"]
        assert coupling["left"]["pressure"] != coupling["right"]["pressure"]
        reset = cycle["reset"]
        assert reset["reference"]["identity_residual"] == 0
        assert set(reset["exact_epi_residual"]) == {Fraction(0)}
        assert max(map(abs, reset["exact_capacity_residual"])) < Fraction(1, 10**13)
        phase_error = reset["exact_phase_coordinate_residual"]
        assert max(map(abs, phase_error)) < Fraction(1, 10**13)
        assert (
            reset["actual_energy_change"]
            == reset["reference"]["energy_change"]
            + reset["exact_energy_realization_residual"]
        )
    expected = ("UM", "SHA") * 4
    assert case["final"]["glyph_history"][0] == ("SHA",) + expected
    assert all(history == expected for history in case["final"]["glyph_history"][1:])


def test_held_support_control_has_equal_physical_time_and_different_endpoints(cases):
    held, evolving = cases["held_support"], cases["evolving_support"]
    assert held["initial"] == evolving["initial"]
    assert held["final"]["time"] == evolving["final"]["time"] == 2.0
    assert held["final"]["capacity"] == held["initial"]["capacity"]
    assert held["final"]["phase"] == held["initial"]["phase"]
    assert held["final"]["epi"] != held["initial"]["epi"]
    assert held["final"]["epi"] != evolving["final"]["epi"]
    assert (
        held["final"]["reference"]["dirichlet_energy"]
        < held["initial"]["reference"]["dirichlet_energy"]
    )
    assert held["final"]["ordinary_epi_dirichlet_energy"] > 0.0
    for cycle in held["cycles"]:
        assert cycle["events"] == []
        assert cycle["reset"] is None
    assert held["node_reorganization_clocks"] == tuple(
        2 * Fraction.from_float(nu) for nu in held["initial"]["capacity"]
    )


def test_uniform_capacity_bump_dissipates_with_a_shorter_reorganization_clock(cases):
    case = cases["uniform_bump"]
    assert case["capacity_preparation"]["actual_operator_history"] == ()
    assert case["initial"]["epi"] == (1.0,) + (0.5,) * 7
    assert case["final"]["epi_range"] < case["initial"]["epi_range"]
    assert (
        case["final"]["ordinary_epi_dirichlet_energy"]
        < case["initial"]["ordinary_epi_dirichlet_energy"]
    )
    assert len(set(case["final"]["capacity"])) == 1
    assert len(set(case["node_reorganization_clocks"])) == 1
    assert 0 < case["node_reorganization_clocks"][0] < 2
    for cycle in case["cycles"]:
        assert len(set(cycle["after"]["capacity"])) == 1
        assert (
            max(
                abs(angle_diff(a, b))
                for a, b in zip(cycle["before"]["phase"], cycle["after"]["phase"])
            )
            < 1e-13
        )


def test_unknown_preparation_is_rejected_before_execution():
    with pytest.raises(ValueError, match="case must be one of"):
        run_cycle_support_case("unspecified")
