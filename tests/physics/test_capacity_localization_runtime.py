"""Capacity-gradient balance is distinct from winding and self-localization."""

from fractions import Fraction

import pytest

from benchmarks.capacity_localization import (
    run_capacity_release_case, run_fixed_capacity_case,
)


def _assert_finite_physical_scope(flow, winding):
    initial, final = flow["initial"], flow["final"]
    assert initial["winding"]["winding"] == final["winding"]["winding"] == winding
    assert final["phase"] == initial["phase"]
    assert final["capacity"] == initial["capacity"]
    assert final["edge_support"] == initial["edge_support"]
    assert initial["time"] == 0.0
    assert final["time"] == flow["duration"] == 4.0
    assert len(flow["segments"]) == 16
    assert len(flow["boundaries"]) == flow["physical_pressure_refresh_calls"] == 17
    assert abs(flow["exact_fixed_metric_total_drift"]) < Fraction(1, 10**12)
    for segment in flow["segments"]:
        assert segment["method"] == "euler"
        assert segment["clipping_applied"] is False
        assert segment["gamma_is_none"] is True
        residual = segment["exact_held_input_euler_residual"]
        assert max(map(abs, residual)) < Fraction(1, 10**13)
    for boundary in flow["boundaries"]:
        assert boundary["pressure_only_refresh"]
        assert boundary["capacity"] == initial["capacity"]
        defect = boundary["channel_profile"]["exact_runtime_minus_model_pressure"]
        assert max(map(abs, defect)) < Fraction(1, 10**13)


@pytest.mark.parametrize("count", [8, 16])
def test_regular_winding_does_not_retain_an_epi_bump_under_physical_flow(count):
    result = run_fixed_capacity_case(count, "epi_bump")
    flow = result["flow"]
    _assert_finite_physical_scope(flow, 1)
    assert flow["initial"]["epi"][0] == 1.0
    assert 0.7 < flow["final"]["epi"][0] < 0.8
    assert flow["final"]["epi"][1] > 0.5
    assert (
        flow["final_error_energy_to_fixed_profile"]
        < flow["initial_error_energy_to_fixed_profile"]
    )
    assert flow["final"]["core_minus_background_mean"] < 0.5
    reference = flow["initial"]["reference"]
    assert len(set(reference["equilibrium_epi"])) == 1


@pytest.mark.parametrize("count", [8, 16])
def test_one_canonical_silence_creates_capacity_pressure_and_an_epi_profile(count):
    result = run_fixed_capacity_case(count, "single_silence")
    preparation, flow = result["preparation"], result["flow"]
    _assert_finite_physical_scope(flow, 1)
    assert preparation["actual_operator_history"] == ("SHA",)
    assert preparation["before"]["epi"] == preparation["after"]["epi"] == (0.5,) * count
    assert preparation["after"]["capacity"][0] < 1.0
    assert preparation["after"]["capacity"][1:] == (1.0,) * (count - 1)
    assert preparation["after"]["pressure"][0] > 0.0
    initial_reference = flow["initial"]["reference"]
    target = initial_reference["equilibrium_epi"]
    assert target[0] > target[1]
    assert len(set(target[1:])) == 1
    assert 0.5 < flow["final"]["epi"][0] < float(target[0])
    assert (
        flow["final_error_energy_to_fixed_profile"]
        < flow["initial_error_energy_to_fixed_profile"]
    )
    assert flow["final"]["core_minus_background_mean"] > 0.0
    # EPI contrast grows even as deviation from the shifted balance decreases.
    assert flow["initial"]["reference"]["epi_dirichlet_energy"] == 0
    assert flow["final"]["reference"]["epi_dirichlet_energy"] > 0


@pytest.mark.parametrize("preparation", ["epi_bump", "single_silence"])
def test_zero_winding_has_the_same_prepared_profile_evolution_on_c8(preparation):
    nonzero = run_fixed_capacity_case(8, preparation)["flow"]
    zero = run_fixed_capacity_case(8, preparation, winding=0)["flow"]
    _assert_finite_physical_scope(zero, 0)
    assert zero["initial"]["epi"] == nonzero["initial"]["epi"]
    assert zero["final"]["epi"] == nonzero["final"]["epi"]


def test_actual_capacity_writing_operators_release_a_prepared_balance():
    result = run_capacity_release_case()
    initial = result["balanced_initial"]
    coupling, silence = result["canonical_release_events"]
    assert max(map(abs, initial["pressure"])) < 1e-14
    assert coupling["operator"] == "coupling"
    assert silence["operator"] == "silence"
    assert coupling["epi"] == silence["epi"] == initial["epi"]
    assert coupling["capacity"] != initial["capacity"]
    assert silence["capacity"] != coupling["capacity"]
    assert max(map(abs, coupling["pressure"])) > 1e-5
    assert max(map(abs, silence["pressure"])) > 1e-5
    assert (
        coupling["edge_support"] == silence["edge_support"]
        == initial["edge_support"]
    )
    assert coupling["winding"]["winding"] == silence["winding"]["winding"] == 1
    flow = result["released_flow"]
    _assert_finite_physical_scope(flow, 1)
    assert flow["final"]["epi"] != flow["initial"]["epi"]
    assert (
        flow["final_error_energy_to_fixed_profile"]
        < flow["initial_error_energy_to_fixed_profile"]
    )
