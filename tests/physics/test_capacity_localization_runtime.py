"""Capacity-gradient balance is distinct from winding and self-localization."""

from fractions import Fraction

import pytest

from benchmarks.capacity_localization import (
    run_capacity_release_case,
    run_fixed_capacity_case,
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


def _assert_common_model_with_recorded_euler_defects(zero, nonzero):
    """Propagate the observed channel/rounding defects with no chosen tolerance."""
    reference = zero["initial"]["reference"]
    assert reference == nonzero["initial"]["reference"]
    assert (
        zero["initial"]["normalized_channel_weights"]
        == nonzero["initial"]["normalized_channel_weights"]
    )
    capacity, laplacian = reference["capacity"], reference["laplacian"]
    count = len(capacity)
    for flow in (zero, nonzero):
        assert flow["boundaries"][0]["epi"] == flow["initial"]["epi"]
        assert flow["boundaries"][-1]["epi"] == flow["final"]["epi"]

    difference = (Fraction(0),) * count
    bound = Fraction(0)
    for index, (zero_segment, nonzero_segment) in enumerate(
        zip(zero["segments"], nonzero["segments"], strict=True)
    ):
        zero_left, zero_right = zero["boundaries"][index : index + 2]
        nonzero_left, nonzero_right = nonzero["boundaries"][index : index + 2]
        dt = zero_right["exact_time"] - zero_left["exact_time"]
        assert dt == nonzero_right["exact_time"] - nonzero_left["exact_time"]
        assert dt == Fraction(zero_segment["duration"])
        assert dt == Fraction(nonzero_segment["duration"])
        # The common held-capacity affine model has stochastic Euler matrix M.
        # Hence ||M*v||_inf <= ||v||_inf, proved here with exact entries.
        matrix = tuple(
            tuple(
                Fraction(i == j) - dt * nu * reference["epi_weight"] * entry
                for j, entry in enumerate(row)
            )
            for i, (nu, row) in enumerate(zip(capacity, laplacian, strict=True))
        )
        assert all(entry >= 0 for row in matrix for entry in row)
        assert all(sum(row) == 1 for row in matrix)
        local_defects = []
        for left, segment in (
            (zero_left, zero_segment),
            (nonzero_left, nonzero_segment),
        ):
            pressure_defect = left["channel_profile"][
                "exact_runtime_minus_model_pressure"
            ]
            euler_residual = segment["exact_held_input_euler_residual"]
            local_defects.append(
                tuple(
                    dt * nu * pressure + rounding
                    for nu, pressure, rounding in zip(
                        capacity, pressure_defect, euler_residual, strict=True
                    )
                )
            )
        defect_difference = tuple(a - b for a, b in zip(*local_defects, strict=True))
        predicted_difference = tuple(
            sum(entry * value for entry, value in zip(row, difference, strict=True))
            + defect
            for row, defect in zip(matrix, defect_difference, strict=True)
        )
        difference = tuple(
            Fraction(a) - Fraction(b)
            for a, b in zip(zero_right["epi"], nonzero_right["epi"], strict=True)
        )
        # Every measured discrepancy must be explained by the saved defects;
        # the bound concerns these finite observations, not future rounding.
        assert difference == predicted_difference
        bound += max(map(abs, defect_difference))
        assert max(map(abs, difference)) <= bound


@pytest.mark.parametrize("preparation", ["epi_bump", "single_silence"])
def test_zero_winding_shares_prepared_c8_model_with_recorded_runtime_defects(
    preparation,
):
    nonzero = run_fixed_capacity_case(8, preparation)["flow"]
    zero = run_fixed_capacity_case(8, preparation, winding=0)["flow"]
    _assert_finite_physical_scope(zero, 0)
    _assert_finite_physical_scope(nonzero, 1)
    assert zero["initial"]["epi"] == nonzero["initial"]["epi"]
    _assert_common_model_with_recorded_euler_defects(zero, nonzero)


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
        coupling["edge_support"] == silence["edge_support"] == initial["edge_support"]
    )
    assert coupling["winding"]["winding"] == silence["winding"]["winding"] == 1
    flow = result["released_flow"]
    _assert_finite_physical_scope(flow, 1)
    assert flow["final"]["epi"] != flow["initial"]["epi"]
    assert (
        flow["final_error_energy_to_fixed_profile"]
        < flow["initial_error_energy_to_fixed_profile"]
    )
