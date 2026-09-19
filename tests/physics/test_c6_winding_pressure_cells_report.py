"""Retained k1 drift survives exactly balanced pressures inside its trace box."""

import json
import math
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_cells as campaign

INPUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/research/c6_winding_phase_kernel.json"
)


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("the retained local B20 comparison is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_pressure_cells(parent)


def test_detached_audit_uses_shared_validation_once_per_case_and_never_runs_graphs(
    parent, monkeypatch
):
    from benchmarks import c6_winding_joint_domain as producer
    from benchmarks import c6_winding_phase_kernel as comparison

    def forbidden(*args, **kwargs):
        raise AssertionError("B21 must use retained inputs without new graph execution")

    monkeypatch.setattr(producer, "run_c6_joint_case", forbidden)
    monkeypatch.setattr(comparison, "compare_c6_phase_kernel", forbidden)
    original = deepcopy(parent)
    actual = campaign.analyze_c6_defect_case
    calls = []

    def tracked(case):
        calls.append(case["mode"])
        return actual(case)

    monkeypatch.setattr(campaign, "analyze_c6_defect_case", tracked)
    campaign.analyze_c6_pressure_cells(parent)
    assert calls == ["null", "k1", "k3"]
    assert parent == original


def test_all_six_boxes_and_numeric_witnesses_bind_complete_recorded_traces(
    parent, report
):
    for source, result in zip(parent["current_cases"], report["cases"], strict=True):
        for cycle, observed in zip(source["cycles"], result["cycles"], strict=True):
            assert all(observed["endpoint_bindings"].values())
            box = observed["pressure_box"]
            assert tuple(map(F, box["flow"]["epi"])) == tuple(
                map(F, cycle["flow"]["before"]["epi"])
            )
            assert tuple(map(F, box["flow"]["endpoint"])) == tuple(
                map(F, cycle["flow"]["raw_after_integrator"]["epi"])
            )
            trace = tuple(step["after"] for step in box["flow"]["substeps"])
            for key in ("zero_sum_witness", "opposite_pair_witness"):
                witness = observed[key]
                assert witness["status"] == "feasible"
                assert witness["substeps"] == trace
                assert witness["same_trace"] and witness["same_mean"]
                assert witness["exact_zero_sum"] and witness["pressure_sum"] == 0
                assert sum(map(F, witness["pressure"])) == 0
                assert witness["minimum_pressure_margin"] > 0
                for value, cell in zip(
                    witness["pressure"], box["coordinates"], strict=True
                ):
                    assert cell["first_pressure"] <= value <= cell["last_pressure"]
                assert witness["canonical_pressure_generated"] is False
            assert observed["opposite_pair_witness"]["pressure_pair_sums"] == (0, 0, 0)


def test_k1_one_ulp_zero_sum_corrections_preserve_the_same_mean_defect(report):
    k1 = report["cases"][1]
    for ordinal, node in ((0, 0), (1, 1)):
        cycle = k1["cycles"][ordinal]
        source = cycle["pressure_box"]["flow"]["pressure"]
        witness = cycle["zero_sum_witness"]
        assert witness["adjusted_node"] == node
        assert witness["pressure"][node] == math.nextafter(source[node], -math.inf)
        assert all(witness["pressure"][i] == source[i] for i in range(6) if i != node)
        assert witness["max_pressure_change"] == F(1, 2 ** (68 + ordinal))
        assert witness["mean_budget"]["held_pressure_mean_effect"] == 0
        assert witness["mean_budget"]["scaling_mean_effect"] == 0
    changes = tuple(
        cycle["zero_sum_witness"]["mean_budget"]["actual_mean_change"]
        for cycle in k1["cycles"]
    )
    assert changes == (F(1, 3 * 2**52), 0)


def test_k1_opposite_completion_has_two_positive_binade_biases_and_one_zero(
    parent, report
):
    first = report["cases"][1]["cycles"][0]
    source = tuple(
        map(
            float,
            map(
                F, parent["current_cases"][1]["cycles"][0]["flow"]["before"]["pressure"]
            ),
        )
    )
    witness = first["opposite_pair_witness"]
    assert witness["pressure"][:3] == source[:3]
    assert witness["pressure"][3:] == tuple(-p for p in source[:3])
    fractional_parts = []
    for value in source[:3]:
        ratio = abs(F(value)) / 16 * 2**53
        fractional_parts.append(ratio - ratio.numerator // ratio.denominator)
    assert tuple(fractional_parts) == (
        F(8875, 262144),
        F(153711, 262144),
        F(314293, 524288),
    )
    assert fractional_parts[0] < F(1, 4)
    assert all(F(1, 2) < value < F(3, 4) for value in fractional_parts[1:])
    for step in witness["substep_pair_sums"]:
        assert step["change"] == (0, F(1, 2**54), F(1, 2**54))
    for step in report["cases"][1]["cycles"][1]["opposite_pair_witness"][
        "substep_pair_sums"
    ]:
        assert step["change"] == (0, 0, 0)


def test_signed_accounting_keeps_every_cycle_and_zero_net_drift_separate(report):
    sums = []
    for case in report["cases"]:
        total = F(0)
        for cycle in case["cycles"]:
            for budget in (
                cycle["mean_budget"],
                cycle["zero_sum_witness"]["mean_budget"],
                cycle["opposite_pair_witness"]["mean_budget"],
            ):
                assert budget["actual_mean_change"] == (
                    budget["held_pressure_mean_effect"]
                    + budget["scaling_mean_effect"]
                    + budget["addition_mean_effect"]
                )
                assert budget["identity_residual"] == 0
            total += cycle["mean_budget"]["actual_mean_change"]
        sums.append(total)
    assert sums == [0, F(1, 3 * 2**52), 0]
    assert all(
        cycle["mean_budget"]["actual_mean_change"] != 0
        for cycle in report["cases"][2]["cycles"]
    )


def test_scope_never_promotes_pressure_feasibility_to_production_or_future(report):
    for flag in (
        "runtime_executed",
        "live_provenance_certified",
        "future_bounds_verified",
        "canonical_pressure_witnesses_generated",
    ):
        assert report[flag] is False
    assert "not maximal endpoint classes" in report["scope"]
    json.dumps(campaign._payload(report), allow_nan=False)


@pytest.mark.parametrize("change", ("seed", "horizon", "order", "state"))
def test_changed_parent_controls_or_state_are_rejected(parent, change):
    changed = deepcopy(parent)
    if change == "seed":
        changed["manifest"]["seed"] = 18
    elif change == "horizon":
        changed["cycle_count_per_case"] = 3
    elif change == "order":
        changed["current_cases"].reverse()
    else:
        changed["current_cases"][1]["cycles"][0]["after_capture"]["snapshot"]["epi"][
            0
        ] = "3/4"
    with pytest.raises(ValueError):
        campaign.analyze_c6_pressure_cells(changed)


def test_infeasible_pair_boxes_and_absent_single_coordinate_witness_are_distinct():
    box = campaign.derive_binary64_quarter_pressure_box(
        epi=(0.5,) * 6, pressure=(0.125,) * 6
    )
    paired = campaign._opposite_pair_witness(box)
    assert paired["status"] == "infeasible"
    assert all(not interval["feasible"] for interval in paired["pair_intervals"])
    single = campaign._zero_sum_witness(box)
    assert single["status"] == "no_single_coordinate_witness"
    assert single["general_zero_sum_feasibility_decided"] is False
