"""Matched live branch budgets separate generated input from accumulation."""

import json
from fractions import Fraction as F

import pytest

from benchmarks import c6_winding_remainder_runtime as owner
from benchmarks.thol_pressure_feedback import _payload


@pytest.fixture(scope="module")
def report():
    return owner.run_c6_remainder_runtime_comparison()


def test_fixed_inherited_cases_share_an_explicit_pressure_refresh_mesh(report):
    assert (
        tuple((case["mode"], float(case["epsilon"])) for case in report["cases"])
        == owner.CASES
    )
    assert report["runtime_executed"] and report["pressure_refreshed_on_live_epi"]
    assert report["case_count"] == 3
    assert report["cycle_count_per_branch"] == 2
    assert report["physical_segments_per_branch"] == 8
    for case in report["cases"]:
        assert case["schedule"]["operator_names"] == owner.WORD
        assert case["final_time"] == 0.5
        assert len(case["carried"]["flows"]) == len(case["ordinary"]["flows"]) == 8
        assert case["ordinary"]["event_names"] == owner.WORD
        assert len(case["physical_partitions"]) == 2


def test_every_live_flow_obeys_an_independent_exact_reconstructed_balance(report):
    for case in report["cases"]:
        area = [F(0)] * 6
        carried = case["carried"]
        for flow, prefix in zip(
            carried["flows"], carried["prefix_budget"]["prefixes"], strict=True
        ):
            step = flow["step"]
            for i in range(6):
                assert step["pressure"][i] == flow["before_snapshot"]["delta_nfr"][i]
                area[i] += (
                    F(step["timestep"])
                    * F(step["capacity"][i])
                    * F(step["pressure"][i])
                )
                expected = F(1, 2) + area[i]
                assert (
                    F(step["after"]["epi"][i]) + step["after"]["remainder"][i]
                    == expected
                )
            assert prefix["cumulative_nodal_area"] == tuple(area)
            assert prefix["mean_nodal_area"] == sum(area) / 6
            assert prefix["mean_identity_residual"] == 0
        assert carried["reconstructed_mean_change"] == sum(area) / 6
        assert carried["runtime_provenance_certified"]


def test_mean_difference_is_split_between_actual_source_and_executor(report):
    for case in report["cases"]:
        carried, ordinary, comparison = (
            case["carried"],
            case["ordinary"],
            case["comparison"],
        )
        assert (
            comparison["visible_mean_gap"]
            == carried["visible_mean_change"] - ordinary["visible_mean_change"]
        )
        assert (
            comparison["source_mean_gap"]
            == carried["mean_area"] - ordinary["mean_area"]
        )
        assert (
            comparison["executor_mean_gap"]
            == carried["executor_mean_defect"] - ordinary["executor_mean_defect"]
        )
        assert (
            comparison["visible_mean_gap"]
            == comparison["source_mean_gap"] + comparison["executor_mean_gap"]
        )
        assert comparison["identity_residual"] == 0
        assert (
            carried["executor_mean_defect"]
            == -sum(carried["final_state"]["remainder"]) / 6
        )
        assert abs(carried["executor_mean_defect"]) <= F(1, 2**53)


def test_pressure_readout_shift_cancels_in_the_regular_common_capacity_mean(report):
    for case in report["cases"]:
        for flow in case["carried"]["flows"]:
            diagnostic = flow["pressure_readout"]
            assert diagnostic["pressure_identity_residual"] == (F(0),) * 6
            assert diagnostic["mean_nodal_readout_shift"] == 0
            assert diagnostic["degree_weighted_pressure_readout_shift"] == 0
            assert diagnostic["reversible_mean_nodal_readout_shift"] == 0


def test_nonzero_carry_survives_actual_intervening_events(report):
    witnessed = False
    for case in report["cases"]:
        for event in case["carried"]["events"]:
            assert event["before_binding"] == event["after_binding"]
            assert event["exact_epi_jump"] == (F(0),) * 6
            witnessed |= any(event["before_binding"]["remainder"])
    assert witnessed


def test_source_feedback_changes_are_retained_instead_of_forced_equal(report):
    assert any(
        case["comparison"]["any_pressure_difference"] for case in report["cases"]
    )
    for case in report["cases"]:
        rows = case["comparison"]["pressure_differences"]
        assert case["comparison"]["any_pressure_difference"] == any(
            any(row) for row in rows
        )


def test_report_keeps_finite_runtime_scope_and_serializes_exact_budgets(report):
    assert report["production_default_integrator_modified"] is False
    assert report["future_stability_certified"] is False
    assert report["empirical_correspondence_tested"] is False
    decoded = json.loads(json.dumps(_payload(report), allow_nan=False))
    for original, encoded in zip(report["cases"], decoded["cases"], strict=True):
        assert (
            F(encoded["carried"]["executor_mean_defect"])
            == original["carried"]["executor_mean_defect"]
        )


@pytest.mark.parametrize(
    "mode,epsilon", (("k2", 2.0**-12), ("null", 0), ("k1", -(2.0**-12)))
)
def test_additional_preparations_cannot_expand_the_campaign(mode, epsilon):
    with pytest.raises(ValueError):
        owner.run_c6_remainder_runtime_case(mode, epsilon)


def test_cli_refuses_source_changes_before_writing(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    provenance = iter((("head", True, "old"), ("head", True, "new")))
    monkeypatch.setattr(
        owner, "current_git_source_provenance", lambda *args: next(provenance)
    )
    monkeypatch.setattr(owner, "run_c6_remainder_runtime_comparison", lambda: {})
    monkeypatch.setattr(owner.sys, "argv", ["benchmark", "--output", str(output)])
    with pytest.raises(RuntimeError, match="source changed"):
        owner.main()
    assert not output.exists()
