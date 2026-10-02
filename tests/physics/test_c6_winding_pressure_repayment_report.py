"""Actual positive-source duration and exact, baseline-preserving repayment."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from math import gcd
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_repayment as campaign
from tests.c6_lineage_fixtures import write_synthetic_c6_lineage

DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts/research"
INPUTS = tuple(
    DIRECTORY / name
    for name in (
        "c6_winding_pressure_sign.json",
        "c6_winding_carry_itinerary.json",
        "c6_winding_pressure_lattice.json",
    )
)


@pytest.fixture(scope="module")
def parents():
    if any(not path.is_file() for path in INPUTS):
        pytest.skip("the retained B28/B27/B26 evidence chain is unavailable")
    return tuple(json.loads(path.read_bytes()) for path in INPUTS)


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_pressure_repayment(*parents)


def _offsets(state):
    return tuple((F(value) - F(1, 2)) * 2**54 for value in state["epi"])


def test_one_analytic_boundary_ends_the_positive_sector_after_two_steps(
    report, parents
):
    assert report["source"]["retained_B28_report_replayed"] is True
    assert (
        campaign._payload(report["source"]["inherited_state"])
        == parents[0]["continuation"]["endpoint"]
    )
    assert report["analysis_limits"] == {
        "boundary_budget": 1,
        "step_budget": 256,
        "maximum_steps": 256,
    }
    assert (
        report["horizon"]["max_unchanged_steps"],
        report["horizon"]["first_exit_step"],
    ) == (1, 2)
    assert report["horizon"]["first_exit_leaves_cell"] == (
        False,
        False,
        True,
        False,
        False,
        False,
    )
    assert not any(report["horizon"]["first_exit_leaves_band"])
    assert report["stop_reason"] == "first_nonpositive_node1_pressure"
    assert report["positive_sector"] == {
        "completed_steps": 2,
        "duration_exact_steps": 2,
        "duration_lower_bound_steps": 2,
        "termination_observed": True,
        "future_duration_open": False,
    }
    assert _offsets(report["source"]["inherited_state"]) == (-1, 0, 2, -2, 4, -3)
    assert _offsets(report["continuation"]["endpoint"]) == (-1, 0, 0, -2, 4, -3)
    assert report["source_pressure"]["gradient_indices"][1] == 1
    assert report["continuation"]["refreshed_pressure"]["gradient_indices"][1] == -1
    assert F(report["source_pressure"]["pressure"][1]) == F(2786216251278205, 2**108)
    assert F(report["continuation"]["refreshed_pressure"]["pressure"][1]) == -F(
        128295757220873, 2**106
    )
    assert (
        report["post_exit_steps_integrated"]
        == report["hypothetical_crossing_steps_integrated"]
        == 0
    )


def test_complete_incoming_vector_matches_the_reconstructed_B27_to_B28_change(
    report, parents
):
    source = report["source"]
    baseline, initial = source["B27_area_reference_state"], source["inherited_state"]
    assert campaign._payload(baseline) == parents[1]["continuation"]["endpoint"]
    assert any(initial["remainder"]) and initial["remainder"] != baseline["remainder"]
    incoming = tuple(
        F(x) + r - F(y) - s
        for x, r, y, s in zip(
            initial["epi"],
            initial["remainder"],
            baseline["epi"],
            baseline["remainder"],
            strict=True,
        )
    )
    assert incoming == source["incoming_B27_nodal_area"]
    assert (
        source["incoming_B27_mean_nodal_area"]
        == sum(incoming) / 6
        == -F(11, 3 * 2**113)
    )
    assert incoming[1] == -F(2437619387196587, 2**110)
    assert report["continuation"]["local_area_reference"] == "B28.continuation.endpoint"
    assert report["continuation"]["net_area_reference"] == "B27.continuation.endpoint"


def test_two_actual_steps_are_canonical_and_keep_all_six_area_identities(report):
    continuation = report["continuation"]
    assert continuation["step_count"] == len(continuation["steps"]) == 2
    initial = previous = report["source"]["inherited_state"]
    baseline = report["source"]["B27_area_reference_state"]
    incoming = report["source"]["incoming_B27_nodal_area"]
    local = [F(0)] * 6
    for step, row in zip(
        continuation["steps"], continuation["net_prefixes"], strict=True
    ):
        assert step["before"] == previous and step["before"]["epi"] == initial["epi"]
        assert step["timestep"] == 1 / 16 and step["capacity"] == (1.0,) * 6
        assert step["pressure"] == report["source_pressure"]["pressure"]
        assert step["exact_increment"] == tuple(
            F(value) / 16 for value in step["pressure"]
        )
        assert step["nodal_balance_residual"] == (0,) * 6
        local = [a + b for a, b in zip(local, step["exact_increment"], strict=True)]
        net = tuple(a + b for a, b in zip(incoming, local, strict=True))
        assert row["local_nodal_area"] == tuple(local)
        assert row["B27_nodal_area"] == row["B27_reconstructed_change"] == net
        assert (
            row["identity_residual"]
            == row["display_carry_identity_residual"]
            == (0,) * 6
        )
        for i in range(6):
            exact = F(step["after"]["epi"][i]) + step["after"]["remainder"][i]
            assert exact - F(baseline["epi"][i]) - baseline["remainder"][i] == net[i]
            assert (
                row["B27_visible_change"][i] + row["B27_remainder_change"][i] == net[i]
            )
            assert F(3, 8) <= exact <= F(5, 8)
        assert row["exact_zero_nodes"] == ()
        assert (
            row["node1_first_side_restored"]
            is row["joint_zero"]
            is row["mean_zero"]
            is False
        )
        previous = step["after"]
    assert tuple(local) == continuation["local_nodal_area"]
    assert previous == continuation["endpoint"]
    assert continuation["local_nodal_area"][1] == F(2786216251278205, 2**111)
    assert continuation["B27_nodal_area"][1] == -F(2089022523114969, 2**111)
    assert continuation["local_mean_nodal_area"] == -F(5, 3 * 2**112)
    assert continuation["B27_mean_nodal_area"] == -F(7, 2**113)


def test_held_pressure_needs_four_steps_but_only_two_are_valid_and_executed(report):
    crossing = report["held_source_crossings"]
    node = crossing["coordinates"][1]
    assert crossing["max_steps"] == report["horizon"]["first_exit_step"] == 2
    assert node["continuous_zero_step"] == F(9750477548786348, 2786216251278205)
    assert node["first_zero_or_opposite_step"] == 4
    assert node["integer_zero_step"] is None
    assert (
        node["crossing_in_prefix"]
        is node["exact_zero_in_prefix"]
        is node["crossing_is_exact_zero"]
        is False
    )
    d, a = node["initial_area"], node["increment"]
    assert node["area_before_crossing"] == d + 3 * a == -F(1391828794951733, 2**112)
    assert node["area_at_crossing"] == d + 4 * a == F(1394387456326472, 2**112)
    assert crossing["endpoint_area"] == report["continuation"]["B27_nodal_area"]
    assert crossing["positive_joint_zero_step"] is None
    assert (
        crossing["initially_joint_zero"]
        is crossing["joint_zero_in_prefix"]
        is crossing["joint_zero_for_all_steps"]
        is False
    )
    assert (
        crossing["pressure_provenance_certified"]
        is crossing["band_provenance_certified"]
        is False
    )
    assert report["first_observed_node1_nonnegative_area_step"] is None
    assert report["first_observed_node1_exact_zero_step"] is None


def test_two_captured_pressure_levels_have_no_short_exact_scalar_return(report):
    result = report["two_level_return"]
    assert result["common_denominator"] == 2**108
    assert result["negative_integer"] == 513183028883492
    assert result["positive_integer"] == 2786216251278205
    assert (
        result["gcd"]
        == gcd(result["negative_integer"], result["positive_integer"])
        == 1
    )
    assert result["minimum_negative_steps"] == 2786216251278205
    assert result["minimum_positive_steps"] == 513183028883492
    assert result["minimum_total_steps"] == 3299399280161697
    assert (
        result["minimum_negative_steps"] * F(result["negative_pressure"])
        + result["minimum_positive_steps"] * F(result["positive_pressure"])
        == 0
    )
    assert (
        result["negative_pressure"]
        == report["continuation"]["refreshed_pressure"]["pressure"][1]
    )
    assert result["positive_pressure"] == report["source_pressure"]["pressure"][1]
    assert (
        result["pressure_provenance_certified"]
        is result["periodic_execution_certified"]
        is False
    )


def test_displayed_return_is_distinct_from_reconstructed_nodal_return(report):
    continuation = report["continuation"]
    assert continuation["displayed_matches_B27_reference"] == (
        True,
        True,
        True,
        False,
        False,
        False,
    )
    assert continuation["reconstructed_matches_B27_reference"] == (False,) * 6
    assert continuation["displayed_vector_returns_to_B27_reference"] is False
    assert continuation["reconstructed_vector_returns_to_B27_reference"] is False
    for prefix in continuation["net_prefixes"]:
        assert prefix["displayed_matches_B27_reference"][1] is True
        assert prefix["reconstructed_matches_B27_reference"][1] is False
        assert prefix["B27_visible_change"][1] == 0
        assert prefix["B27_remainder_change"][1] == prefix["B27_nodal_area"][1] < 0
    for flag in (
        "node1_nodal_area_compensated",
        "mean_nodal_area_compensated",
        "total_nodal_area_compensated",
    ):
        assert report[flag] is False


def test_gradient_and_inverse_itinerary_preserve_actual_carry(report):
    continuation = report["continuation"]
    gradient = continuation["gradient_balance"]
    assert (
        gradient["gradient_indices_before"][1] == 1
        and gradient["gradient_indices_after"][1] == -1
    )
    assert gradient["gradient_index_change"] == (0, -2, 4, -2, 0, 0)
    assert gradient["identity_residual"] == (0,) * 6
    assert gradient["gradient_index_change"] == tuple(
        a + b
        for a, b in zip(
            gradient["nodal_gradient_term"],
            gradient["remainder_gradient_term"],
            strict=True,
        )
    )
    itinerary = continuation["itinerary"]
    assert (
        itinerary["feasible"] is True
        and itinerary["zero_initial_carry_feasible"] is False
    )
    assert (
        itinerary["visible_closed"] is itinerary["conditional_carried_cycle"] is False
    )
    assert itinerary["total_nodal_area"] == continuation["local_nodal_area"]
    assert continuation["supplied_initial_carry_coordinate_membership"] == (True,) * 6
    assert continuation["supplied_initial_carry_feasible"] is True
    assert (
        itinerary["pressure_provenance_certified"]
        is itinerary["runtime_provenance_certified"]
        is False
    )


def test_budget_one_censors_before_any_new_cell_replay(parents, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("the too-long new boundary must be censored before replay")

    monkeypatch.setattr(campaign, "observe_nodal_remainder_cell_exit", forbidden)
    result = campaign.analyze_c6_winding_pressure_repayment(*parents, step_budget=1)
    assert result["stop_reason"] == "next_boundary_exceeds_step_budget"
    assert result["continuation"]["step_count"] == 0
    assert result["continuation"]["itinerary"] is None
    assert result["continuation"]["endpoint"] == result["source"]["inherited_state"]
    assert result["continuation"]["local_nodal_area"] == (0,) * 6
    assert (
        result["continuation"]["B27_nodal_area"]
        == result["source"]["incoming_B27_nodal_area"]
    )
    assert result["positive_sector"]["duration_exact_steps"] is None
    assert result["positive_sector"]["termination_observed"] is False
    assert result["positive_sector"]["future_duration_open"] is True


@pytest.mark.parametrize("budget", (0, True, 257, 2.0))
def test_invalid_budget_fails_before_input_analysis(parents, monkeypatch, budget):
    def forbidden(*args):
        raise AssertionError("invalid resource budget must fail before replay")

    monkeypatch.setattr(campaign, "_verified_source", forbidden)
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_repayment(*parents, step_budget=budget)


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "baseline",
        "incoming_area",
        "carry",
        "pressure",
        "prefix",
        "cached_crossing",
        "false_provenance",
        "lineage",
        "source_weight",
    ),
)
def test_corrupted_parents_are_rejected_by_complete_source_replay(parents, change):
    parent, previous, source = deepcopy(parents)
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "baseline":
        parent["continuation"]["area_reference"] = "B26"
    elif change == "incoming_area":
        parent["continuation"]["total_nodal_area"][1] = "0"
    elif change == "carry":
        parent["continuation"]["endpoint"]["remainder"] = ["0"] * 6
    elif change == "pressure":
        parent["continuation"]["refreshed_pressure"]["pressure"][1] = 0.0
    elif change == "prefix":
        parent["continuation"]["prefix_balances"][0]["mean_nodal_area"] = "1"
    elif change == "cached_crossing":
        parent["first_positive_pressure_observed"] = False
    elif change == "false_provenance":
        parent["live_provenance_certified"] = True
    elif change == "lineage":
        parent["input_evidence"]["producer_manifest"]["claim_id"] = "unrelated"
    else:
        source["lattice_reference"]["source"]["epi_weight"] = "1/2"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_repayment(parent, previous, source)


def test_only_one_two_step_boundary_is_run_with_no_graph_word(parents, monkeypatch):
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("the repayment audit must not run graph events")

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    owner, calls = campaign.observe_nodal_remainder_cell_exit, []

    def tracked(**kwargs):
        result = owner(**kwargs)
        calls.append((kwargs, len(result.sequence.steps)))
        return result

    monkeypatch.setattr(campaign, "observe_nodal_remainder_cell_exit", tracked)
    before = deepcopy(parents)
    result = campaign.analyze_c6_winding_pressure_repayment(*parents)
    assert parents == before and len(calls) == 1 and calls[0][1] == 2
    assert calls[0][0]["timestep"] == 1 / 16 and calls[0][0]["capacity"] == (1.0,) * 6
    assert result["new_graph_trajectories"] == 0


def test_scope_and_serialization_do_not_promote_a_pressure_or_displayed_return(report):
    for flag in (
        "runtime_executed",
        "live_provenance_certified",
        "original_tail_reachability_certified",
        "future_compensation_cycle_certified",
        "positive_band_exit_certified",
    ):
        assert report[flag] is False
    assert "witness_sequence" not in report["continuation"]["itinerary"]
    json.dumps(campaign._payload(report), allow_nan=False)


def test_cli_retains_three_source_hashes_and_separate_producer_manifest(
    tmp_path, monkeypatch, report
):
    paths = tuple(tmp_path / f"input_{i}.json" for i in range(3))
    for path, source in zip(paths, INPUTS, strict=True):
        path.write_bytes(source.read_bytes())
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repayment",
            "--input",
            str(paths[0]),
            "--parent-input",
            str(paths[1]),
            "--source-input",
            str(paths[2]),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        campaign,
        "analyze_c6_winding_pressure_repayment",
        lambda *args: deepcopy(report),
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *args: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-pressure-repayment"
    for key, source in zip(
        ("input_evidence", "parent_input_evidence", "source_input_evidence"),
        INPUTS,
        strict=True,
    ):
        assert result[key]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert result[key]["producer_manifest"] != result["manifest"]
    assert tuple(path.read_bytes() for path in paths) == tuple(
        source.read_bytes() for source in INPUTS
    )


@pytest.mark.parametrize("wrong_index", (1, 2))
def test_cli_checks_parent_and_source_byte_lineage_before_analysis(
    tmp_path, monkeypatch, wrong_index
):
    paths = write_synthetic_c6_lineage(tmp_path, 3)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repayment",
            "--input",
            str(paths[0]),
            "--parent-input",
            str(paths[1]),
            "--source-input",
            str(paths[2]),
            "--output",
            str(output),
        ],
    )

    def forbidden(*args):
        raise AssertionError("lineage failure must precede analysis")

    monkeypatch.setattr(campaign, "analyze_c6_winding_pressure_repayment", forbidden)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *args: None)
    with pytest.raises(AssertionError, match="lineage failure must precede analysis"):
        campaign.main()
    paths[wrong_index].write_bytes(paths[wrong_index].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="lineage hashes"):
        campaign.main()
    assert not output.exists()
