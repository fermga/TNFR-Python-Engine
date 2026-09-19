"""Frozen-stencil horizons, an earlier neighbor change and honest sign censoring."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_stencil as campaign

DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts/research"
INPUTS = tuple(
    DIRECTORY / name
    for name in (
        "c6_winding_pressure_levels.json",
        "c6_winding_pressure_repayment.json",
        "c6_winding_pressure_sign.json",
        "c6_winding_carry_itinerary.json",
        "c6_winding_pressure_lattice.json",
    )
)


@pytest.fixture(scope="module")
def parents():
    if any(not path.is_file() for path in INPUTS):
        pytest.skip("the retained B30 through B26 source chain is unavailable")
    return tuple(json.loads(path.read_bytes()) for path in INPUTS)


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_pressure_stencil(*parents)


def _offsets(state):
    return tuple((F(x) - F(1, 2)) * 2**54 for x in state["epi"])


def test_eight_boundary_ceiling_censors_the_sign_question_after_eighteen_steps(
    report, parents
):
    assert report["source"]["retained_B30_report_replayed"] is True
    assert (
        campaign._payload(report["source"]["inherited_state"])
        == parents[0]["continuation"]["endpoint"]
    )
    assert report["analysis_limits"] == {
        "boundary_budget": 8,
        "step_budget": 256,
        "maximum_boundaries": 8,
        "maximum_steps": 256,
    }
    assert report["stop_reason"] == "boundary_budget_exhausted"
    assert report["censor"] is None
    assert report["node4_nonpositive_pressure_observed"] is False
    assert report["nonpositive_node4_pressure_steps_integrated"] == 0
    assert (
        report["continuation"]["step_count"] == 18
        and report["continuation"]["boundary_count"] == 8
    )
    rows = report["boundaries"]
    assert tuple(row["step_count"] for row in rows) == (1, 5, 1, 4, 1, 3, 2, 1)
    assert tuple(row["cumulative_steps"] for row in rows) == (
        1,
        6,
        7,
        11,
        12,
        15,
        17,
        18,
    )
    assert tuple(_offsets(row["endpoint"]) for row in rows) == (
        (-2, 0, 2, -2, 4, -3),
        (-2, 0, 0, -2, 4, -3),
        (-2, 0, 2, -2, 4, -3),
        (-2, 0, 0, -2, 4, -3),
        (-2, 0, 2, -2, 4, -3),
        (-2, 0, 2, -2, 4, -4),
        (-2, 0, 0, -2, 4, -4),
        (-2, 0, 2, -2, 4, -4),
    )


def test_neighbor_five_changes_before_the_selected_center_horizon_and_keeps_pressure_positive(
    report,
):
    changed = report["first_stencil_change"]
    assert report["target_node"] == 4 and report["stencil_nodes"] == (3, 4, 5)
    assert (changed["step"], changed["boundary"], changed["changed_nodes"]) == (
        15,
        6,
        (5,),
    )
    assert changed["before"]["epi"][4] == changed["after"]["epi"][4]
    assert _offsets(changed["before"])[5] == -3 and _offsets(changed["after"])[5] == -4
    before, after = F(1668868774373469, 2**105), F(1462656319363363, 2**105)
    assert (
        F(changed["pressure_before"])
        == before
        > after
        == F(changed["pressure_after"])
        > 0
    )
    assert tuple(row["changed_stencil_nodes"] for row in report["boundaries"]) == (
        (),
        (),
        (),
        (),
        (),
        (5,),
        (),
        (),
    )
    assert (
        tuple(row["node4_gradient_after"] for row in report["boundaries"])
        == (-13,) * 5 + (-14,) * 3
    )
    steps = report["continuation"]["steps"]
    assert (
        tuple(F(step["pressure"][4]) for step in steps) == (before,) * 15 + (after,) * 3
    )
    initial = report["source"]["inherited_state"]["epi"]
    assert all(step["after"]["epi"][3:] == initial[3:] for step in steps[:14])
    assert steps[14]["after"]["epi"][3:] != initial[3:]
    assert all(step["after"]["epi"][4] == initial[4] for step in steps)


def test_initial_and_endpoint_stencil_bounds_are_separately_conditioned(report):
    initial, endpoint = (
        report["initial_frozen_stencil"],
        report["endpoint_frozen_stencil"],
    )
    assert initial["stencil"] == endpoint["stencil"] == (3, 4, 5)
    assert (
        initial["initial_max_frozen_steps"],
        initial["initial_first_exit_bound"],
    ) == (20, 21)
    assert (
        initial["uniform_max_frozen_steps"],
        initial["uniform_first_exit_bound"],
    ) == (43, 44)
    assert (
        endpoint["initial_max_frozen_steps"],
        endpoint["initial_first_exit_bound"],
    ) == (3, 4)
    assert (
        endpoint["uniform_max_frozen_steps"],
        endpoint["uniform_first_exit_bound"],
    ) == (49, 50)
    assert report["first_stencil_change"]["step"] < initial["initial_first_exit_bound"]
    assert initial["state"] == report["source"]["inherited_state"]
    assert endpoint["state"] == report["continuation"]["endpoint"]
    assert initial["state"]["remainder"][4] == F(4148688737535375, 2**110)
    for owner in (initial, endpoint):
        assert owner["center_cell_width"] == F(1, 2**53)
        assert owner["exact_increment"] == F(owner["pressure"]) / 16
        exact = F(owner["state"]["epi"][4]) + owner["state"]["remainder"][4]
        distance = owner["center_cell"]["upper"] - exact
        assert owner["initial_max_frozen_steps"] == distance // owner["exact_increment"]
        assert (
            owner["uniform_max_frozen_steps"]
            == owner["center_cell_width"] // owner["exact_increment"]
        )
        assert (
            owner["graph_provenance_certified"]
            is owner["sign_hit_certified"]
            is owner["positive_band_exit_certified"]
            is False
        )
    assert report["continuation"]["step_count"] == 18


def test_integer_cut_distances_are_necessary_and_not_a_future_trajectory(report):
    initial, endpoint = (
        report["initial_nonpositive_cut"],
        report["endpoint_nonpositive_cut"],
    )
    assert initial["nonpositive_max_index"] == endpoint["nonpositive_max_index"] == -22
    assert initial["current_gradient_index"] == -13
    assert endpoint["current_gradient_index"] == -14
    assert initial["necessary_gradient_change_upper_bound"] == -9
    assert endpoint["necessary_gradient_change_upper_bound"] == -8
    assert (
        initial["future_feasible_transition_certified"]
        is endpoint["future_feasible_transition_certified"]
        is False
    )
    gradient = report["continuation"]["gradient_balance"]
    assert gradient["gradient_index_change"] == (-1, 2, -4, 2, -1, 2)
    assert gradient["identity_residual"] == (0,) * 6
    assert gradient["gradient_index_change"] == tuple(
        a + b
        for a, b in zip(
            gradient["nodal_gradient_term"],
            gradient["remainder_gradient_term"],
            strict=True,
        )
    )
    assert (
        gradient["gradient_index_change"][4]
        == -1
        > initial["necessary_gradient_change_upper_bound"]
    )


def test_all_six_area_and_carry_identities_keep_the_B27_reference(report, parents):
    source, continuation = report["source"], report["continuation"]
    initial, baseline = source["inherited_state"], source["B27_area_reference_state"]
    assert campaign._payload(baseline) == parents[3]["continuation"]["endpoint"]
    incoming = source["incoming_B27_nodal_area"]
    assert incoming == tuple(
        F(x) + r - F(y) - s
        for x, r, y, s in zip(
            initial["epi"],
            initial["remainder"],
            baseline["epi"],
            baseline["remainder"],
            strict=True,
        )
    )
    assert continuation["local_area_reference"] == "B30.continuation.endpoint"
    assert continuation["net_area_reference"] == "B27.continuation.endpoint"
    local, previous = [F(0)] * 6, initial
    for step, prefix in zip(
        continuation["steps"], continuation["net_prefixes"], strict=True
    ):
        assert step["before"] == previous
        assert step["timestep"] == 1 / 16 and step["capacity"] == (1.0,) * 6
        assert step["exact_increment"] == tuple(F(p) / 16 for p in step["pressure"])
        assert step["nodal_balance_residual"] == (0,) * 6
        local = [a + b for a, b in zip(local, step["exact_increment"], strict=True)]
        net = tuple(a + b for a, b in zip(incoming, local, strict=True))
        assert prefix["local_nodal_area"] == tuple(local)
        assert prefix["B27_nodal_area"] == prefix["B27_reconstructed_change"] == net
        assert (
            prefix["identity_residual"]
            == prefix["display_carry_identity_residual"]
            == (0,) * 6
        )
        for i in range(6):
            exact = F(step["after"]["epi"][i]) + step["after"]["remainder"][i]
            assert exact - F(baseline["epi"][i]) - baseline["remainder"][i] == net[i]
            assert F(3, 8) <= exact <= F(5, 8)
        previous = step["after"]
    assert (
        previous == continuation["endpoint"]
        and tuple(local) == continuation["local_nodal_area"]
    )
    assert continuation["local_mean_nodal_area"] == -F(131, 3 * 2**114)
    assert continuation["B27_mean_nodal_area"] == -F(275, 3 * 2**114)
    assert continuation["local_nodal_area"][4] == F(7355250143423031, 2**107)
    assert continuation["B27_nodal_area"][4] == F(199468445328943663, 2**110)
    assert all(
        prefix["local_nodal_area"][4] > 0 for prefix in continuation["net_prefixes"]
    )
    assert (
        report["mean_nodal_area_compensated"]
        is report["total_nodal_area_compensated"]
        is False
    )


def test_visible_stasis_at_node_four_does_not_reset_carry_or_close_the_itinerary(
    report,
):
    continuation = report["continuation"]
    initial, endpoint = report["source"]["inherited_state"], continuation["endpoint"]
    assert initial["epi"][4] == endpoint["epi"][4]
    assert (
        endpoint["remainder"][4] - initial["remainder"][4]
        == continuation["local_nodal_area"][4]
        > 0
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


@pytest.mark.parametrize(
    "boundary_budget,step_budget,count,calls,reason,changed",
    (
        (8, 1, 1, 1, "step_budget_exhausted", False),
        (5, 256, 12, 5, "boundary_budget_exhausted", False),
        (8, 14, 12, 5, "next_boundary_exceeds_step_budget", False),
        (8, 15, 15, 6, "step_budget_exhausted", True),
        (8, 17, 17, 7, "step_budget_exhausted", True),
    ),
)
def test_smaller_budgets_stop_before_an_oversized_boundary(
    parents, monkeypatch, boundary_budget, step_budget, count, calls, reason, changed
):
    owner, observed = campaign.observe_nodal_remainder_cell_exit, []

    def tracked(**kwargs):
        result = owner(**kwargs)
        observed.append(len(result.sequence.steps))
        return result

    monkeypatch.setattr(campaign, "observe_nodal_remainder_cell_exit", tracked)
    result = campaign.analyze_c6_winding_pressure_stencil(
        *parents, boundary_budget=boundary_budget, step_budget=step_budget
    )
    assert result["stop_reason"] == reason
    assert (
        result["continuation"]["step_count"] == sum(observed) == count
        and len(observed) == calls
    )
    assert (result["first_stencil_change"] is not None) is changed
    assert result["node4_nonpositive_pressure_observed"] is False
    if result["censor"] is not None:
        assert (
            result["censor"]["horizon"]["first_exit_step"]
            > result["censor"]["remaining_step_budget"]
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"step_budget": 0},
        {"step_budget": True},
        {"step_budget": 257},
        {"boundary_budget": 0},
        {"boundary_budget": 9},
    ),
)
def test_invalid_limits_fail_before_parent_replay(parents, monkeypatch, kwargs):
    def forbidden(*args):
        raise AssertionError(
            "invalid resource limits must fail before historical replay"
        )

    monkeypatch.setattr(campaign, "_verified_source", forbidden)
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_stencil(*parents, **kwargs)


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "carry",
        "area",
        "baseline",
        "pressure",
        "prefix",
        "cached_bound",
        "flag",
        "lineage",
        "source",
    ),
)
def test_source_tampering_fails_full_replay_validation(parents, change):
    parent, previous, ancestor, earlier, source = deepcopy(parents)
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "carry":
        parent["continuation"]["endpoint"]["remainder"] = ["0"] * 6
    elif change == "area":
        parent["continuation"]["B27_nodal_area"][4] = "0"
    elif change == "baseline":
        parent["continuation"]["net_area_reference"] = "B30"
    elif change == "pressure":
        parent["continuation"]["refreshed_pressure"]["pressure"][4] = 0.0
    elif change == "prefix":
        parent["continuation"]["prefix_balances"][0]["mean_nodal_area"] = "1"
    elif change == "cached_bound":
        parent["finite_class_drift"]["observation"]["max_confined_steps"] = 1
    elif change == "flag":
        parent["live_provenance_certified"] = True
    elif change == "lineage":
        parent["previous_input_evidence"]["producer_manifest"]["claim_id"] = "unrelated"
    else:
        source["lattice_reference"]["source"]["epi_weight"] = "1/2"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_stencil(
            parent, previous, ancestor, earlier, source
        )


def test_exactly_eight_new_boundaries_are_run_without_a_graph_word(
    parents, monkeypatch
):
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("the stencil audit must not execute graph words")

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    owner, calls = campaign.observe_nodal_remainder_cell_exit, []

    def tracked(**kwargs):
        result = owner(**kwargs)
        calls.append(len(result.sequence.steps))
        return result

    monkeypatch.setattr(campaign, "observe_nodal_remainder_cell_exit", tracked)
    before = deepcopy(parents)
    result = campaign.analyze_c6_winding_pressure_stencil(*parents)
    assert parents == before and calls == [1, 5, 1, 4, 1, 3, 2, 1]
    assert result["new_graph_trajectories"] == 0


def test_scope_and_finite_serialization_preserve_the_censored_outcome(report):
    for flag in (
        "runtime_executed",
        "live_provenance_certified",
        "original_tail_reachability_certified",
        "future_pressure_sign_exit_certified",
        "future_compensation_cycle_certified",
        "positive_band_exit_certified",
    ):
        assert report[flag] is False
    assert "witness_sequence" not in report["continuation"]["itinerary"]
    json.dumps(campaign._payload(report), allow_nan=False)


def _cli(monkeypatch, paths, output):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stencil",
            "--input",
            str(paths[0]),
            "--parent-input",
            str(paths[1]),
            "--previous-input",
            str(paths[2]),
            "--earlier-input",
            str(paths[3]),
            "--source-input",
            str(paths[4]),
            "--output",
            str(output),
        ],
    )


def test_cli_retains_all_five_input_hashes_and_distinct_output_source(
    tmp_path, monkeypatch, report
):
    paths = tuple(tmp_path / f"source_{i}.json" for i in range(5))
    for path, source in zip(paths, INPUTS, strict=True):
        path.write_bytes(source.read_bytes())
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_pressure_stencil", lambda *args: deepcopy(report)
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *args: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-pressure-stencil"
    for key, source in zip(
        (
            "input_evidence",
            "parent_input_evidence",
            "previous_input_evidence",
            "earlier_input_evidence",
            "source_input_evidence",
        ),
        INPUTS,
        strict=True,
    ):
        assert result[key]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert result[key]["producer_manifest"] != result["manifest"]
    assert tuple(path.read_bytes() for path in paths) == tuple(
        source.read_bytes() for source in INPUTS
    )


@pytest.mark.parametrize("changed_index", (1, 2, 3, 4))
def test_cli_rejects_wrong_historical_bytes_before_replay(
    tmp_path, monkeypatch, changed_index
):
    paths = tuple(tmp_path / f"source_{i}.json" for i in range(5))
    for i, (path, source) in enumerate(zip(paths, INPUTS, strict=True)):
        path.write_bytes(source.read_bytes() + (b"\n" if i == changed_index else b""))
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)

    def forbidden(*args):
        raise AssertionError("wrong historical bytes must fail before analysis")

    monkeypatch.setattr(campaign, "analyze_c6_winding_pressure_stencil", forbidden)
    with pytest.raises(ValueError, match="lineage hashes"):
        campaign.main()
    assert not output.exists()
