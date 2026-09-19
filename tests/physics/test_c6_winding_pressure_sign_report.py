"""A generated sign hit, preserved carry, resource censoring and exact budgets."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_sign as campaign

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/research"
PARENT = ARTIFACTS / "c6_winding_carry_itinerary.json"
SOURCE = ARTIFACTS / "c6_winding_pressure_lattice.json"


@pytest.fixture(scope="module")
def parents():
    if not PARENT.is_file() or not SOURCE.is_file():
        pytest.skip("retained B27/B26 source reports are unavailable")
    return json.loads(PARENT.read_bytes()), json.loads(SOURCE.read_bytes())


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_pressure_sign(*parents)


def _offsets(state):
    return tuple((F(value) - F(1, 2)) * 2**54 for value in state["epi"])


def _integer_gradient(state):
    x = tuple(map(F, state["epi"]))
    return tuple((x[i - 1] + x[(i + 1) % 6] - 2 * x[i]) * 2**54 for i in range(6))


def test_first_hit_is_after_three_complete_boundaries_and_nineteen_steps(
    report, parents
):
    assert report["source"]["retained_B27_report_replayed"] is True
    assert report["source"]["retained_B26_selected_prefix_replayed"] is True
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
    assert report["stop_reason"] == "first_positive_node1_pressure"
    assert report["censor"] is None
    continuation = report["continuation"]
    assert continuation["area_reference"] == "B27.continuation.endpoint"
    assert continuation["step_count"] == 19 and continuation["boundary_count"] == 3
    assert tuple(row["step_count"] for row in report["boundaries"]) == (12, 3, 4)
    assert tuple(row["cumulative_steps"] for row in report["boundaries"]) == (
        12,
        15,
        19,
    )
    assert _offsets(report["source"]["inherited_state"]) == (-1, 0, 0, -1, 2, -2)
    assert tuple(_offsets(row["endpoint"]) for row in report["boundaries"]) == (
        (-1, 0, 0, -1, 4, -2),
        (-1, 0, 0, -2, 4, -2),
        (-1, 0, 2, -2, 4, -3),
    )
    assert tuple(
        row["horizon"]["first_exit_leaves_cell"] for row in report["boundaries"]
    ) == (
        (False, False, False, False, True, False),
        (False, False, False, True, False, False),
        (False, False, True, False, False, True),
    )


def test_sign_flip_comes_from_neighbor_two_while_selected_visible_node_stays_fixed(
    report,
):
    rows = report["boundaries"]
    assert tuple(row["source_pressure"]["gradient_indices"][1] for row in rows) == (
        -1,
        -1,
        -1,
    )
    assert tuple(row["refreshed_pressure"]["gradient_indices"][1] for row in rows) == (
        -1,
        -1,
        1,
    )
    expected_negative = -F(128295757220873, 2**106)
    expected_positive = F(2786216251278205, 2**108)
    assert all(
        F(row["source_pressure"]["pressure"][1]) == expected_negative for row in rows
    )
    assert F(rows[-1]["refreshed_pressure"]["pressure"][1]) == expected_positive
    assert report["first_positive_pressure_observed"] is True
    assert report["positive_node1_pressure_steps_integrated"] == 0
    initial = report["source"]["inherited_state"]
    assert all(
        step["after"]["epi"][1] == initial["epi"][1]
        for step in report["continuation"]["steps"]
    )
    before, after = rows[-1]["initial"], rows[-1]["endpoint"]
    changes = tuple(
        (F(b) - F(a)) * 2**54 for a, b in zip(before["epi"], after["epi"], strict=True)
    )
    assert changes == (0, 0, 2, 0, 0, -1)
    assert (
        _integer_gradient(after)[1] - _integer_gradient(before)[1]
        == changes[0] + changes[2] - 2 * changes[1]
        == 2
    )


def test_all_steps_preserve_the_inherited_carry_and_actual_held_nodal_product(report):
    initial = report["source"]["inherited_state"]
    assert any(initial["remainder"])
    previous, area = initial, [F(0)] * 6
    steps = report["continuation"]["steps"]
    for step in steps:
        assert step["before"] == previous
        assert step["timestep"] == 1 / 16 and step["capacity"] == (1.0,) * 6
        expected = tuple(F(p) / 16 for p in step["pressure"])
        assert (
            step["exact_increment"] == expected
            and step["nodal_balance_residual"] == (0,) * 6
        )
        area = [a + b for a, b in zip(area, expected, strict=True)]
        after = step["after"]
        for i in range(6):
            assert (
                F(after["epi"][i]) + after["remainder"][i]
                == F(initial["epi"][i]) + initial["remainder"][i] + area[i]
            )
            assert F(3, 8) <= F(after["epi"][i]) + after["remainder"][i] <= F(5, 8)
        previous = after
    assert previous == report["continuation"]["endpoint"]
    assert tuple(area) == report["continuation"]["total_nodal_area"]
    cursor = 0
    for row in report["boundaries"]:
        local = steps[cursor : cursor + row["step_count"]]
        assert all(step["before"]["epi"] == row["initial"]["epi"] for step in local)
        assert all(step["after"]["epi"] == row["initial"]["epi"] for step in local[:-1])
        assert all(
            step["pressure"] == row["source_pressure"]["pressure"] for step in local
        )
        assert local[-1]["after"] == row["endpoint"]
        cursor += len(local)


def test_exact_gradient_change_is_split_into_nodal_and_carry_terms_at_every_prefix(
    report,
):
    initial = report["source"]["inherited_state"]
    old = _integer_gradient(initial)
    rows = report["continuation"]["gradient_prefixes"]
    assert len(rows) == 19
    for step, row in zip(report["continuation"]["steps"], rows, strict=True):
        new = _integer_gradient(step["after"])
        assert (
            row["gradient_indices_before"] == old
            and row["gradient_indices_after"] == new
        )
        area, dr = row["nodal_area"], row["remainder_change"]
        independent_area = tuple(
            (area[i - 1] + area[(i + 1) % 6] - 2 * area[i]) * 2**54 for i in range(6)
        )
        independent_carry = tuple(
            (2 * dr[i] - dr[i - 1] - dr[(i + 1) % 6]) * 2**54 for i in range(6)
        )
        assert row["nodal_gradient_term"] == independent_area
        assert row["remainder_gradient_term"] == independent_carry
        assert row["gradient_index_change"] == tuple(
            b - a for a, b in zip(old, new, strict=True)
        )
        assert row["gradient_index_change"] == tuple(
            a + b for a, b in zip(independent_area, independent_carry, strict=True)
        )
        assert row["identity_residual"] == (0,) * 6
        assert (
            row["node1_nodal_area"]
            == F(row["ordinal"], 16) * F(row["node1_source_pressure"])
            < 0
        )
    final = report["continuation"]["gradient_balance"]
    assert final["gradient_index_change"] == (-1, 2, -5, 6, -6, 4)
    assert final["nodal_gradient_term"][1] == F(86621202941648393, 2**58)
    assert final["remainder_gradient_term"][1] == F(489839549361775095, 2**58)
    assert final["nodal_gradient_term"][1] + final["remainder_gradient_term"][1] == 2


def test_pressure_sign_mean_change_and_vector_compensation_remain_distinct(report):
    rows = report["boundaries"]
    assert tuple(row["prefix_balances"][-1]["mean_nodal_area"] for row in rows) == (
        -F(1, 2**111),
        F(3, 2**113),
        -F(1, 3 * 2**110),
    )
    assert tuple(
        sum(map(F, row["refreshed_pressure"]["pressure"])) for row in rows
    ) == (
        F(3, 2**108),
        -F(1, 2**107),
        -F(5, 2**108),
    )
    continuation = report["continuation"]
    assert (
        continuation["mean_nodal_area"]
        == continuation["mean_reconstructed_change"]
        == -F(11, 3 * 2**113)
    )
    assert continuation["total_nodal_area"][1] == -F(2437619387196587, 2**110)
    for prefix in continuation["prefix_balances"]:
        assert (
            prefix["mean_identity_residual"] == 0
            and prefix["nodal_identity_residual"] == (0,) * 6
        )
        assert prefix["mean_nodal_area"] == prefix["mean_reconstructed_change"]
    for name in (
        "node1_nodal_area_compensated",
        "mean_nodal_area_compensated",
        "total_nodal_area_compensated",
    ):
        assert report[name] is False


def test_uniform_sector_bound_is_conditional_and_uses_the_actual_carry(report):
    certificate = report["initial_negative_sector"]
    sector = certificate["sector"]
    assert (sector["node"], sector["sign"], sector["cut_index"]) == (1, -1, -1)
    assert sector["signed_pressure_margin"] == F(128295757220873, 2**106)
    initial = report["source"]["inherited_state"]
    assert certificate["outward_distance"] == F(initial["epi"][1]) + initial[
        "remainder"
    ][1] - F(3, 8)
    assert certificate["per_step_margin"] == sector["signed_pressure_margin"] / 16
    assert (
        certificate["max_sector_steps"]
        == certificate["outward_distance"] // certificate["per_step_margin"]
        == 1264728314825478005
    )
    assert certificate["first_exit_bound"] == 1264728314825478006
    assert certificate["initial_in_sector"] is True
    assert certificate["opposite_sign_hit_certified"] is False
    assert certificate["positive_band_exit_certified"] is False


def test_inverse_itinerary_binds_the_actual_carry_and_rejects_a_zero_reset(report):
    continuation = report["continuation"]
    itinerary = continuation["itinerary"]
    assert itinerary["feasible"] is True
    assert itinerary["zero_initial_carry_feasible"] is False
    assert (
        itinerary["visible_closed"] is False
        and itinerary["conditional_carried_cycle"] is False
    )
    assert continuation["supplied_initial_carry_coordinate_membership"] == (True,) * 6
    assert continuation["supplied_initial_carry_feasible"] is True
    assert itinerary["total_nodal_area"] == continuation["total_nodal_area"]
    initial = report["source"]["inherited_state"]
    for x, r, cell in zip(
        initial["epi"], initial["remainder"], itinerary["coordinates"], strict=True
    ):
        exact = F(x) + r
        assert exact > cell["lower"] or exact == cell["lower"] and cell["lower_closed"]
        assert exact < cell["upper"] or exact == cell["upper"] and cell["upper_closed"]
    assert itinerary["runtime_provenance_certified"] is False
    assert itinerary["pressure_provenance_certified"] is False


@pytest.mark.parametrize(
    "boundaries,steps,expected_calls,expected_steps,reason",
    (
        (8, 1, 0, 0, "next_boundary_exceeds_step_budget"),
        (8, 11, 0, 0, "next_boundary_exceeds_step_budget"),
        (1, 256, 1, 12, "boundary_budget_exhausted"),
        (8, 12, 1, 12, "step_budget_exhausted"),
        (8, 14, 1, 12, "next_boundary_exceeds_step_budget"),
        (8, 18, 2, 15, "next_boundary_exceeds_step_budget"),
    ),
)
def test_budget_censoring_precedes_any_oversized_shared_replay(
    parents, monkeypatch, boundaries, steps, expected_calls, expected_steps, reason
):
    original, calls = campaign.observe_nodal_remainder_cell_exit, []

    def checked(**kwargs):
        calls.append(kwargs["step_budget"])
        return original(**kwargs)

    monkeypatch.setattr(campaign, "observe_nodal_remainder_cell_exit", checked)
    result = campaign.analyze_c6_winding_pressure_sign(
        *parents, boundary_budget=boundaries, step_budget=steps
    )
    assert result["stop_reason"] == reason
    assert len(calls) == expected_calls
    assert result["continuation"]["step_count"] == expected_steps
    assert result["first_positive_pressure_observed"] is False
    if not expected_steps:
        assert result["continuation"]["itinerary"] is None
        assert result["continuation"]["supplied_initial_carry_feasible"] is None
        assert result["continuation"]["endpoint"] == result["source"]["inherited_state"]
        assert result["continuation"]["total_nodal_area"] == (0,) * 6
        assert result["node1_nodal_area_compensated"] is False
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
        {"step_budget": 2.0},
        {"boundary_budget": 0},
        {"boundary_budget": 9},
        {"boundary_budget": True},
    ),
)
def test_invalid_resource_budget_is_rejected_before_parent_replay(
    parents, monkeypatch, kwargs
):
    def forbidden(*args):
        raise AssertionError("invalid resource input must fail before analysis")

    monkeypatch.setattr(campaign, "_verified_source", forbidden)
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_sign(*parents, **kwargs)


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "flag",
        "bool_count",
        "carry",
        "pressure",
        "prefix",
        "cached_sector",
        "source_weight",
        "producer_manifest",
    ),
)
def test_corrupted_retained_source_is_replayed_and_rejected(parents, change):
    parent, source = deepcopy(parents)
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "flag":
        parent["original_tail_reachability_certified"] = True
    elif change == "bool_count":
        parent["new_graph_trajectories"] = False
    elif change == "carry":
        parent["continuation"]["endpoint"]["remainder"] = ["0"] * 6
    elif change == "pressure":
        parent["boundaries"][-1]["refreshed_pressure"]["pressure"][1] = 0.0
    elif change == "prefix":
        parent["continuation"]["prefix_balances"][0]["mean_nodal_area"] = "1"
    elif change == "cached_sector":
        parent["four_state_drift"]["observation"]["escape_step_bound"] = 1
    elif change == "source_weight":
        source["lattice_reference"]["source"]["epi_weight"] = "1/2"
    else:
        parent["input_evidence"]["producer_manifest"]["claim_id"] = "unrelated"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_sign(parent, source)


def test_no_graph_word_is_run_and_inputs_remain_unchanged(parents, monkeypatch):
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("the sign audit must not run a live operator word")

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    before = deepcopy(parents)
    result = campaign.analyze_c6_winding_pressure_sign(*parents, boundary_budget=1)
    assert parents == before
    assert result["new_graph_trajectories"] == 0


def test_detached_scope_and_finite_json_are_explicit(report):
    for flag in (
        "runtime_executed",
        "live_provenance_certified",
        "original_tail_reachability_certified",
        "future_compensation_cycle_certified",
        "positive_band_exit_certified",
    ):
        assert report[flag] is False
    assert report["new_graph_trajectories"] == 0
    assert "witness_sequence" not in report["continuation"]["itinerary"]
    json.dumps(campaign._payload(report), allow_nan=False)


def test_cli_binds_both_distinct_input_hashes_and_retains_historical_sources(
    tmp_path, monkeypatch, report
):
    parent_path, source_path, output = (
        tmp_path / name for name in ("parent.json", "source.json", "output.json")
    )
    parent_path.write_bytes(PARENT.read_bytes())
    source_path.write_bytes(SOURCE.read_bytes())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sign-audit",
            "--input",
            str(parent_path),
            "--source-input",
            str(source_path),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_pressure_sign", lambda *args: deepcopy(report)
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *args: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-pressure-sign-hit"
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64
    assert (
        result["input_evidence"]["sha256"]
        == hashlib.sha256(PARENT.read_bytes()).hexdigest()
    )
    assert (
        result["source_input_evidence"]["sha256"]
        == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    )
    assert (
        result["source_input_evidence"]["sha256"]
        == result["input_evidence"]["producer_input_evidence"]["sha256"]
    )
    assert (
        parent_path.read_bytes() == PARENT.read_bytes()
        and source_path.read_bytes() == SOURCE.read_bytes()
    )
    assert result["input_evidence"]["producer_manifest"] != result["manifest"]


def test_cli_rejects_wrong_source_bytes_before_replay(tmp_path, monkeypatch):
    parent_path, source_path, output = (
        tmp_path / name for name in ("parent.json", "source.json", "output.json")
    )
    parent_path.write_bytes(PARENT.read_bytes())
    source_path.write_bytes(SOURCE.read_bytes() + b"\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sign-audit",
            "--input",
            str(parent_path),
            "--source-input",
            str(source_path),
            "--output",
            str(output),
        ],
    )

    def forbidden(*args):
        raise AssertionError("the wrong source bytes must fail before replay")

    monkeypatch.setattr(campaign, "analyze_c6_winding_pressure_sign", forbidden)
    with pytest.raises(ValueError, match="lineage hash"):
        campaign.main()
    assert not output.exists()
