"""A first novel source, nonzero overshoot and a finite carried-class bound."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from functools import reduce
from math import gcd
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_levels as campaign
from tests.c6_lineage_fixtures import write_synthetic_c6_lineage

DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts/research"
INPUTS = tuple(
    DIRECTORY / name
    for name in (
        "c6_winding_pressure_repayment.json",
        "c6_winding_pressure_sign.json",
        "c6_winding_carry_itinerary.json",
        "c6_winding_pressure_lattice.json",
    )
)


@pytest.fixture(scope="module")
def parents():
    if any(not path.is_file() for path in INPUTS):
        pytest.skip("the retained B29/B28/B27/B26 source chain is unavailable")
    return tuple(json.loads(path.read_bytes()) for path in INPUTS)


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_pressure_levels(*parents)


def _offsets(state):
    return tuple((F(x) - F(1, 2)) * 2**54 for x in state["epi"])


def test_first_new_level_stops_four_boundaries_and_eleven_steps(report, parents):
    assert (
        campaign._payload(report["source"]["inherited_state"])
        == parents[0]["continuation"]["endpoint"]
    )
    assert report["source"]["retained_B29_report_replayed"] is True
    assert report["stop_reason"] == "first_node1_pressure_outside_two_levels"
    assert report["censor"] is None
    assert report["analysis_limits"] == {
        "boundary_budget": 8,
        "step_budget": 256,
        "maximum_boundaries": 8,
        "maximum_steps": 256,
    }
    continuation = report["continuation"]
    assert (continuation["boundary_count"], continuation["step_count"]) == (4, 11)
    rows = report["boundaries"]
    assert tuple(row["step_count"] for row in rows) == (1, 5, 1, 4)
    assert tuple(row["cumulative_steps"] for row in rows) == (1, 6, 7, 11)
    assert tuple(row["source_pressure"]["gradient_indices"][1] for row in rows) == (
        -1,
        1,
        -1,
        1,
    )
    assert tuple(row["refreshed_pressure"]["gradient_indices"][1] for row in rows) == (
        1,
        -1,
        1,
        -2,
    )
    assert tuple(_offsets(row["endpoint"]) for row in rows) == (
        (-1, 0, 2, -2, 4, -3),
        (-1, 0, 0, -2, 4, -3),
        (-1, 0, 2, -2, 4, -3),
        (-2, 0, 0, -2, 4, -3),
    )
    assert tuple(row["horizon"]["first_exit_leaves_cell"] for row in rows) == (
        (False, False, True, False, False, False),
        (False, False, True, False, False, False),
        (False, False, True, False, False, False),
        (True, False, True, False, False, False),
    )


def test_only_the_two_old_levels_act_and_the_third_is_captured_without_evolution(
    report,
):
    levels = report["pressure_levels"]
    negative, positive = levels["inherited_levels"]
    assert F(negative) == -F(128295757220873, 2**106)
    assert F(positive) == F(2786216251278205, 2**108)
    assert F(levels["captured_novel_level"]) == -F(4325765337928681, 2**109)
    assert levels["executed_step_counts"] == (2, 9)
    assert levels["novel_level_steps_integrated"] == 0
    values = tuple(step["pressure"][1] for step in report["continuation"]["steps"])
    assert values == (negative,) + (positive,) * 5 + (negative,) + (positive,) * 4
    assert levels["captured_novel_level"] not in values
    assert (
        report["continuation"]["refreshed_pressure"]["pressure"][1]
        == levels["captured_novel_level"]
    )


def test_visible_revisits_keep_different_carries_and_different_next_exit_times(report):
    first, second, third, fourth = report["boundaries"]
    assert first["initial"]["epi"] == second["endpoint"]["epi"]
    assert first["initial"]["remainder"] != second["endpoint"]["remainder"]
    assert second["initial"]["epi"] == fourth["initial"]["epi"]
    assert second["initial"]["remainder"] != fourth["initial"]["remainder"]
    assert second["source_pressure"] == fourth["source_pressure"]
    assert second["horizon"]["first_exit_step"] == 5
    assert fourth["horizon"]["first_exit_step"] == 4
    assert first["endpoint"] == second["initial"]
    assert (
        second["endpoint"] == third["initial"]
        and third["endpoint"] == fourth["initial"]
    )


def test_all_six_local_and_B27_area_budgets_match_actual_reconstructed_states(
    report, parents
):
    source, continuation = report["source"], report["continuation"]
    initial, baseline = source["inherited_state"], source["B27_area_reference_state"]
    assert campaign._payload(baseline) == parents[2]["continuation"]["endpoint"]
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
    assert continuation["local_area_reference"] == "B29.continuation.endpoint"
    assert continuation["net_area_reference"] == "B27.continuation.endpoint"
    local, previous = [F(0)] * 6, initial
    for step, row in zip(
        continuation["steps"], continuation["net_prefixes"], strict=True
    ):
        assert step["before"] == previous
        assert step["timestep"] == 1 / 16 and step["capacity"] == (1.0,) * 6
        assert step["exact_increment"] == tuple(F(p) / 16 for p in step["pressure"])
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
            assert F(3, 8) <= exact <= F(5, 8)
            assert exact - F(baseline["epi"][i]) - baseline["remainder"][i] == net[i]
            assert (
                row["B27_visible_change"][i] + row["B27_remainder_change"][i] == net[i]
            )
        previous = step["after"]
    assert (
        previous == continuation["endpoint"]
        and tuple(local) == continuation["local_nodal_area"]
    )
    assert tuple(value * 2**112 for value in continuation["local_nodal_area"]) == (
        -53012427250770350,
        24049580203736861,
        9296044417434,
        -16515588489643380,
        146860452144865272,
        -101391312652605888,
    )
    assert tuple(value * 2**112 for value in continuation["B27_nodal_area"]) == (
        -122873677022523069,
        19871535157506923,
        124954297600791358,
        -187220008878510488,
        562505776726237660,
        -397237923583502456,
    )


def test_actual_overshoot_is_not_an_exact_zero_or_mean_or_vector_return(report):
    continuation = report["continuation"]
    assert report["first_observed_node1_nonnegative_area_step"] == 3
    assert report["first_observed_node1_exact_zero_step"] is None
    prefixes = continuation["net_prefixes"]
    assert prefixes[1]["B27_nodal_area"][1] < 0
    assert prefixes[2]["B27_nodal_area"][1] == F(220301106860745, 2**110) > 0
    assert all(row["B27_nodal_area"][1] != 0 for row in prefixes)
    assert continuation["local_mean_nodal_area"] == -F(17, 2**113)
    assert continuation["B27_mean_nodal_area"] == -F(3, 2**110)
    assert all(row["B27_mean_nodal_area"] < 0 for row in prefixes)
    assert all(row["displayed_matches_B27_reference"][1] for row in prefixes)
    assert all(not row["reconstructed_matches_B27_reference"][1] for row in prefixes)
    for flag in (
        "node1_nodal_area_compensated",
        "mean_nodal_area_compensated",
        "total_nodal_area_compensated",
    ):
        assert report[flag] is False


def test_new_level_preserves_the_necessary_return_modulus_without_execution_claim(
    report,
):
    old = report["pressure_levels"]["inherited_return_arithmetic"]
    new = report["pressure_levels"]["captured_return_arithmetic"]
    assert old["common_denominator"] == 2**108 and new["common_denominator"] == 2**109
    assert old["integer_levels"] == (-513183028883492, 2786216251278205)
    assert new["integer_levels"] == (
        -1026366057766984,
        5572432502556410,
        -4325765337928681,
    )
    for result in (old, new):
        integers = result["integer_levels"]
        assert result["difference_gcd"] == reduce(
            gcd, (abs(z - integers[0]) for z in integers), 0
        )
        assert result["residue_gcd"] == gcd(result["difference_gcd"], integers[0]) == 1
        assert result["necessary_length_multiple"] == 3299399280161697
        assert all((z - integers[0]) % result["difference_gcd"] == 0 for z in integers)
        assert result["arithmetic_zero_area_possible"] is True
        assert (
            result["pressure_provenance_certified"]
            is result["periodic_execution_certified"]
            is False
        )


def test_three_deduplicated_states_have_a_conditional_44_step_escape_bound(report):
    certificate = report["finite_class_drift"]
    observation = certificate["observation"]
    assert certificate["unique_state_count"] == 3
    assert len(set(observation["epi_states"])) == len(observation["epi_states"]) == 3
    assert observation["functional"] == (0, 0, 0, 0, 1, 0)
    pressure = F(1668868774373469, 2**105)
    assert observation["functional_projections"] == (pressure,) * 3
    assert (
        tuple(F(row[4]) for row in observation["pressure_vectors"]) == (pressure,) * 3
    )
    assert observation["minimum_pressure_projection"] == pressure
    assert observation["width"] == F(1, 2**53)
    assert (
        observation["max_confined_steps"]
        == observation["width"] // (pressure / 16)
        == 43
    )
    assert observation["escape_step_bound"] == 44
    assert certificate["conditional_class_escape_certified"] is True
    assert (
        certificate["pressure_provenance_certified"]
        is certificate["positive_band_exit_certified"]
        is False
    )
    assert report["continuation"]["step_count"] == 11


@pytest.mark.parametrize(
    "remote", ((0.375, 0.375, 0.375), (0.625, 0.625, 0.625), (0.375, 0.5, 0.625))
)
def test_generated_node_four_pressure_depends_only_on_its_fixed_local_epi_stencil(
    parents, report, remote
):
    source = parents[3]["lattice_reference"]["source"]
    reference = campaign.derive_c6_pressure_lattice(
        phase=report["source"]["phase"],
        epi_weight=campaign._represented_scalar(source["epi_weight"]),
        phase_weight=campaign._represented_scalar(source["phase_weight"]),
        epi_lower=0.375,
        epi_upper=0.625,
    )
    epi = remote + report["continuation"]["endpoint"]["epi"][3:]
    observed = campaign.observe_c6_pressure_lattice(reference, epi=epi)
    assert observed.gradient_indices[4] == -13
    assert F(observed.pressure[4]) == F(1668868774373469, 2**105)
    assert observed.fused_pressure_agreement is True


def test_gradient_and_inverse_itinerary_distinguish_inherited_carry_from_zero_reset(
    report,
):
    continuation = report["continuation"]
    gradient = continuation["gradient_balance"]
    assert gradient["gradient_index_change"] == (2, -1, 0, 0, 0, -1)
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


@pytest.mark.parametrize(
    "boundaries,budget,completed,calls,stop",
    (
        (8, 1, 1, 1, "step_budget_exhausted"),
        (1, 256, 1, 1, "boundary_budget_exhausted"),
        (8, 5, 1, 1, "next_boundary_exceeds_step_budget"),
        (8, 6, 6, 2, "step_budget_exhausted"),
        (8, 10, 7, 3, "next_boundary_exceeds_step_budget"),
    ),
)
def test_resource_censoring_does_not_run_an_oversized_next_boundary(
    parents, monkeypatch, boundaries, budget, completed, calls, stop
):
    owner, observed = campaign.observe_nodal_remainder_cell_exit, []

    def checked(**kwargs):
        result = owner(**kwargs)
        observed.append(len(result.sequence.steps))
        return result

    monkeypatch.setattr(campaign, "observe_nodal_remainder_cell_exit", checked)
    result = campaign.analyze_c6_winding_pressure_levels(
        *parents, boundary_budget=boundaries, step_budget=budget
    )
    assert result["stop_reason"] == stop
    assert result["continuation"]["step_count"] == completed and len(observed) == calls
    assert sum(observed) == completed
    assert result["pressure_levels"]["captured_novel_level"] is None
    assert result["pressure_levels"]["captured_return_arithmetic"] is None
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
        {"boundary_budget": 9},
        {"boundary_budget": 0},
    ),
)
def test_invalid_budget_fails_before_replaying_sources(parents, monkeypatch, kwargs):
    def forbidden(*args):
        raise AssertionError(
            "invalid resource limits must fail before historical replay"
        )

    monkeypatch.setattr(campaign, "_verified_source", forbidden)
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_levels(*parents, **kwargs)


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "carry",
        "area",
        "baseline",
        "pressure",
        "cached_period",
        "flag",
        "lineage",
        "source",
    ),
)
def test_retained_claims_and_numerics_are_rederived_before_acceptance(parents, change):
    parent, previous, ancestor, source = deepcopy(parents)
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "carry":
        parent["continuation"]["endpoint"]["remainder"] = ["0"] * 6
    elif change == "area":
        parent["continuation"]["B27_nodal_area"][1] = "0"
    elif change == "baseline":
        parent["continuation"]["net_area_reference"] = "B28"
    elif change == "pressure":
        parent["continuation"]["refreshed_pressure"]["pressure"][1] = 0.0
    elif change == "cached_period":
        parent["two_level_return"]["minimum_total_steps"] = 1
    elif change == "flag":
        parent["live_provenance_certified"] = True
    elif change == "lineage":
        parent["parent_input_evidence"]["producer_manifest"]["claim_id"] = "unrelated"
    else:
        source["lattice_reference"]["source"]["epi_weight"] = "1/2"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_levels(parent, previous, ancestor, source)


def test_only_four_new_pressure_refreshes_and_no_graph_word(parents, monkeypatch):
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("the pressure-level report must not run graph words")

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    owner, observed = campaign.observe_c6_pressure_lattice, []

    def tracked(reference, *, epi):
        observed.append(epi)
        return owner(reference, epi=epi)

    monkeypatch.setattr(campaign, "observe_c6_pressure_lattice", tracked)
    before = deepcopy(parents)
    result = campaign.analyze_c6_winding_pressure_levels(*parents)
    assert parents == before and len(observed) == 4 and len(set(observed)) == 3
    assert result["finite_class_drift"]["unique_state_count"] == 3


def test_detached_scope_and_finite_json(report):
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


def _cli(monkeypatch, paths, output):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "levels",
            "--input",
            str(paths[0]),
            "--parent-input",
            str(paths[1]),
            "--previous-input",
            str(paths[2]),
            "--source-input",
            str(paths[3]),
            "--output",
            str(output),
        ],
    )


def test_cli_retains_four_input_hashes_and_distinct_output_source_provenance(
    tmp_path, monkeypatch, report
):
    paths = tuple(tmp_path / f"source_{i}.json" for i in range(4))
    for path, source in zip(paths, INPUTS, strict=True):
        path.write_bytes(source.read_bytes())
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_pressure_levels", lambda *args: deepcopy(report)
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *args: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-finite-pressure-levels"
    for key, source in zip(
        (
            "input_evidence",
            "parent_input_evidence",
            "previous_input_evidence",
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


@pytest.mark.parametrize("changed_index", (1, 2, 3))
def test_cli_rejects_wrong_historical_bytes_before_analysis(
    tmp_path, monkeypatch, changed_index
):
    paths = write_synthetic_c6_lineage(tmp_path, 4)
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)

    def forbidden(*args):
        raise AssertionError("wrong historical bytes must fail before replay")

    monkeypatch.setattr(campaign, "analyze_c6_winding_pressure_levels", forbidden)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *args: None)
    with pytest.raises(AssertionError, match="wrong historical bytes"):
        campaign.main()
    paths[changed_index].write_bytes(paths[changed_index].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="lineage hashes"):
        campaign.main()
    assert not output.exists()
