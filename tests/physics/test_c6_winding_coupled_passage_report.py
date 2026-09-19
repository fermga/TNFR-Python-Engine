"""A derived finite budget precedes the first-hit campaign and its signed balances."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_coupled_passage as campaign
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)

DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts/research"
INPUTS = tuple(DIRECTORY / name for name in campaign.INPUT_NAMES)


@pytest.fixture(scope="module")
def parents():
    if any(not path.is_file() for path in INPUTS):
        pytest.skip("the retained B36 through B26 evidence chain is unavailable")
    return tuple(json.loads(path.read_bytes()) for path in INPUTS)


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_coupled_passage(*parents)


def _exact(state):
    return tuple(
        F(x) + F(r) for x, r in zip(state["epi"], state["remainder"], strict=True)
    )


def _difference(after, before):
    return tuple(b - a for a, b in zip(before, after, strict=True))


def _mean(values):
    return sum(values, F(0)) / 6


def test_source_is_the_unchanged_carried_endpoint_and_not_a_balance_point(
    report, parents
):
    initial = report["source"]["initial_state"]
    assert report["source"]["source_report_replayed"] is True
    assert campaign._payload(initial) == parents[0]["source"]["endpoint_state"]
    assert campaign._payload(initial) == parents[1]["continuation"]["endpoint"]
    assert any(initial["remainder"])
    assert report["B40_continuation"]["initial"] == initial
    assert all(
        point["state"] != initial
        for point in report["B38_static_pressure_balance"]["points"]
    )
    assert report["new_conditional_numerical_steps"] == 118


def test_self_consistent_closure_is_stricter_and_keeps_the_cut_possible(
    report, parents
):
    closure = report["B37_self_consistent_closure"]
    old = F(parents[0]["B35_uniform_tube"]["reference"]["energy_bound"])
    assert 0 < closure["effective_norm_factor"] < 1
    assert 0 < closure["energy_bound"] < old
    assert closure["energy_bound"] == closure["energy_floor"]
    assert (
        closure["rounding_bound"]
        == closure["product_error_bound"] + closure["assembly_error_bound"]
    )
    assert closure["gradient_index_lower"] == (-26, -28, -33, -17, -49, -13)
    assert closure["gradient_index_upper"] == (29, 27, 22, 38, 6, 42)
    assert (
        closure["gradient_index_lower"][4] <= -22 <= closure["gradient_index_upper"][4]
    )
    assert closure["mean_increment_lower"] < 0 < closure["mean_increment_upper"]


def test_static_positive_convex_balance_has_the_actual_initial_mean_without_temporal_promotion(
    report,
):
    balance = report["B38_static_pressure_balance"]
    points, weights = balance["points"], balance["weights"]
    assert len(points) == len(weights) == 7
    assert all(type(weight) is F and weight > 0 for weight in weights)
    assert sum(weights) == balance["weight_sum"] == 1
    actual_mean = _mean(_exact(report["source"]["initial_state"]))
    assert balance["on_origin_mean_slice"] is True
    assert balance["common_reconstructed_mean"] == actual_mean
    assert balance["reconstructed_means"] == (actual_mean,) * 7
    for point in points:
        assert _mean(_exact(point["state"])) == actual_mean
        assert len(set(point["state"]["remainder"])) == 1
        assert point["energy"] == sum(value**2 for value in point["relative_error"])
        assert point["energy"] <= report["B37_self_consistent_closure"]["energy_bound"]
        assert point["reachable_from_initial_state_certified"] is False
    pressure_sum = tuple(
        sum(
            (
                weight * F(point["observation"]["pressure"][i])
                for point, weight in zip(points, weights, strict=True)
            ),
            F(0),
        )
        for i in range(6)
    )
    assert pressure_sum == balance["pressure_balance"] == (0,) * 6
    pressure_means = tuple(
        _mean(tuple(map(F, point["observation"]["pressure"]))) for point in points
    )
    assert min(pressure_means) < 0 < max(pressure_means)
    assert balance["class_wide_strict_linear_drift_excluded"] is True
    for key in (
        "temporal_compensation_certified",
        "bounded_trajectory_certified",
        "future_runtime_certified",
    ):
        assert balance[key] is False
    assert report["static_balance_is_temporal_itinerary"] is False


def test_finite_passage_bound_precedes_and_covers_the_observed_first_hit(report):
    passage, continuation = (
        report["B39_finite_pressure_passage"],
        report["B40_continuation"],
    )
    assert passage["sign_hit_certified"] is passage["band_covers_contradiction"] is True
    assert passage["minimum_positive_index"] == -21
    assert passage["minimum_positive_pressure"] == F(38338268585241, 2**106)
    assert passage["mean_pressure_upper"] < passage["minimum_positive_pressure"]
    assert (
        passage["centered_increment_lower"]
        == (passage["minimum_positive_pressure"] - passage["mean_pressure_upper"]) / 16
    )
    assert passage["contradiction_steps"] == 30255
    assert passage["latest_pressure_index"] == 30254
    assert (
        passage["band_horizon"]["maximum_steps"]
        == 196713720348826219
        > passage["contradiction_steps"]
    )
    assert continuation["step_count"] == 118 < passage["latest_pressure_index"]
    assert passage["abstention_reason"] is None
    for key in (
        "graph_provenance_certified",
        "future_runtime_certified",
        "infinite_trapping_certified",
    ):
        assert passage[key] is False


def test_every_transition_has_fresh_canonical_pressure_and_the_stop_is_the_first_cut(
    report, parents
):
    source = parents[-1]["lattice_reference"]["source"]
    lattice = derive_c6_pressure_lattice(
        phase=tuple(report["source"]["phase"]),
        epi_weight=float(F(source["epi_weight"])),
        phase_weight=float(F(source["phase_weight"])),
    )
    continuation = report["B40_continuation"]
    previous = report["source"]["initial_state"]
    cache = {}
    for step in continuation["steps"]:
        assert step["before"] == previous
        epi = tuple(step["before"]["epi"])
        if epi not in cache:
            cache[epi] = observe_c6_pressure_lattice(lattice, epi=epi)
        reading = cache[epi]
        assert step["pressure"] == reading.pressure and reading.pressure[4] > 0
        assert step["capacity"] == (1.0,) * 6 and step["timestep"] == 1 / 16
        change = _difference(_exact(step["after"]), _exact(step["before"]))
        assert change == tuple(F(value) / 16 for value in reading.pressure)
        assert (
            step["exact_increment"] == change
            and step["nodal_balance_residual"] == (0,) * 6
        )
        previous = step["after"]
    assert previous == continuation["endpoint"]
    final = observe_c6_pressure_lattice(lattice, epi=tuple(previous["epi"]))
    assert final.pressure == continuation["refreshed_pressure"]["pressure"]
    assert final.gradient_indices[4] == -22 and final.pressure[4] < 0
    assert continuation["all_prior_node4_pressures_positive"] is True
    assert continuation["stop_reason"] == "first_nonpositive_node4_pressure"


def test_boundaries_partition_the_complete_118_step_first_hit_without_a_reset(report):
    continuation = report["B40_continuation"]
    boundaries = continuation["boundaries"]
    assert len(boundaries) == continuation["boundary_count"] == 59
    previous, count = continuation["initial"], 0
    for ordinal, boundary in enumerate(boundaries, 1):
        assert boundary["ordinal"] == ordinal and boundary["initial"] == previous
        length = boundary["step_count"]
        assert length > 0
        selected = continuation["steps"][count : count + length]
        assert (
            selected[0]["before"] == previous
            and selected[-1]["after"] == boundary["endpoint"]
        )
        assert all(
            step["pressure"] == boundary["source_pressure"]["pressure"]
            for step in selected
        )
        assert all(step["before"]["epi"] == previous["epi"] for step in selected)
        count += length
        assert boundary["cumulative_steps"] == count
        previous = boundary["endpoint"]
    assert count == 118 and previous == continuation["endpoint"]
    assert (
        continuation["all_centered_energies_within_closure"]
        is continuation["within_proven_pressure_index"]
        is True
    )


def test_signed_prefix_repayment_is_an_overshoot_not_a_complete_vector_return(report):
    section = report["B40_continuation"]
    initial, endpoint = _exact(section["initial"]), _exact(section["endpoint"])
    baseline = _exact(section["B27_area_reference_state"])
    area, net_area = _difference(endpoint, initial), _difference(endpoint, baseline)
    assert section["local_nodal_area"] == area
    assert section["B27_nodal_area"] == net_area
    assert section["local_mean_area"] == _mean(area)
    assert section["B27_mean_area"] == _mean(net_area)
    assert (
        section["B27_incoming_mean_area"]
        == _mean(_difference(initial, baseline))
        == -F(275, 3 * 2**114)
    )
    prefixes = tuple(
        _mean(_difference(_exact(step["after"]), baseline)) for step in section["steps"]
    )
    assert section["B27_mean_area_prefixes"] == prefixes
    first = section["first_B27_mean_repayment_step"]
    assert first == 59
    assert all(value < 0 for value in prefixes[: first - 1]) and prefixes[first - 1] > 0
    assert section["B27_mean_repayment_exact_zero"] is False
    assert all(
        any(_difference(_exact(step["after"]), baseline)) for step in section["steps"]
    )
    assert section["complete_return_observed"] is False
    assert section["mean_source_area"] == -F(118, 6 * 2**113)
    assert (
        section["local_mean_area"]
        == section["mean_source_area"] + section["mean_rounding_area"]
    )
    assert section["mean_carry_feedback_area"] == section["mean_identity_residual"] == 0
    assert sum(
        F(step["pressure"][i]) / 16 for step in section["steps"] for i in range(6)
    ) / 6 == _mean(area)


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "endpoint_carry",
        "phase",
        "bound",
        "flag",
        "ancestor_pressure",
        "lineage",
    ),
)
def test_corrupted_parent_or_ancestor_cannot_become_a_new_continuation(parents, change):
    values = deepcopy(parents)
    parent = values[0]
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "endpoint_carry":
        parent["source"]["endpoint_state"]["remainder"] = ["0"] * 6
    elif change == "phase":
        parent["source"]["phase"][4] = 0.0
    elif change == "bound":
        parent["B35_uniform_tube"]["reference"]["energy_bound"] = "0"
    elif change == "flag":
        parent["live_provenance_certified"] = True
    elif change == "ancestor_pressure":
        values[1]["continuation"]["steps"][0]["pressure"][4] = 0.0
    else:
        parent["source_input_evidence"]["producer_manifest"]["claim_id"] = "unrelated"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_coupled_passage(*values)


def test_campaign_preserves_inputs_and_never_executes_a_graph_operator_word(
    parents, monkeypatch
):
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "a detached first-passage campaign must not execute graph operators"
        )

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    saved = deepcopy(parents)
    result = campaign.analyze_c6_winding_coupled_passage(*parents)
    assert parents == saved
    assert (
        result["new_graph_trajectories"] == 0
        and result["new_conditional_numerical_steps"] == 118
    )


def test_report_retains_the_scope_and_json_is_finite(report):
    for key in (
        "runtime_executed",
        "live_provenance_certified",
        "original_tail_reachability_certified",
        "infinite_band_invariance_certified",
        "full_runtime_stability_certified",
        "static_balance_is_temporal_itinerary",
    ):
        assert report[key] is False
    assert "tube" not in report["B39_finite_pressure_passage"]["band_horizon"]
    assert "base_tube" not in report["B37_self_consistent_closure"]
    assert "closure" not in report["B38_static_pressure_balance"]
    json.dumps(campaign._payload(report), allow_nan=False)


def _cli(tmp_path, monkeypatch):
    paths = tuple(tmp_path / name for name in campaign.INPUT_NAMES)
    for path, source in zip(paths, INPUTS, strict=True):
        if not source.is_file():
            pytest.skip("the retained B36 through B26 evidence chain is unavailable")
        path.write_bytes(source.read_bytes())
    output = tmp_path / "passage.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["passage", "--history-dir", str(tmp_path), "--output", str(output)],
    )
    return paths, output


def test_cli_binds_all_seven_original_byte_hashes_and_its_own_source_manifest(
    tmp_path, monkeypatch, report
):
    paths, output = _cli(tmp_path, monkeypatch)
    before = tuple(path.read_bytes() for path in paths)
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_coupled_passage", lambda *args: deepcopy(report)
    )
    provenance = ("a" * 40, True, "sha256:" + "b" * 64)
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: provenance
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-coupled-pressure-passage"
    assert result["manifest"]["git_sha"] == provenance[0]
    assert len(result["input_evidence"]) == 7
    for evidence, original in zip(result["input_evidence"], before, strict=True):
        assert evidence["sha256"] == hashlib.sha256(original).hexdigest()
        assert evidence["producer_manifest"] != result["manifest"]
    assert tuple(path.read_bytes() for path in paths) == before


@pytest.mark.parametrize("changed_index", range(1, 7))
def test_cli_checks_each_historical_byte_link_before_analysis(
    tmp_path, monkeypatch, changed_index
):
    paths, output = _cli(tmp_path, monkeypatch)
    changed = paths[changed_index]
    changed.write_bytes(changed.read_bytes() + b"\n")

    def forbidden(*args):
        raise AssertionError("historical byte corruption must fail before analysis")

    monkeypatch.setattr(campaign, "analyze_c6_winding_coupled_passage", forbidden)
    with pytest.raises(ValueError, match="lineage hashes"):
        campaign.main()
    assert not output.exists()


@pytest.mark.parametrize("change", ("source", "input"))
def test_cli_rejects_source_or_input_change_during_the_campaign(
    tmp_path, monkeypatch, report, change
):
    paths, output = _cli(tmp_path, monkeypatch)
    original = ("a" * 40, True, "sha256:" + "b" * 64)
    later = ("a" * 40, True, "sha256:" + "c" * 64) if change == "source" else original
    values = iter((original, later))
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: next(values)
    )

    def analysis(*args):
        if change == "input":
            paths[0].write_bytes(paths[0].read_bytes() + b"\n")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_coupled_passage", analysis)
    with pytest.raises(RuntimeError, match="changed during"):
        campaign.main()
    assert not output.exists()


def test_cli_cannot_overwrite_any_historical_input(tmp_path, monkeypatch):
    paths, _ = _cli(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["passage", "--history-dir", str(tmp_path), "--output", str(paths[3])],
    )
    before = paths[3].read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert paths[3].read_bytes() == before
