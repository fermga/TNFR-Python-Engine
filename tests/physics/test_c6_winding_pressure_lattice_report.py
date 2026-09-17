"""Actual local compensation and its first carried-cell boundary."""

from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_lattice as campaign

INPUT = Path(__file__).resolve().parents[2] / "artifacts/research/c6_winding_phase_orbit.json"


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("the retained local B25 phase orbit is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_winding_pressure_lattice(parent)


def test_complete_phase_orbit_is_replayed_and_the_static_stencil_is_fixed(report):
    source = report["phase_source"]
    assert source["complete_retained_orbit_replayed"] is True
    assert source["new_phase_search_executed"] is False
    assert (source["preperiod"], source["period"]) == (89, 1)
    assert source["terminal"][0].hex() == "0x1.0b8fb3e3956cbp-55"
    points = report["stencil"]
    assert len(points) == 13 and report["static_pressure_points"] == 14
    assert points[0]["name"] == "half" and points[0]["observation"]["epi"] == (.5,) * 6
    for node in range(6):
        for offset, name, direction in ((1, "down", -math.inf), (2, "up", math.inf)):
            point = points[2 * node + offset]
            assert point["name"] == f"node_{node}_{name}"
            expected = tuple(math.nextafter(.5, direction) if i == node else .5 for i in range(6))
            assert point["observation"]["epi"] == expected


def test_actual_pressure_sums_and_integer_cycle_compatibility(report):
    expected_sums = (-8, 0, -40, -8, -32, -16, -24, -8, -8, -8, -8, -24, -56)
    weight = report["lattice_reference"]["source"]["epi_weight"]
    source = tuple(map(F, report["lattice_reference"]["sources"]))
    for point, expected_sum in zip(report["stencil"], expected_sums, strict=True):
        item = point["observation"]
        x = tuple(map(F, item["epi"]))
        q = tuple((x[(i - 1) % 6] + x[(i + 1) % 6]) / 2 - x[i] for i in range(6))
        assert item["gradient"] == q
        assert item["gradient_indices"] == tuple(value * 2**55 for value in q)
        assert sum(q) == item["gradient_index_sum"] == 0
        assert item["epi_contributions"] == tuple(float(weight * value) for value in q)
        assert sum(map(F, item["pressure"])) == F(expected_sum, 2**112)
        assert item["mean_epi_reduction_error"] == 0
        assert item["mean_pressure"] == (
            item["mean_phase_contribution"] + item["mean_epi_reduction_error"] + item["mean_assembly_error"]
        )
        assert item["mean_identity_residual"] == 0
        assert item["assembly_error"] == tuple(
            F(p) - a - F(g) for p, a, g in zip(item["pressure"], source, item["epi_contributions"], strict=True)
        )
        assert all(item[name] for name in ("scalar_reducer_agreement", "vector_reducer_agreement", "fused_pressure_agreement"))
        assert min(item["pressure"]) < 0 < max(item["pressure"])


def test_lattice_sign_constraints_exclude_cartesian_corners_not_correlated_sets(report):
    reference = report["lattice_reference"]
    assert reference["epi_quantum"] == F(1, 2**54)
    assert reference["gradient_quantum"] == F(1, 2**55)
    assert (reference["nonnegative_min_sum"], reference["nonpositive_max_sum"]) == (2, -4)
    assert tuple((row["nonnegative_min_index"], row["nonpositive_max_index"]) for row in reference["rows"]) == (
        (2, 1), (0, -1), (-5, -6), (11, 10), (-21, -22), (15, 14),
    )
    assert reference["every_pressure_has_positive"] and reference["every_pressure_has_negative"]
    assert reference["no_cartesian_trap"] is True
    assert report["correlated_invariant_set_excluded"] is False


def test_canonical_compensation_is_exact_until_the_first_cell_exit_then_is_lost(report):
    assert report["balanced_stencil_indices"] == (1,)
    balanced = report["stencil"][1]["observation"]
    assert balanced["mean_pressure"] == 0
    assert balanced["mean_assembly_error"] == -balanced["mean_phase_contribution"] == F(1, 6 * 2**109)
    boundary = report["balanced_boundary"]
    assert boundary["horizon"]["max_unchanged_steps"] == 5
    assert boundary["horizon"]["first_exit_step"] == 6
    assert boundary["shared_replay_before_states_unchanged"] == (True,) * 6
    assert boundary["shared_replay_intermediate_states_unchanged"] == (True,) * 5
    for prefix in boundary["shared_prefix_balances"]:
        assert prefix["mean_nodal_area"] == prefix["mean_reconstructed_change"] == prefix["mean_identity_residual"] == 0
        assert prefix["nodal_identity_residual"] == (0,) * 6
    initial, last = boundary["initial"], boundary["last_unchanged_state"]
    assert initial["epi"] == last["epi"] == balanced["epi"]
    assert initial["remainder"] == (0,) * 6 and any(last["remainder"])
    step = boundary["first_exit_step"]
    assert step["before"] == last and step["pressure"] == balanced["pressure"]
    assert step["after"]["epi"] == (math.nextafter(.5, -math.inf), .5, .5, .5, .5, math.nextafter(.5, -math.inf))
    assert step["nodal_balance_residual"] == (0,) * 6
    assert boundary["reconstructed_mean_change_through_exit"] == 0
    assert boundary["visible_mean_change_through_exit"] == -F(1, 6 * 2**54)
    assert boundary["pressure_balanced_after_refresh"] is False
    assert boundary["refreshed_pressure"]["epi"] == step["after"]["epi"]
    assert boundary["refreshed_pressure"]["mean_pressure"] == -F(1, 6 * 2**109)


def test_exact_separating_functional_bounds_stencil_residence_without_changing_pressure(report):
    certificate = report["finite_class_drift"]
    observation = certificate["observation"]
    b = tuple(map(F, report["stencil"][1]["observation"]["pressure"]))
    epsilon = certificate["separator_epsilon"]
    assert epsilon == 1 / (1 + 2 * certificate["ratio_bound"])
    functional = tuple(-1 + epsilon * value for value in b)
    assert observation["functional"] == functional
    projections = tuple(sum((a * F(p) for a, p in zip(functional, point["observation"]["pressure"], strict=True)), F(0))
                        for point in report["stencil"])
    assert observation["functional_projections"] == projections and min(projections) > 0
    assert projections[1] == epsilon * sum(value * value for value in b)
    minimum, width = observation["minimum_pressure_projection"], observation["width"]
    count = observation["max_confined_steps"]
    assert count == 15653039385204422266
    assert count * F(1, 16) * minimum <= width < (count + 1) * F(1, 16) * minimum
    assert observation["escape_step_bound"] == count + 1
    assert certificate["conditional_class_escape_certified"] is True
    assert certificate["pressure_provenance_certified"] is False
    assert certificate["positive_band_exit_certified"] is False


def test_actual_pressure_evaluated_once_per_point_with_no_new_word_or_phase_search(parent, monkeypatch):
    from benchmarks import c6_winding_phase_orbit as previous
    from tnfr.operators import event_runtime, nodal_remainder_runtime
    from tnfr.physics import c6_pressure_lattice as owner

    def forbidden(*args, **kwargs):
        raise AssertionError("the local audit must not discover another orbit or execute a graph word")

    monkeypatch.setattr(previous, "_discover", forbidden)
    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden)
    actual, values = owner.fused_dnfr.compute_fused_gradients_symmetric, []

    def tracked(**kwargs):
        epi = tuple(map(float, kwargs["epi"]))
        if any(epi):
            values.append(epi)
        return actual(**kwargs)

    monkeypatch.setattr(owner.fused_dnfr, "compute_fused_gradients_symmetric", tracked)
    before = deepcopy(parent)
    campaign.analyze_c6_winding_pressure_lattice(parent)
    assert parent == before
    assert len(values) == len(set(values)) == 14


@pytest.mark.parametrize("change", ("claim", "scope", "count", "preperiod_bool", "initial", "transition", "margin", "method", "weights", "source_bias"))
def test_parent_phase_or_source_corruption_fails_closed(parent, change):
    changed = deepcopy(parent)
    if change == "claim":
        changed["manifest"]["claim_id"] = "unrelated-source"
    elif change == "scope":
        changed["future_runtime_certified"] = True
    elif change == "count":
        changed["discovery_transitions"] = 91
    elif change == "preperiod_bool":
        changed["phase_orbit"]["preperiod"] = True
    elif change == "initial":
        changed["phase_orbit"]["phase_states"][0][0] = .125
    elif change == "transition":
        changed["phase_orbit"]["phase_states"][17][0] = .125
    elif change == "margin":
        changed["phase_orbit"]["steps"][1]["edge_margins"][0][0] = .125
    elif change == "method":
        changed["phase_orbit"]["steps"][3]["coherence_methods"][0] = "unverified"
    elif change == "weights":
        changed["source"]["normalized_weights"][1][1] = "1/2"
    else:
        changed["periodic_phase_source"]["mean_source"] = "0"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_pressure_lattice(changed)


def test_scope_retains_missing_tail_reachability_and_future_compensation(report):
    for flag in ("runtime_executed", "live_provenance_certified", "tail_epi_state_reachability_certified",
                 "future_mean_compensation_verified", "correlated_invariant_set_excluded", "positive_band_exit_certified"):
        assert report[flag] is False
    assert report["new_graph_trajectories"] == 0
    json.dumps(campaign._payload(report), allow_nan=False)
