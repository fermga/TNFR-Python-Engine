"""Inherited carry, exact itinerary feasibility and generated source reversals."""

from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path

import pytest

from benchmarks import c6_winding_carry_itinerary as campaign

INPUT = Path(__file__).resolve().parents[2] / "artifacts/research/c6_winding_pressure_lattice.json"


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("the retained local B26 pressure-lattice report is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_winding_carry_itinerary(parent)


def _offsets(state):
    return tuple((F(value) - F(1, 2)) * 2**54 for value in state["epi"])


def test_retained_carry_is_preserved_and_two_boundaries_are_derived(report, parent):
    initial = report["source"]["inherited_state"]
    retained = parent["balanced_boundary"]["first_exit_step"]["after"]
    assert campaign._payload(initial) == retained
    assert any(initial["remainder"])
    assert report["source"]["retained_B26_prefix_replayed"] is True
    assert report["boundary_budget"] == len(report["boundaries"]) == 2
    first, second = report["boundaries"]
    assert first["initial"] == initial and second["initial"] == first["endpoint"]
    assert _offsets(initial) == (-1, 0, 0, 0, 0, -1)
    assert (first["horizon"]["max_unchanged_steps"], first["horizon"]["first_exit_step"]) == (2, 3)
    assert (second["horizon"]["max_unchanged_steps"], second["horizon"]["first_exit_step"]) == (11, 12)
    assert first["horizon"]["first_exit_leaves_cell"] == (False, False, False, True, True, False)
    assert second["horizon"]["first_exit_leaves_cell"] == (False, False, False, False, False, True)
    assert _offsets(first["endpoint"]) == (-1, 0, 0, -1, 2, -1)
    assert _offsets(second["endpoint"]) == (-1, 0, 0, -1, 2, -2)
    assert all(not boundary["endpoint_in_original_stencil"] for boundary in (first, second))


def test_generated_pressure_sign_changes_do_not_hide_the_exact_nodal_areas(report):
    first, second = report["boundaries"]
    assert first["source_pressure"]["mean_pressure"] == -F(1, 6 * 2**109)
    assert first["refreshed_pressure"]["mean_pressure"] == F(1, 2**110)
    assert second["source_pressure"] == first["refreshed_pressure"]
    assert second["refreshed_pressure"]["mean_pressure"] == -F(1, 6 * 2**108)
    assert first["prefix_balances"][-1]["mean_nodal_area"] == -F(1, 2**114)
    assert second["prefix_balances"][-1]["mean_nodal_area"] == F(3, 2**112)
    continuation = report["continuation"]
    assert continuation["step_count"] == 15
    assert continuation["mean_nodal_area"] == continuation["mean_reconstructed_change"] == F(11, 2**114)
    previous = report["source"]["inherited_state"]
    area = [F(0)] * 6
    for step in continuation["steps"]:
        assert step["before"] == previous
        assert step["capacity"] == (1.0,) * 6 and step["timestep"] == 1 / 16
        assert step["exact_increment"] == tuple(F(p) / 16 for p in step["pressure"])
        assert step["nodal_balance_residual"] == (0,) * 6
        area = [total + value for total, value in zip(area, step["exact_increment"], strict=True)]
        previous = step["after"]
    assert tuple(area) == continuation["itinerary"]["total_nodal_area"]
    assert sum(area) / 6 == F(11, 2**114)
    assert previous == continuation["endpoint"]
    for prefix in continuation["prefix_balances"]:
        assert prefix["mean_nodal_area"] == prefix["mean_reconstructed_change"]
        assert prefix["mean_identity_residual"] == 0
        assert prefix["nodal_identity_residual"] == (0,) * 6


def test_inverse_itinerary_accepts_inherited_carry_but_refuses_zero_reset(report):
    continuation = report["continuation"]
    itinerary = continuation["itinerary"]
    assert itinerary["feasible"] is True
    assert itinerary["zero_initial_carry_feasible"] is False
    assert itinerary["visible_closed"] is False
    assert itinerary["conditional_carried_cycle"] is False
    assert continuation["supplied_initial_carry_feasible"] is True
    assert continuation["supplied_initial_carry_coordinate_membership"] == (True,) * 6
    initial = report["source"]["inherited_state"]
    exact = tuple(F(x) + r for x, r in zip(initial["epi"], initial["remainder"], strict=True))
    for value, cell in zip(exact, itinerary["coordinates"], strict=True):
        assert cell["lower"] <= value <= cell["upper"]
        assert value > cell["lower"] or cell["lower_closed"]
        assert value < cell["upper"] or cell["upper_closed"]
        index = value * 2**3222
        assert index.denominator == 1
        assert cell["first_grid_index"] <= index <= cell["last_grid_index"]
    assert len(itinerary["cumulative_nodal_area"]) == 16
    assert itinerary["cumulative_nodal_area"][0] == (0,) * 6
    assert itinerary["witness_initial"] is not None
    assert len(itinerary["witness_prefix_balances"]) == 15
    assert itinerary["pressure_provenance_certified"] is False
    assert itinerary["runtime_provenance_certified"] is False


def test_twelve_visible_self_loops_are_feasible_but_thirteen_are_not(report):
    twelve, thirteen = report["balanced_visible_self_loops"]
    assert (twelve["length"], thirteen["length"]) == (12, 13)
    assert twelve["epi"] == thirteen["epi"] and twelve["pressure"] == thirteen["pressure"]
    for control in (twelve, thirteen):
        itinerary = control["itinerary"]
        assert control["pressure_mean"] == 0
        assert itinerary["visible_closed"] is True
        assert itinerary["conditional_carried_cycle"] is False
        assert itinerary["zero_initial_carry_feasible"] is False
        expected = tuple(F(control["length"], 16) * F(value) for value in control["pressure"])
        assert itinerary["total_nodal_area"] == expected
        assert sum(expected) == 0 and any(expected)
    assert twelve["itinerary"]["feasible"] is True
    assert twelve["itinerary"]["witness_initial"] is not None
    assert len(twelve["itinerary"]["witness_prefix_balances"]) == 12
    assert thirteen["itinerary"]["feasible"] is False
    assert thirteen["itinerary"]["witness_initial"] is None
    assert thirteen["itinerary"]["witness_prefix_balances"] is None
    assert any(not cell["feasible"] for cell in thirteen["itinerary"]["coordinates"])


def test_four_state_axis_separator_excludes_return_despite_pressure_mean_reversal(report):
    certificate = report["four_state_drift"]
    assert certificate["status"] == "axis_separator"
    assert (certificate["node"], certificate["sign"]) == (1, -1)
    observation = certificate["observation"]
    assert observation["functional"] == (0, -1, 0, 0, 0, 0)
    pressure = tuple(record[1] for record in observation["pressure_vectors"])
    assert len(set(pressure)) == 1 and pressure[0] < 0
    assert observation["functional_projections"] == tuple(-F(value) for value in pressure)
    assert observation["width"] == F(3, 2**55)
    assert (observation["max_confined_steps"], observation["escape_step_bound"]) == (842, 843)
    assert certificate["conditional_class_escape_certified"] is True
    assert certificate["pressure_provenance_certified"] is False
    assert certificate["positive_band_exit_certified"] is False


def test_generated_pressure_is_reused_between_boundaries_without_live_execution(parent, monkeypatch):
    from tnfr.operators import event_runtime, nodal_remainder_runtime
    from tnfr.physics import c6_pressure_lattice as owner

    def forbidden(*args, **kwargs):
        raise AssertionError("the finite itinerary must not execute a graph word")

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden)
    actual, points = owner.fused_dnfr.compute_fused_gradients_symmetric, []

    def tracked(**kwargs):
        epi = tuple(map(float, kwargs["epi"]))
        if any(epi):
            points.append(epi)
        return actual(**kwargs)

    monkeypatch.setattr(owner.fused_dnfr, "compute_fused_gradients_symmetric", tracked)
    original = deepcopy(parent)
    campaign.analyze_c6_winding_carry_itinerary(parent)
    assert parent == original
    assert len(points) == len(set(points)) == 4


@pytest.mark.parametrize("change", ("claim", "scope", "balanced_bool", "carry_reset", "carry_change", "pressure", "horizon", "prefix", "weights", "cached_lattice"))
def test_parent_pressure_carry_or_scope_corruption_fails_closed(parent, change):
    changed = deepcopy(parent)
    boundary = changed["balanced_boundary"]
    if change == "claim":
        changed["manifest"]["claim_id"] = "unrelated-source"
    elif change == "scope":
        changed["tail_epi_state_reachability_certified"] = True
    elif change == "balanced_bool":
        changed["balanced_stencil_indices"] = [True]
    elif change == "carry_reset":
        boundary["first_exit_step"]["after"]["remainder"] = ["0"] * 6
    elif change == "carry_change":
        boundary["first_exit_step"]["after"]["remainder"][0] = "0"
    elif change == "pressure":
        boundary["refreshed_pressure"]["pressure"][0] = 0.0
    elif change == "horizon":
        boundary["horizon"]["first_exit_step"] = 7
    elif change == "prefix":
        boundary["shared_prefix_balances"][0]["mean_nodal_area"] = "1"
    elif change == "weights":
        changed["lattice_reference"]["source"]["epi_weight"] = "1/2"
    else:
        changed["lattice_reference"]["nonnegative_min_sum"] = 0
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_carry_itinerary(changed)


def test_itinerary_report_keeps_detached_scope_and_compact_alternative_witnesses(report):
    for flag in ("runtime_executed", "live_provenance_certified", "original_tail_reachability_certified",
                 "future_compensation_cycle_certified", "positive_band_exit_certified"):
        assert report[flag] is False
    assert report["new_graph_trajectories"] == 0
    assert "witness_sequence" not in report["continuation"]["itinerary"]
    assert all("witness_sequence" not in control["itinerary"] for control in report["balanced_visible_self_loops"])
    json.dumps(campaign._payload(report), allow_nan=False)
