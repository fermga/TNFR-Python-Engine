"""Static null-source pressure obstruction without extending a trajectory."""

from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path

import pytest

from benchmarks import c6_winding_pressure_equilibrium as campaign

INPUT = Path(__file__).resolve().parents[2] / "artifacts/research/c6_winding_phase_kernel.json"


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("the retained local B20 comparison is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_pressure_equilibrium(parent)


def test_only_selected_retained_case_is_validated_without_graph_execution(parent, monkeypatch):
    from benchmarks import c6_winding_joint_domain as producer
    from benchmarks import c6_winding_phase_response as preparation
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("the static report must not execute a graph or operator event")

    monkeypatch.setattr(producer, "run_c6_joint_case", forbidden)
    monkeypatch.setattr(preparation, "prepare_c6_phase_response", forbidden)
    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden)
    actual, calls = campaign.analyze_c6_defect_case, []

    def tracked(case):
        calls.append(case["mode"])
        return actual(case)

    monkeypatch.setattr(campaign, "analyze_c6_defect_case", tracked)
    before = deepcopy(parent)
    campaign.analyze_c6_pressure_equilibrium(parent)
    assert parent == before
    assert calls == ["null"]


def test_source_is_actual_post_il_phase_and_default_pressure(parent, report):
    source = report["source"]
    retained = parent["current_cases"][0]["cycles"][0]["post_il_capture"]
    assert tuple(map(F, source["phase"])) == tuple(map(F, retained["phase"]))
    assert tuple(map(F, source["phase"])) != tuple(map(F, parent["current_cases"][0]["initial_capture"]["phase"]))
    assert source["epi"] == (.5,) * 6 and source["capacity"] == (1.0,) * 6
    assert source["event_prefix"] == ("coupling", "coherence")
    assert source["shared_cpu_pressure_matches_capture"] is True
    assert tuple(map(F, source["pressure"])) == tuple(map(F, retained["snapshot"]["stored_pressure"]))
    assert source["pressure_sum"] == F(25, 2**112)
    assert source["pressure_mean"] == F(25, 6 * 2**112)
    assert source["exact_midpoint_sum_rational"] == source["exact_midpoint_sum_pi_coefficient"] == 0
    assert dict(source["normalized_weights"])["epi"] == F(3299399280161697, 2**54)


def test_all_six_exact_inverse_cells_exclude_the_entire_epi_gradient_lattice(report):
    obstruction = report["equilibrium_obstruction"]
    assert obstruction["gradient_quantum"] == F(1, 2**58)
    assert obstruction["no_zero_pressure"] is True
    expected = ((8, 7), (-1, -2), (-42, -43), (85, 84), (-168, -169), (121, 120))
    for row, (first, last) in zip(obstruction["rows"], expected, strict=True):
        assert (row["first_grid_index"], row["last_grid_index"]) == (first, last)
        assert row["grid_excluded"] is True
        cell = row["cancellation_cell"]
        assert row["inverse_gradient_lower"] * obstruction["epi_weight"] == cell["lower"]
        assert row["inverse_gradient_upper"] * obstruction["epi_weight"] == cell["upper"]
        # Strict bracketing here makes endpoint tie policy immaterial: no
        # integer can lie in this interval, even before applying parity.
        lower = row["inverse_gradient_lower"] / obstruction["gradient_quantum"]
        upper = row["inverse_gradient_upper"] / obstruction["gradient_quantum"]
        assert last < lower <= upper < first
        assert row["target_epi_pressure"] == -row["phase_contribution"]
    assert report["fixed_positive_step_convergence_excluded"] is True


def test_held_source_carry_exits_its_first_cell_on_step_six_without_leaving_band(report):
    horizon = report["conditional_cell_horizon"]
    assert horizon["max_unchanged_steps"] == 5
    assert horizon["first_exit_step"] == 6
    exact = tuple(map(F, report["source"]["epi"]))
    increment = tuple(F(p) / 16 for p in report["source"]["pressure"])
    assert horizon["exact_increment"] == increment
    assert horizon["mean_increment"] == report["source"]["pressure_mean"] / 16
    assert horizon["unchanged_endpoint"] == tuple(x + 5 * a for x, a in zip(exact, increment, strict=True))
    assert horizon["first_exit_exact"] == tuple(x + 6 * a for x, a in zip(exact, increment, strict=True))
    assert all(cell["lower"] <= y <= cell["upper"] for cell, y in zip(
        horizon["source_cells"], horizon["unchanged_endpoint"], strict=True,
    ))
    outside = tuple(not (cell["lower"] <= y <= cell["upper"]) for cell, y in zip(
        horizon["source_cells"], horizon["first_exit_exact"], strict=True,
    ))
    assert outside == (False, False, False, False, False, True)
    assert horizon["first_exit_leaves_cell"] == outside
    assert horizon["first_exit_leaves_band"] == (False,) * 6
    assert float(horizon["first_exit_exact"][5]) == math.nextafter(.5, -math.inf)


def test_scope_and_serialization_do_not_promote_fixed_phase_to_the_event_schedule(report):
    for name in ("runtime_executed", "live_provenance_certified", "B23_phase_identity_certified",
                 "positive_band_exit_certified", "signed_mean_convergence_decided", "full_runtime_stability_certified"):
        assert report[name] is False
    assert report["new_graph_trajectories"] == 0
    assert "not the B23 schedule" in report["cell_horizon_conditions"]
    json.dumps(campaign._payload(report), allow_nan=False)


@pytest.mark.parametrize("change", ("seed", "horizon", "order", "phase", "pressure", "support", "band", "nonrepresented"))
def test_changed_source_or_parent_controls_fail_closed(parent, change):
    changed = deepcopy(parent)
    case = changed["current_cases"][0]
    capture = case["cycles"][0]["post_il_capture"]
    if change == "seed":
        changed["manifest"]["seed"] = 18
    elif change == "horizon":
        changed["cycle_count_per_case"] = 3
    elif change == "order":
        changed["current_cases"].reverse()
    elif change == "phase":
        capture["phase"][0] = "1/8"
    elif change == "pressure":
        capture["snapshot"]["stored_pressure"][0] = "0"
    elif change == "support":
        capture["snapshot"]["conductance"][0][2] = "2"
    elif change == "band":
        case["admission_band"]["positive_epi_lower"] = "1/4"
    else:
        capture["phase"][0] = "1/3"
    with pytest.raises(ValueError):
        campaign.analyze_c6_pressure_equilibrium(changed)


def test_changed_shared_cpu_source_is_not_accepted_as_historical_pressure(parent, monkeypatch):
    actual = campaign.compute_fused_gradients_symmetric

    def changed(**kwargs):
        pressure = actual(**kwargs).copy()
        pressure[0] = math.nextafter(float(pressure[0]), math.inf)
        return pressure

    monkeypatch.setattr(campaign, "compute_fused_gradients_symmetric", changed)
    with pytest.raises(ValueError, match="canonical CPU realization"):
        campaign.analyze_c6_pressure_equilibrium(parent)
