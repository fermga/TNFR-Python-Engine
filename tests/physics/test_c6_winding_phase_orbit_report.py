"""The inherited null phase closes exactly without a live EPI campaign."""

import json
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_phase_orbit as campaign

INPUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/research/c6_winding_phase_kernel.json"
)
BASE_HEX = (
    "0x0.0p+0",
    "0x1.0c152382d7365p+0",
    "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1",
    "0x1.0c152382d7365p+2",
    "0x1.4f1a6c638d03fp+2",
)


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("the retained local B20 comparison is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_winding_phase_orbit(parent)


def test_bounded_discovery_finds_first_repeat_and_preserves_every_transient(report):
    assert report["status"] == "exact_phase_cycle"
    assert report["discovery_ceiling"] == 256
    assert report["discovery_transitions"] == 90
    orbit = report["phase_orbit"]
    assert orbit["preperiod"] == 89 and orbit["period"] == 1
    states = orbit["phase_states"]
    assert len(states) == 91 and len(orbit["steps"]) == 90
    signatures = tuple(tuple(value.hex() for value in phase) for phase in states)
    assert signatures[0] == BASE_HEX
    assert len(set(signatures[:-1])) == 90
    assert signatures[-1] == signatures[89]
    assert signatures[-1][0] == "0x1.0b8fb3e3956cbp-55"
    assert all(row[1:] == BASE_HEX[1:] for row in signatures)
    for before, after, step in zip(
        states[:-1], states[1:], orbit["steps"], strict=True
    ):
        assert before == step["phase_before"] and after == step["phase_after_coherence"]
        assert (
            tuple(value.hex() for value in step["phase_after_coupling"][1:])
            == BASE_HEX[1:]
        )
    terminal = orbit["steps"][-1]
    assert terminal["phase_after_coupling"][0].hex() == "0x1.ab75f42bdf52ep-55"
    assert terminal["phase_after_coupling"] != terminal["phase_after_coherence"]


def test_first_two_phase_transitions_bind_the_historical_actual_operators(
    parent, report
):
    assert len(report["source"]["first_two_retained_bindings"]) == 2
    for recorded, step, binding in zip(
        parent["current_cases"][0]["cycles"],
        report["phase_orbit"]["steps"][:2],
        report["source"]["first_two_retained_bindings"],
        strict=True,
    ):
        assert all(
            binding[name] for name in ("before_matches", "UM_matches", "IL_matches")
        )
        assert tuple(map(F, step["phase_after_coupling"])) == tuple(
            map(F, recorded["um"]["raw_capture"]["phase"])
        )
        assert tuple(map(F, step["phase_after_coherence"])) == tuple(
            map(F, recorded["il"]["raw_capture"]["phase"])
        )


def test_all_transition_boundaries_retain_unit_cycle_phase_support(report):
    for step in report["phase_orbit"]["steps"]:
        assert step["coherence_methods"] == ("exact_two_neighbor_midpoint",) * 6
        assert len(step["edge_margins"]) == len(step["nonedge_margins"]) == 3
        for edges, nonedges in zip(
            step["edge_margins"], step["nonedge_margins"], strict=True
        ):
            assert len(edges) == 6 and len(nonedges) == 9
            assert min(edges) > 0.52 and min(nonedges) > 0.52
        assert step["um_phase_factor"].hex() == "0x1.ee7eea04ddca0p-3"
        assert step["il_phase_factor"] == 0.3


def test_closed_phase_source_has_nonzero_exact_bias_and_no_zero_pressure_point(report):
    assert report["conditional_phase_periodic"] is True
    assert report["every_cycle_phase_excludes_zero_pressure"] is True
    (obstruction,) = report["cycle_pressure_obstructions"]
    assert (
        obstruction["phase"]
        == report["phase_orbit"]["steps"][-1]["phase_after_coherence"]
    )
    assert obstruction["no_zero_pressure"] is True
    assert tuple(
        (row["first_grid_index"], row["last_grid_index"]) for row in obstruction["rows"]
    ) == (
        (16, 15),
        (-5, -6),
        (-42, -43),
        (85, 84),
        (-168, -169),
        (117, 116),
    )
    source = tuple(row["phase_contribution"] for row in obstruction["rows"])
    assert sum(map(F, source)) == -F(1, 2**109)
    assert (
        sum(row["phase_response"]["delta_rational"] for row in obstruction["rows"]) == 0
    )
    assert (
        sum(
            row["phase_response"]["delta_pi_coefficient"] for row in obstruction["rows"]
        )
        == 0
    )
    budget = report["periodic_phase_source"]
    assert budget["phase_contributions"] == (source,)
    assert budget["period"] == 1 and budget["block_duration"] == 0.25
    assert budget["mean_sources"] == (-F(1, 6 * 2**109),)
    assert budget["mean_source"] == -F(1, 6 * 2**109)
    assert budget["phase_area_per_period"] == -F(1, 24 * 2**109)
    assert budget["centered_prefixes"] == (0, 0)
    assert budget["prefix_amplitude_bound"] == 0


def test_only_selected_source_is_validated_and_no_live_word_or_flow_runs(
    parent, monkeypatch
):
    from benchmarks import c6_winding_joint_domain as producer
    from benchmarks import c6_winding_phase_response as preparation
    from tnfr.operators import event_runtime, nodal_remainder_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "B25 must not execute a new graph campaign or operator word"
        )

    monkeypatch.setattr(producer, "run_c6_joint_case", forbidden)
    monkeypatch.setattr(preparation, "prepare_c6_phase_response", forbidden)
    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    actual, calls = campaign.analyze_c6_defect_case, []

    def tracked(case):
        calls.append(case["mode"])
        return actual(case)

    monkeypatch.setattr(campaign, "analyze_c6_defect_case", tracked)
    original = deepcopy(parent)
    campaign.analyze_c6_winding_phase_orbit(parent)
    assert calls == ["null"] and parent == original


def test_phase_closure_never_promotes_the_unobserved_epi_and_mean_conditions(report):
    assert report["phase_proposal_kernels_executed"] is True
    assert report["new_graph_trajectories"] == 0
    for flag in (
        "runtime_executed",
        "operator_word_admission_certified",
        "live_provenance_certified",
        "future_runtime_certified",
        "EPI_band_invariance_certified",
        "actual_mean_compensation_verified",
    ):
        assert report[flag] is False
    assert "has not been evolved or bounded" in report["remaining_mean_condition"]
    assert (
        report["source_budget_alignment"]
        == "Each quarter-duration block follows the cycle step's Coherence output"
    )
    json.dumps(campaign._payload(report), allow_nan=False)


@pytest.mark.parametrize(
    "change", ("seed", "horizon", "order", "initial_phase", "stage_phase", "support")
)
def test_parent_or_selected_phase_corruption_fails_closed(parent, change):
    changed = deepcopy(parent)
    case = changed["current_cases"][0]
    if change == "seed":
        changed["manifest"]["seed"] = 18
    elif change == "horizon":
        changed["cycle_count_per_case"] = 3
    elif change == "order":
        changed["current_cases"].reverse()
    elif change == "initial_phase":
        case["initial_capture"]["phase"][0] = "1/8"
    elif change == "stage_phase":
        case["cycles"][0]["um"]["raw_capture"]["phase"][0] = "1/3"
    else:
        case["cycles"][0]["um"]["raw_capture"]["snapshot"]["conductance"][0][2] = "2"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_phase_orbit(changed)
