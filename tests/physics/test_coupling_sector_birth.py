"""Prospective one-stage C5 closure under the existing Coupling policy.

The prepared path has successive fifth-turn gaps before a cycle exists. UM factors
move both ends of the target's existing edge, and only then score the candidate
closing edge. These finite observations test policy-selected sector birth, not
an endogenous support law, a sine lock or subsequent persistence. Si variations
audit an existing policy dependency; they do not propose a new dynamics.
"""

import math
from dataclasses import FrozenInstanceError

import pytest

from tests.joint_phase_helpers import execute_coupling_cycle_birth
from tnfr.operators.network_stage import STAGE_SCHEDULE_KEY, TWO_PHASE_JACOBI
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.phase_cycle_geometry import (
    derive_phase_chord_extension,
    derive_phase_cycle_geometry,
)
from tnfr.physics.phase_response import observe_phase_lock_source
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff


@pytest.fixture(scope="module")
def executions():
    """Execute each declared control once and retain pre/raw/refreshed states."""
    return {
        "default": execute_coupling_cycle_birth(),
        "si_only": execute_coupling_cycle_birth(candidate_si=0.0),
        "links_disabled": execute_coupling_cycle_birth(functional_links=False),
    }


def test_prospective_candidate_uses_snapshot_and_final_u3_gates(executions):
    case = executions["default"]
    graph = case["before"]
    assert derive_phase_cycle_geometry(graph).cycle_rank == 0
    initial = certify_phase_winding(graph, range(5))
    assert not initial.cycle_exists and initial.winding is None
    assert graph.graph["UM_CANDIDATE_COUNT"] == 0  # all eligible candidates
    assert "_node_sample" not in graph.graph
    assert "GLYPH_FACTORS" not in graph.graph
    assert "UM_COMPAT_THRESHOLD" not in graph.graph
    assert case["factors"]["UM_theta_push"] == 1 / (math.pi + 1)
    assert case["threshold"] == math.pi / (math.pi + 1)
    proposal = case["prediction"].target_proposals[0]
    assert proposal.compatible_neighbors == (1,)
    assert tuple(candidate.target for candidate in proposal.link_candidates) == (4,)
    assert proposal.effective_phase_limit == math.pi / 2
    assert proposal.compatibility_threshold == case["threshold"]
    assert abs(angle_diff(graph.nodes[4]["theta"], graph.nodes[0]["theta"])) < (
        math.pi / 2
    )
    assert proposal.write_vf and proposal.write_dnfr
    assert case["admission"].allowed
    assert [(edge.left, edge.right) for edge in case["prediction"].edges] == [(0, 4)]


def test_actual_um_creates_one_strictly_acute_winding_cycle(executions):
    case = executions["default"]
    graph, before = case["graph"], case["before"]
    assert set(graph.edges) - set(before.edges) == {(0, 4)}
    assert set(before.edges) <= set(graph.edges)
    assert graph.edges[0, 4]["weight"] == pytest.approx(case["score"], abs=1e-15)
    extension = derive_phase_chord_extension(
        derive_phase_cycle_geometry(before), derive_phase_cycle_geometry(graph)
    )
    assert extension.after.cycle_rank == 1
    assert extension.inherited_cycle_coordinates == ()
    created_cycle = tuple(
        extension.after.nodes[index] for index in extension.created_cycle
    )
    assert certify_phase_winding(graph, created_cycle).winding == -1
    winding = certify_phase_winding(graph, range(5))
    assert winding.winding == 1 and winding.u3_admissible
    assert winding.quantization_residual < 1e-15
    assert winding.minimum_u3_margin == pytest.approx(
        math.pi / 2 - case["max_gap"], abs=2e-15
    )
    assert winding.minimum_u3_margin > 0
    assert certify_phase_winding(graph, reversed(range(5))).winding == -1


def test_actual_phase_capacity_pressure_writes_and_stage_provenance(executions):
    case = executions["default"]
    graph, before = case["raw_graph"], case["before"]
    push = case["factors"]["UM_theta_push"]
    delta = math.tau / 5
    displacement = push * math.pi / 5
    expected_phases = (
        displacement,
        delta - displacement,
        2 * delta,
        3 * delta,
        4 * delta,
    )
    assert tuple(graph.nodes[node]["theta"] for node in graph) == pytest.approx(
        expected_phases, abs=2e-15
    )
    assert tuple(graph.nodes[node]["nu_f"] for node in graph) == (1.0,) * 5
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (0.125,) * 5
    assert tuple(graph.nodes[node]["dEPI"] for node in graph) == (0.0,) * 5
    alignment = 1 - (1 - push) * delta / math.pi
    expected_pressure = before.nodes[0]["delta_nfr"] * (
        1 - case["factors"]["UM_dnfr_reduction"] * alignment
    )
    assert graph.nodes[0]["delta_nfr"] == pytest.approx(expected_pressure, abs=1e-15)
    assert graph.nodes[0]["delta_nfr"] > 0
    assert all(
        graph.nodes[node]["delta_nfr"] == before.nodes[node]["delta_nfr"]
        for node in range(1, 5)
    )
    assert tuple(graph.nodes[0]["glyph_history"]) == ("AL", "UM")
    assert all(
        tuple(graph.nodes[node]["glyph_history"]) == ("AL",) for node in range(1, 5)
    )
    assert graph.graph["_t"] == before.graph["_t"] == 0.0
    assert graph.graph["RANDOM_SEED"] == 17
    assert case["result"].schedule == TWO_PHASE_JACOBI
    assert case["result"].glyph == "UM"
    assert case["result"].nodes_processed == 1
    assert graph.graph[STAGE_SCHEDULE_KEY]["nodes_processed"] == 1
    with pytest.raises(FrozenInstanceError):
        case["result"].nodes_processed = 2


def test_created_sector_does_not_claim_a_locked_or_stationary_state(executions):
    case = executions["default"]
    push = case["factors"]["UM_theta_push"]
    expected_source = tuple(push * value / 10 for value in (-3, 3, -1, 0, 1))
    actual = case["refreshed"]
    assert tuple(map(float, actual.phase_gradient)) == pytest.approx(
        expected_source, abs=2e-15
    )
    assert tuple(map(float, actual.snapshot.stored_pressure)) == pytest.approx(
        tuple(value / 2 for value in expected_source), abs=2e-15
    )
    assert all(value == 0 for value in actual.stored_pressure_residual)
    assert case["raw"].snapshot.stored_pressure[0] > 0
    assert actual.snapshot.stored_pressure[0] < 0
    assert any(case["raw"].stored_pressure_residual)
    lock = observe_phase_lock_source(case["graph"], coupling_strength=0.5)
    assert lock.full_u3_admission and lock.strict_acute_edges_estimate
    assert max(map(abs, lock.full_support_rate_residual)) > 0.01
    assert any(lock.capture.forcing)
    # The new edge has its actual compatibility weight, not unit conductance.
    # The resulting instantaneous source drives the new transport-weighted mean.
    balance = derive_forced_support_balance(
        actual.snapshot, epi_weight=actual.epi_weight, forcing=actual.forcing
    )
    weight = case["graph"].edges[0, 4]["weight"]
    assert weight < 1
    assert tuple(map(float, balance.strengths)) == (1 + weight, 2, 2, 2, 1 + weight)
    assert float(balance.compatibility_residual) == pytest.approx(
        push * (1 - case["score"]) / 10, abs=3e-15
    )
    assert balance.compatibility_residual > 0 and balance.mean_drift > 0
    assert not balance.has_zero_pressure_equilibrium


@pytest.mark.parametrize("name", ["si_only", "links_disabled"])
def test_existing_policy_controls_prevent_birth_without_changing_primary_writes(
    executions, name
):
    baseline, control = executions["default"], executions[name]
    assert control["admission"].allowed
    assert control["prediction"].edges == ()
    assert set(control["graph"].edges) == set(control["before"].edges)
    assert derive_phase_cycle_geometry(control["graph"]).cycle_rank == 0
    assert not certify_phase_winding(control["graph"], range(5)).cycle_exists
    for field in ("theta", "EPI", "nu_f", "delta_nfr", "dEPI", "glyph_history"):
        assert tuple(control["raw_graph"].nodes[node][field] for node in range(5)) == (
            tuple(baseline["raw_graph"].nodes[node][field] for node in range(5))
        )
    if name == "si_only":
        source = control["prediction"].target_proposals[0]
        assert tuple(candidate.target for candidate in source.link_candidates) == (4,)
        assert source.link_candidates[0].si_target == 0.0
        assert baseline["score"] - 0.2 < source.compatibility_threshold
    else:
        assert control["prediction"].target_proposals[0].link_candidates == ()
