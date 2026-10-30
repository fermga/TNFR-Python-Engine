"""Executed controls for the conditional counted phase/form quotient on K3.

The fixed triangle, common capacity, channel recipe and phase law are supplied.
Reduced pressure is computed from inherited geometry and its own state before
comparison; fine endpoints never fit a source or replace a reduced endpoint.
Exact endpoint budgets retain projection, pressure and Euler materialization.
These finite controls do not prove an autonomous clock or future runtime closure.
"""

import math
from copy import deepcopy
from fractions import Fraction as F

import networkx as nx
import pytest

from tests.joint_phase_helpers import configure as _configure
from tests.joint_phase_helpers import project as _project
from tests.joint_phase_helpers import triangle as _triangle
from tnfr.alias import set_theta
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import DefaultIntegrator
from tnfr.dynamics.phase_evolution import (
    _propose_u3_phase_from_neighbors,
    propose_u3_gated_phase_step,
)
from tnfr.physics.joint_quotient import (
    _counted_phase_gradient,
    observe_joint_nodal_quotient,
    propose_joint_nodal_phase_step,
)
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_euler,
)
from tnfr.utils import angle_diff

BLOCKS = ((0,), (1, 2))
H = F(1, 8)
K = F(1, 4)


def _reduced_pressure(joint, epi, phase):
    """Refresh the reduced source from its own state and inherited counts."""
    structure = joint.structure
    weights = dict(joint.fine_capture.normalized_weights)
    phase_gradient = _counted_phase_gradient(phase, structure.multiplicity)
    return tuple(
        weights["epi"]
        * sum((weight * (epi[j] - epi[i]) for j, weight in enumerate(row)), F(0))
        / structure.macro_strengths[i]
        + structure.source_scale[i]
        * (
            weights["phase"] * phase_gradient[i]
            + weights["vf"] * structure.capacity_gradient[i]
            + weights["topo"] * structure.topology_gradient[i]
        )
        for i, row in enumerate(structure.macro_conductance)
    )


def _execute_comparison(epi):
    fine = _triangle(epi=epi)
    proposal = propose_joint_nodal_phase_step(fine, BLOCKS, dt=H, coupling_strength=K)
    joint = proposal.joint
    coarse = nx.path_graph(2)
    _configure(coarse)
    coarse.edges[0, 1].update(weight=2.0, length=1.0)
    for i in coarse:
        coarse.nodes[i].update(
            EPI=float(joint.closure.projected_epi[i]),
            theta=float(joint.block_phase[i]),
            nu_f=float(joint.structure.effective_capacity[i]),
            delta_nfr=0.0,
            dEPI=0.0,
        )
    coarse_initial = observe_support_transport(coarse)
    pressure = _reduced_pressure(joint, coarse_initial.epi, joint.block_phase)
    for i, value in enumerate(pressure):
        coarse.nodes[i]["delta_nfr"] = float(value)
    before = tuple(observe_support_transport(graph) for graph in (fine, coarse))
    for graph in (fine, coarse):
        DefaultIntegrator().integrate(
            graph, dt=float(H), t=0.0, method="euler", n_jobs=1
        )
    after = tuple(observe_support_transport(graph) for graph in (fine, coarse))
    budgets = tuple(
        observe_support_transport_euler(a, b, H)
        for a, b in zip(before, after, strict=True)
    )
    for graph, phases in (
        (fine, proposal.fine_phase_after),
        (coarse, proposal.block_phase_after),
    ):
        for i, theta in enumerate(phases):
            set_theta(graph, i, float(theta))
    default_compute_delta_nfr(fine)
    endpoint_pressure = _reduced_pressure(
        joint, after[1].epi, proposal.block_phase_after
    )
    for i, value in enumerate(endpoint_pressure):
        coarse.nodes[i]["delta_nfr"] = float(value)
    endpoint_joint = observe_joint_nodal_quotient(fine, BLOCKS)
    endpoint = tuple(observe_support_transport(graph) for graph in (fine, coarse))
    return proposal, before, after, budgets, endpoint_pressure, endpoint_joint, endpoint


@pytest.fixture(
    scope="module",
    params=(
        (0.25, -0.125, 0.375),
        (0.25, 0.125, math.nextafter(0.125, math.inf)),
    ),
    ids=("dyadic-hidden-form", "nonrepresentable-projection"),
)
def executed(request):
    return _execute_comparison(request.param)


def test_shared_integrators_satisfy_exact_joint_endpoint_budget(executed):
    proposal, before, after, budgets, *_ = executed
    joint = proposal.joint
    projected_initial, projected_after = _project(before[0].epi), _project(after[0].epi)
    projected_defect = _project(budgets[0].state_defect)
    initial_error = tuple(x - y for x, y in zip(projected_initial, before[1].epi))
    rate_error = tuple(
        x - y for x, y in zip(joint.projected_fresh_rate, before[1].rate)
    )
    assert tuple(x - y for x, y in zip(projected_after, after[1].epi)) == tuple(
        initial_error[i]
        + H * rate_error[i]
        + projected_defect[i]
        - budgets[1].state_defect[i]
        for i in range(2)
    )
    assert rate_error == tuple(
        joint.phase_materialization_rate_defect[i]
        + joint.projected_kernel_rate_defect[i]
        + joint.effective_rate[i]
        - before[1].rate[i]
        for i in range(2)
    )
    assert all(budget.identity_residual == 0 for budget in budgets)
    assert all(abs(x) < F(1, 2) for snapshot in after for x in snapshot.epi)
    assert joint.fine_capture.stored_pressure_residual == (0, 0, 0)
    assert joint.closure.hidden_rate_contribution == (0, 0)
    if before[0].epi[2] == F(math.nextafter(0.125, math.inf)):
        assert initial_error[1] != 0
    else:
        assert initial_error == (0, 0)
        assert any(joint.closure.hidden_epi)


def test_endpoint_pressure_is_refreshed_from_each_own_phase_and_form(executed):
    proposal, before, after, _, pressure, endpoint_joint, endpoint = executed
    assert (
        endpoint[0].stored_pressure == endpoint_joint.fine_capture.full_kernel_pressure
    )
    assert endpoint_joint.fine_capture.stored_pressure_residual == (0, 0, 0)
    assert endpoint[1].stored_pressure == tuple(F(float(p)) for p in pressure)
    assert endpoint[0].stored_pressure != before[0].stored_pressure
    assert endpoint[1].stored_pressure != before[1].stored_pressure
    assert endpoint[0].epi == after[0].epi
    assert endpoint[1].epi == after[1].epi
    assert endpoint_joint.block_phase == proposal.block_phase_after
    # The refresh reads the independently evolved reduced form, not P*x_after.
    assert pressure == _reduced_pressure(
        proposal.joint, after[1].epi, proposal.block_phase_after
    )
    delta = angle_diff(*reversed(tuple(map(float, proposal.block_phase_after))))
    contrast = float(after[1].epi[0] - after[1].epi[1])
    analytic = -contrast / 2 + delta / (2 * math.pi)
    assert tuple(map(float, pressure)) == pytest.approx(
        (analytic, -analytic), abs=3e-16
    )


def test_executed_contrast_follows_the_prospective_k3_row(executed):
    proposal, before, after, *_ = executed
    q = float(before[1].epi[0] - before[1].epi[1])
    # e=w=1/2, kappa=1 and initial delta=1/2 are declared before evaluation.
    expected = q + float(H) * (-3 * q / 4 + 3 / (8 * math.pi))
    actual = float(after[1].epi[0] - after[1].epi[1])
    assert actual == pytest.approx(expected, abs=5e-16)
    assert proposal.joint.structure.source_scale == (1, 2)


def test_counted_phase_preserves_internal_denominator_and_fine_proposals():
    graph = _triangle()
    result = propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)
    assert result.counted_neighbors == ((1, 1), (0, 1))
    assert result.joint.structure.effective_capacity == (1, F(1, 2))
    assert result.joint.structure.block_capacity == (1, 1)
    a, b = map(float, result.joint.block_phase)
    expected = (
        a + float(H) * (1 + float(K) * math.sin(b - a)),
        b + float(H) * (1 - float(K) * math.sin(b - a) / 2),
    )
    assert tuple(map(float, result.block_phase_after)) == pytest.approx(
        expected, abs=2e-16
    )
    assert result.phase_step_defect == (0, 0, 0)
    assert result.fine_phase_after == (
        result.block_phase_after[0],
        result.block_phase_after[1],
        result.block_phase_after[1],
    )
    assert "primitive_phase" not in result.held_parameter_rows
    assert "fine_capacity" in result.held_parameter_rows
    assert abs(angle_diff(*map(float, result.block_phase_after))) < b - a
    naive = _propose_u3_phase_from_neighbors(
        graph.graph,
        (0, 1),
        ((1,), (0,)).__getitem__,
        (a, b),
        (1, 1),
        dt=float(H),
        coupling_strength=float(K),
    )
    assert abs(float(naive[1]) - float(result.block_phase_after[1])) > 0.001


def test_zero_phase_contrast_is_not_created_by_geometric_mobility():
    graph = _triangle(epi=(0, 0, 0), phase=(0.25, 0.25))
    result = propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)
    assert result.joint.effective_pressure == (0, 0)
    assert result.block_phase_after == (F(3, 8), F(3, 8))
    wrong = _propose_u3_phase_from_neighbors(
        graph.graph,
        (0, 1),
        result.counted_neighbors.__getitem__,
        result.joint.block_phase,
        result.joint.structure.effective_capacity,
        dt=float(H),
        coupling_strength=float(K),
    )
    assert F(float(wrong[1])) - F(float(wrong[0])) == -H / 2
    assert tuple(F(float(x)) for x in wrong) != result.block_phase_after


def test_phase_off_control_preserves_transport_coefficients_and_phase_evolution():
    active = _triangle(epi=(0, 0, 0))
    control = _triangle(
        epi=(0, 0, 0),
        weights={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 1.0},
    )
    proposals = tuple(
        propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)
        for graph in (active, control)
    )
    active_weights, control_weights = (
        dict(proposal.joint.fine_capture.normalized_weights) for proposal in proposals
    )
    assert active_weights["epi"] == control_weights["epi"] == F(1, 2)
    assert active_weights["vf"] == control_weights["vf"] == 0
    assert proposals[0].fine_phase_after == proposals[1].fine_phase_after
    assert proposals[0].block_phase_after == proposals[1].block_phase_after
    assert proposals[1].joint.structure.topology_gradient == (0, 0)
    assert proposals[1].joint.effective_pressure == (0, 0)
    assert any(proposals[0].joint.effective_pressure)
    for graph in (active, control):
        DefaultIntegrator().integrate(
            graph, dt=float(H), t=0.0, method="euler", n_jobs=1
        )
    assert observe_support_transport(control).epi == (0, 0, 0)
    assert observe_support_transport(active).epi[0] == pytest.approx(
        float(H) / (4 * math.pi), abs=2e-17
    )


def test_observer_leaves_live_state_and_support_unchanged():
    graph = _triangle()
    node_state, edges = deepcopy(dict(graph.nodes(data=True))), tuple(
        graph.edges(data="weight")
    )
    result = propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)
    assert dict(graph.nodes(data=True)) == node_state
    assert tuple(graph.edges(data="weight")) == edges
    assert result.joint.fine_capture.snapshot.epi == tuple(
        F(node_state[i]["EPI"]) for i in graph
    )


def test_raw_phase_api_still_accepts_finite_noncanonical_angles():
    graph = nx.path_graph(2)
    phases = (-0.25, -0.25)
    actual = propose_u3_gated_phase_step(
        graph, (0, 1), phases, (1, 1), dt=0.125, coupling_strength=0.25
    )
    assert tuple(actual) == ((-0.125) % math.tau,) * 2


def test_joint_local_chart_survives_canonical_coordinate_wrap():
    graph = _triangle(phase=(math.tau - 0.0625, 0.0625))
    result = propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)
    assert result.block_phase_after[0] < F(1, 4)
    assert result.phase_step_defect == (0, 0, 0)
    assert abs(angle_diff(*map(float, result.block_phase_after))) < 0.125


@pytest.mark.parametrize("phase", ((-0.25, 0.25), (0, math.tau)))
def test_joint_chart_rejects_noncanonical_phases(phase):
    with pytest.raises(ValueError, match="canonical"):
        propose_joint_nodal_phase_step(
            _triangle(phase=phase), BLOCKS, dt=H, coupling_strength=K
        )


@pytest.mark.parametrize("gap", (math.pi / 2, 2.0, math.pi))
def test_joint_chart_rejects_boundary_or_undefined_resultants(gap):
    with pytest.raises(ValueError, match="strict half-pi"):
        propose_joint_nodal_phase_step(
            _triangle(phase=(0, gap)), BLOCKS, dt=H, coupling_strength=K
        )


def test_joint_proposal_requires_full_configured_u3_admission():
    graph = _triangle()
    graph.graph["UM_MAX_PHASE_DIFF"] = 0.25
    with pytest.raises(ValueError, match="all fine support U3-admitted"):
        propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)


def test_joint_proposal_rejects_endpoint_outside_its_chart():
    with pytest.raises(ValueError, match="strict half-pi"):
        propose_joint_nodal_phase_step(_triangle(), BLOCKS, dt=4, coupling_strength=1)


@pytest.mark.parametrize(
    "dt", (0, -1, True, float("nan"), float("inf"), 1j, F(1, 2**1200))
)
def test_joint_proposal_rejects_invalid_time(dt):
    with pytest.raises((TypeError, ValueError)):
        propose_joint_nodal_phase_step(_triangle(), BLOCKS, dt=dt, coupling_strength=K)


@pytest.mark.parametrize(
    "strength", (-1, True, float("nan"), float("inf"), 1j, F(1, 2**1200))
)
def test_joint_proposal_rejects_invalid_coupling(strength):
    with pytest.raises((TypeError, ValueError)):
        propose_joint_nodal_phase_step(
            _triangle(), BLOCKS, dt=H, coupling_strength=strength
        )


def test_zero_coupling_retains_neutral_common_clock_control():
    graph = _triangle()
    result = propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=0)
    assert result.block_phase_after == (F(3, 8), F(7, 8))
    assert result.phase_step_defect == (0, 0, 0)


def test_joint_proposal_rejects_nonconstant_fine_block_phase():
    graph = _triangle()
    set_theta(graph, 2, 0.875)
    with pytest.raises(ValueError, match="block-constant phase"):
        propose_joint_nodal_phase_step(graph, BLOCKS, dt=H, coupling_strength=K)
