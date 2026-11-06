"""Detached Mutation proposal geometry, not an admitted or selected event.

Mutation changes phase while retaining EPI and stored pressure. Its proposed
fresh source can change a subsequent modeled rate without any instantaneous
EPI-energy jump. Default selection, temporal growth evidence and U4b remain
separate obligations; no event history, selector override or trajectory is
constructed here.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import NODES, _graph, _unit_regional_budget
from tests.physics.test_full_prism_phase_feedback_scope import _full_phase_geometry
from tnfr.operators._mutation_stage_kernel import propose_mutation_stage
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.forcing_realization import capture_non_epi_forcing


def test_nonlinear_source_response_gives_a_signed_reset_work_condition():
    s = pytest.importorskip("sympy")
    source, reference, _ = _full_phase_geometry()
    response = s.Matrix(reference.mean_response)
    score = s.Matrix(s.symbols("score0:6", real=True))
    target = 0
    direction = s.eye(6)[:, target]
    derivative = (response - s.eye(6)) * direction / s.pi
    neighbors = tuple(
        i for i, row in enumerate(source.support_neighbors) if target in row
    )
    assert len(neighbors) == 3 and target not in neighbors
    assert derivative[target] == -1 / s.pi
    assert all(
        derivative[i] == 0 for i in range(6) if i not in neighbors and i != target
    )
    expected = (
        sum(score[i] * response[i, target] for i in neighbors) - score[target]
    ) / s.pi
    assert s.expand(score.dot(derivative) - expected) == 0
    assert (response - s.eye(6)) * s.ones(6, 1) == s.zeros(6, 1)

    phasors = s.Matrix(s.symbols("phasor0:6"))
    multiplier = s.Symbol("phase_multiplier")
    changed = phasors.copy()
    changed[target] *= multiplier
    for row in source.support_neighbors:
        before = sum(phasors[i] for i in row)
        after = sum(changed[i] for i in row)
        expected_change = (multiplier - 1) * phasors[target] if target in row else 0
        assert s.expand(after - before - expected_change) == 0
    # For multiplier=exp(i*s), this is the EXACT finite resultant update.
    # On a regular path theta+t*s*e_target, integrating the shared Jacobian
    # gives score^T Delta_g=(s/pi)*(sum score_i*average(R_i,target)-score_target).
    # For S=x^T Q*x choose score=Q*x and multiply by 2*w; for Dirichlet
    # energy choose score=B*x and multiply by w. These are different rates.
    # The formula requires a nonzero resultant and fixed wrap branch along
    # the path. Endpoint-only branch changes must not be omitted.


def test_default_detached_proposal_changes_fresh_rates_without_an_epi_energy_jump():
    graph = _graph()  # Existing preparation; no search or tuning after scoring.
    graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=0.25, vf=0.25, topo=0)
    initial = capture_non_epi_forcing(graph)
    for node, pressure in zip(NODES, initial.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(pressure)
    before = capture_non_epi_forcing(graph)
    target_index = max(range(6), key=before.full_kernel_pressure.__getitem__)
    target = NODES[target_index]
    assert before.full_kernel_pressure[target_index] > 0
    saved_nodes = tuple(dict(graph.nodes[node]) for node in NODES)
    proposal = propose_mutation_stage(
        graph.nodes[target]["theta"], graph.nodes[target]["delta_nfr"]
    )
    assert proposal.theta_shift == 0.25 and not proposal.fixed_mode
    assert tuple(dict(graph.nodes[node]) for node in NODES) == saved_nodes
    # This copied endpoint is the proposal's phase-only image, NOT a call
    # to Mutation or an admission, selection, event, or causal certificate.
    proposed_graph = deepcopy(graph)
    proposed_graph.nodes[target]["theta"] = proposal.theta_after
    after = capture_non_epi_forcing(proposed_graph)
    assert after.snapshot.epi == before.snapshot.epi
    assert after.snapshot.capacity == before.snapshot.capacity
    assert after.snapshot.stored_pressure == before.snapshot.stored_pressure
    assert after.snapshot.dirichlet_energy == before.snapshot.dirichlet_energy
    assert after.snapshot.rate == before.snapshot.rate
    assert after.snapshot.energy_rate == before.snapshot.energy_rate

    shift = proposal.theta_shift
    neighbor_angle = math.atan2(math.sin(shift), 2 + math.cos(shift))
    expected_phase_change = tuple(
        (
            -shift / math.pi
            if i == target_index
            else neighbor_angle / math.pi if target_index in row else 0.0
        )
        for i, row in enumerate(before.snapshot.support_neighbors)
    )
    assert tuple(map(float, after.phase_gradient)) == pytest.approx(
        expected_phase_change, rel=0, abs=1e-15
    )
    assert before.phase_gradient == (0,) * 6
    force_change = tuple(
        a - b for a, b in zip(after.forcing, before.forcing, strict=True)
    )
    gradient = before.snapshot.dirichlet_gradient
    modeled_dirichlet_rate_change = dot(gradient, force_change)
    fresh_rate_change = dot(
        gradient,
        tuple(
            a - b
            for a, b in zip(
                after.full_kernel_pressure, before.full_kernel_pressure, strict=True
            )
        ),
    )
    rounding_change = dot(
        gradient,
        tuple(
            a - b
            for a, b in zip(
                after.kernel_pressure_defect, before.kernel_pressure_defect, strict=True
            )
        ),
    )
    assert fresh_rate_change == modeled_dirichlet_rate_change + rounding_change
    assert modeled_dirichlet_rate_change > 0 and fresh_rate_change > 0
    weight = dict(before.normalized_weights)["phase"]
    predicted_dirichlet = (
        float(weight)
        / math.pi
        * (
            -shift * float(gradient[target_index])
            + neighbor_angle
            * sum(
                float(gradient[i])
                for i in range(6)
                if target_index in before.snapshot.support_neighbors[i]
            )
        )
    )
    assert float(modeled_dirichlet_rate_change) == pytest.approx(
        predicted_dirichlet, rel=0, abs=1e-15
    )

    norm0, rate0, work0 = _unit_regional_budget(
        before.snapshot, before.forcing, e=before.epi_weight
    )
    norm1, rate1, work1 = _unit_regional_budget(
        after.snapshot, after.forcing, e=after.epi_weight
    )
    assert norm1 == norm0
    assert rate1 - rate0 == work1 - work0
    assert work1 - work0 > 0
    centered = tuple(
        value - sum(before.snapshot.epi[3 * (i // 3) : 3 * (i // 3) + 3], Q(0)) / 3
        for i, value in enumerate(before.snapshot.epi)
    )
    assert work1 - work0 == 2 * dot(centered, force_change)
    # This fixture satisfies B*x=3*x_centered, so the two rate increments
    # differ by 3/2; neither should be reported as an event energy jump.
    assert modeled_dirichlet_rate_change == Q(3, 2) * (work1 - work0)
    assert rate1 < 0  # A positive increment is still insufficient to hold this shape.


def test_consensus_pressure_maximum_has_a_positive_dirichlet_rate_increment_only():
    s = pytest.importorskip("sympy")
    angle = s.Symbol("angle", real=True)
    alpha = s.atan(s.sin(angle) / (2 + s.cos(angle)))
    derivative = s.diff(alpha, angle)
    assert s.trigsimp(derivative - (1 + 2 * s.cos(angle)) / (5 + 4 * s.cos(angle))) == 0
    assert (
        s.trigsimp(
            derivative
            - s.Rational(1, 3)
            - 2 * (s.cos(angle) - 1) / (3 * (5 + 4 * s.cos(angle)))
        )
        == 0
    )
    assert alpha.subs(angle, 0) == 0
    # For 0<angle<=1/4<pi/2, alpha'>0 and alpha'<1/3. Integration gives
    # 0<alpha(angle)<angle/3, with no midpoint or small-angle replacement.
    assert 0 < s.Rational(1, 4) < s.pi / 2
    radius, shift, neighbor_angle, excess, weight = s.symbols(
        "radius shift neighbor_angle excess weight", positive=True
    )
    target_gradient = -radius
    neighbor_sum = 3 * target_gradient + excess
    increment = (
        weight * (-shift * target_gradient + neighbor_angle * neighbor_sum) / s.pi
    )
    decomposition = (
        weight
        * (radius * (shift - 3 * neighbor_angle) + neighbor_angle * excess)
        / s.pi
    )
    assert s.expand(increment - decomposition) == 0
    # At consensus with pure fresh EPI pressure p=-e*(B*x)/3, a positive
    # pressure maximum is a negative Dirichlet-gradient minimum, so the
    # neighbor excess is >=0. Thus a positive default reset gives a strictly
    # positive phase contribution to the next fresh Dirichlet rate. This
    # necessary accounting result says nothing about default selection or
    # the signed-growth/U4b gates, and does not prove sufficient maintenance.
