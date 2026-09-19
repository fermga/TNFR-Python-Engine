"""P2 graph read-outs need not realize the auxiliary harmonic flow.

The exact image/tangency proof is in TNFR_VARIATIONAL_PRINCIPLE.md section 3.7.
These finite regressions exercise production readers, pressure refresh and the
shared nodal integrator. They neither fit a phase law nor certify all graphs.
"""

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.physics.symplectic_substrate import (
    evolve_substrate_flow,
    extract_phase_space_point,
    hamiltonian_vector_field,
)


def _p2(*, phase=math.pi / 3, epi=(0.0, 1.0), capacity=(1.0, 1.0)):
    graph = nx.path_graph(2)
    graph.edges[0, 1].update(weight=1.0, length=1.0)
    graph.graph.update(
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        EPI_MIN=-2.0,
        EPI_MAX=2.0,
        CLIP_MODE="hard",
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=float(epi[node]),
            nu_f=float(capacity[node]),
            theta=0.0 if node == 0 else phase % math.tau,
        )
    default_compute_delta_nfr(graph)
    return graph


def _read(graph, aliases):
    return np.array([get_attr(graph.nodes[n], aliases, None) for n in graph])


def test_refreshed_epi_diffusion_is_not_the_extracted_harmonic_flow():
    graph = _p2()
    initial = extract_phase_space_point(graph)
    phases = _read(graph, ALIAS_THETA).copy()
    np.testing.assert_allclose(initial.k_phi, [-math.pi / 3, math.pi / 3])
    np.testing.assert_allclose(initial.j_phi, [math.sqrt(3) / 2, -math.sqrt(3) / 2])
    np.testing.assert_allclose(_read(graph, ALIAS_DNFR), [1.0, -1.0])
    harmonic_velocity = hamiltonian_vector_field(initial).reshape(2, 4)
    np.testing.assert_allclose(harmonic_velocity[:, 0], initial.j_phi)
    np.testing.assert_allclose(harmonic_velocity[:, 1], -initial.k_phi)

    # A held-pressure Euler segment, refreshed explicitly at every boundary.
    # Binary-exact dt is a numerical fixture, not an added physical parameter.
    dt = 1.0 / 8.0
    for step in range(3):
        before = _read(graph, ALIAS_EPI)
        pressure = _read(graph, ALIAS_DNFR)
        update_epi_via_nodal_equation(graph, dt=dt, t=step * dt, method="euler")
        np.testing.assert_allclose(_read(graph, ALIAS_EPI), before + dt * pressure)
        np.testing.assert_array_equal(_read(graph, ALIAS_THETA), phases)
        default_compute_delta_nfr(graph)
        point = extract_phase_space_point(graph)
        np.testing.assert_array_equal(point.k_phi, initial.k_phi)
        np.testing.assert_array_equal(point.j_phi, initial.j_phi)
        difference = (1 - 2 * dt) ** (step + 1)
        np.testing.assert_allclose(_read(graph, ALIAS_DNFR), [difference, -difference])
        np.testing.assert_allclose(point.phi_s, [-difference, difference])
        np.testing.assert_allclose(point.j_dnfr, 2 * point.phi_s)

    harmonic = evolve_substrate_flow(initial, 3 * dt)
    assert not np.allclose(harmonic.k_phi, point.k_phi)
    assert not np.allclose(harmonic.j_phi, point.j_phi)
    assert not np.allclose(harmonic.j_phi + np.sin(harmonic.k_phi), 0.0)


@pytest.mark.parametrize(
    "phase", [-2.5, -math.pi / 2, -math.pi / 3, math.pi / 3, math.pi / 2, 2.5]
)
def test_harmonic_velocity_is_not_tangent_to_nonzero_p2_geometric_image(phase):
    # Read-only snapshots do not request a U3-gated coupling operation.
    point = extract_phase_space_point(_p2(phase=phase))
    k, j = point.k_phi, point.j_phi
    np.testing.assert_allclose(j + np.sin(k), 0.0, atol=1e-14)
    velocity = hamiltonian_vector_field(point).reshape(2, 4)
    constraint_rate = velocity[:, 1] + np.cos(k) * velocity[:, 0]
    np.testing.assert_allclose(constraint_rate, -k - np.sin(k) * np.cos(k))
    assert np.all(np.abs(constraint_rate) > 0.1)
    assert np.all(constraint_rate * k < 0.0)


def test_synchronized_pressure_disagreement_still_obstructs_potential_flow():
    point = extract_phase_space_point(_p2(phase=0.0))
    np.testing.assert_allclose(point.k_phi, 0.0, atol=1e-14)
    np.testing.assert_allclose(point.j_phi, 0.0, atol=1e-14)
    np.testing.assert_allclose(point.phi_s, [-1.0, 1.0])
    np.testing.assert_allclose(point.j_dnfr, [-2.0, 2.0])
    velocity = hamiltonian_vector_field(point).reshape(2, 4)
    constraint_rate = velocity[:, 3] - 2 * velocity[:, 2]
    np.testing.assert_allclose(constraint_rate, -5 * point.phi_s)


@pytest.mark.parametrize("capacity", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
def test_derived_potential_decay_rate_is_the_sum_of_nodal_capacities(capacity):
    graph = _p2(capacity=capacity)
    before = extract_phase_space_point(graph)
    dt = 1.0 / 8.0
    update_epi_via_nodal_equation(graph, dt=dt, t=0.0, method="euler")
    default_compute_delta_nfr(graph)
    after = extract_phase_space_point(graph)
    rate = sum(capacity)
    np.testing.assert_allclose((after.phi_s - before.phi_s) / dt, -rate * before.phi_s)
    np.testing.assert_allclose(
        (after.j_dnfr - before.j_dnfr) / dt, -rate * before.j_dnfr
    )
    np.testing.assert_array_equal(after.k_phi, before.k_phi)
    np.testing.assert_array_equal(after.j_phi, before.j_phi)


def test_zero_pressure_synchronized_control_is_a_common_fixed_point():
    graph = _p2(phase=0.0, epi=(0.5, 0.5))
    point = extract_phase_space_point(graph)
    np.testing.assert_allclose(point.to_vector(), 0.0, atol=1e-14)
    np.testing.assert_allclose(hamiltonian_vector_field(point), 0.0, atol=1e-14)
    update_epi_via_nodal_equation(graph, dt=0.125, t=0.0, method="euler")
    default_compute_delta_nfr(graph)
    np.testing.assert_allclose(_read(graph, ALIAS_EPI), [0.5, 0.5])
    np.testing.assert_allclose(extract_phase_space_point(graph).to_vector(), 0.0)


def test_isolated_harmonic_endpoint_can_return_to_geometric_image():
    point = extract_phase_space_point(_p2())
    half_turn = evolve_substrate_flow(point, math.pi)
    opposite = extract_phase_space_point(_p2(phase=-math.pi / 3))
    np.testing.assert_allclose(half_turn.k_phi, opposite.k_phi, atol=1e-14)
    np.testing.assert_allclose(half_turn.j_phi, opposite.j_phi, atol=1e-14)
    quarter_turn = evolve_substrate_flow(point, math.pi / 2)
    assert not np.allclose(quarter_turn.j_phi + np.sin(quarter_turn.k_phi), 0.0)
    # Endpoint coincidence concerns this sector, not an executed graph event.
