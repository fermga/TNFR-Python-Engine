"""Regression tests for the scoped phase-coupled N-body adapter."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY
from tnfr.dynamics.nbody import (
    NBodySystem,
    gravitational_force,
    gravitational_potential,
)
from tnfr.dynamics.nbody_tnfr import (
    TNFRNBodySystem,
    compute_nbody_pair_forces,
    compute_nbody_pair_potential,
    compute_tnfr_coherence_potential,
    compute_tnfr_delta_nfr,
)
from tnfr.errors.contextual import NetworkConfigError


def _two_body_system(*, phases: tuple[float, float] = (0.0, 0.0)) -> TNFRNBodySystem:
    return TNFRNBodySystem(
        n_bodies=2,
        masses=[1.0, 2.0],
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        phases=np.asarray(phases, dtype=float),
        coupling_strength=0.2,
        coherence_strength=-1.5,
        distance_regularization=0.1,
    )


def test_default_sign_attracts_synchronized_and_repels_antiphase() -> None:
    synchronized = _two_body_system()
    synchronized_force = compute_nbody_pair_forces(
        synchronized.graph, synchronized.positions
    )
    assert synchronized_force[0, 0] > 0.0
    assert synchronized_force[1, 0] < 0.0

    antiphase = _two_body_system(phases=(0.0, np.pi))
    antiphase_force = compute_nbody_pair_forces(antiphase.graph, antiphase.positions)
    assert antiphase_force[0, 0] < 0.0
    assert antiphase_force[1, 0] > 0.0


def test_pair_force_is_negative_position_gradient_of_pair_potential() -> None:
    system = _two_body_system()
    step = 1e-6
    plus = system.positions.copy()
    minus = system.positions.copy()
    plus[0, 0] += step
    minus[0, 0] -= step

    derivative = (
        compute_nbody_pair_potential(system.graph, plus)
        - compute_nbody_pair_potential(system.graph, minus)
    ) / (2.0 * step)
    force = compute_nbody_pair_forces(system.graph, system.positions)

    assert force[0, 0] == pytest.approx(-derivative, rel=1e-8, abs=1e-10)
    assert np.sum(force, axis=0) == pytest.approx(np.zeros(3), abs=1e-14)


def test_pair_force_honors_graph_edges() -> None:
    system = _two_body_system()
    system.graph.remove_edge("body_0", "body_1")

    assert compute_nbody_pair_forces(
        system.graph, system.positions
    ) == pytest.approx(np.zeros((2, 3)))
    assert compute_nbody_pair_potential(system.graph, system.positions) == 0.0


def test_parallel_pair_couplings_add_independently_of_insertion_order() -> None:
    system = _two_body_system()
    parallel = nx.MultiGraph(system.graph)
    parallel.add_edge("body_0", "body_1", weight=0.3)
    combined = system.graph.copy()
    combined["body_0"]["body_1"]["weight"] = 0.5

    assert compute_nbody_pair_forces(
        parallel, system.positions
    ) == pytest.approx(compute_nbody_pair_forces(combined, system.positions))


def test_coincident_positions_have_finite_zero_direction_force() -> None:
    system = _two_body_system()
    coincident = np.zeros_like(system.positions)

    force = compute_nbody_pair_forces(system.graph, coincident)
    potential = compute_nbody_pair_potential(system.graph, coincident)

    assert np.all(np.isfinite(force))
    assert force == pytest.approx(np.zeros((2, 3)), abs=0.0)
    assert np.isfinite(potential)


def test_legacy_hamiltonian_readout_is_not_the_position_potential() -> None:
    system = _two_body_system()
    separated = system.positions.copy()
    separated[1, 0] = 3.0

    ground_initial = compute_tnfr_coherence_potential(
        system.graph, system.positions, system.hbar_str
    )
    ground_separated = compute_tnfr_coherence_potential(
        system.graph, separated, system.hbar_str
    )
    pair_initial = compute_nbody_pair_potential(system.graph, system.positions)
    pair_separated = compute_nbody_pair_potential(system.graph, separated)

    assert ground_initial == pytest.approx(ground_separated, abs=0.0)
    assert pair_initial != pytest.approx(pair_separated)


def test_projector_commutator_rate_is_zero_and_phases_stay_fixed() -> None:
    system = _two_body_system(phases=(0.25, 0.7))
    rates = compute_tnfr_delta_nfr(
        system.graph, ["body_0", "body_1"], system.hbar_str
    )
    initial_phases = system.phases.copy()

    system.step(0.01)

    assert rates == pytest.approx(np.zeros(2), abs=0.0)
    assert system.phases == pytest.approx(initial_phases, abs=0.0)


def test_pair_energy_and_adapter_state_are_consistent_after_evolution() -> None:
    system = _two_body_system()
    canonical_epi = {
        node: system.graph.nodes[node][EPI_PRIMARY] for node in system.graph
    }

    history = system.evolve(t_final=0.1, dt=0.001, store_interval=10)

    assert history["energy_drift"] < 1e-8
    assert history["hamiltonian_ground_state"][0] == pytest.approx(
        compute_tnfr_coherence_potential(
            system.graph, history["positions"][0], system.hbar_str
        )
    )
    assert np.linalg.norm(history["momentum"][-1] - history["momentum"][0]) < 1e-12
    for index, node in enumerate(system.graph):
        state = system.graph.nodes[node]["nbody_state"]
        assert state["position"] == pytest.approx(system.positions[index])
        assert state["velocity"] == pytest.approx(system.velocities[index])
        assert system.graph.nodes[node][EPI_PRIMARY] == canonical_epi[node]


def test_both_adapters_store_the_executed_final_step() -> None:
    adapter = _two_body_system()
    adapter_history = adapter.evolve(t_final=0.005, dt=0.001, store_interval=3)

    reference = NBodySystem(2, masses=[1.0, 2.0], G=1.0, softening=0.1)
    reference.set_state(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.zeros((2, 3)),
    )
    reference_history = reference.evolve(
        t_final=0.005,
        dt=0.001,
        store_interval=3,
    )

    assert adapter_history["time"] == pytest.approx([0.0, 0.003, 0.005])
    assert reference_history["time"] == pytest.approx([0.0, 0.003, 0.005])
    assert adapter_history["positions"][-1] == pytest.approx(adapter.positions)
    assert reference_history["positions"][-1] == pytest.approx(reference.positions)


def test_both_adapters_execute_a_short_step_to_reach_requested_time() -> None:
    adapter = TNFRNBodySystem(
        n_bodies=1,
        masses=[1.0],
        positions=np.zeros((1, 3)),
        velocities=np.array([[1.0, 0.0, 0.0]]),
    )
    reference = NBodySystem(1, masses=[1.0])
    reference.set_state(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]))

    adapter_history = adapter.evolve(t_final=0.105, dt=0.1)
    reference_history = reference.evolve(t_final=0.105, dt=0.1)

    assert adapter_history["time"][-1] == 0.105
    assert reference_history["time"][-1] == 0.105
    assert adapter.positions[0, 0] == pytest.approx(0.105)
    assert reference.positions[0, 0] == pytest.approx(0.105)


def test_newtonian_pair_law_preserves_representable_extreme_separation() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1e155, 0.0, 0.0]])
    masses = np.ones(2)

    potential = gravitational_potential(positions, masses)
    force = gravitational_force(positions, masses)

    assert potential == pytest.approx(-1e-155, rel=1e-15, abs=0.0)
    assert force[0, 0] == pytest.approx(1e-310, rel=1e-12, abs=0.0)
    assert force[1, 0] == pytest.approx(-1e-310, rel=1e-12, abs=0.0)
    assert np.all(np.isfinite(force))


def test_phase_pair_law_preserves_extreme_distance_tail() -> None:
    system = _two_body_system()
    positions = np.array([[0.0, 0.0, 0.0], [1e154, 0.0, 0.0]])
    amplitude = 1.5 * 0.2 * np.sqrt(1.0 * 0.5)

    potential = compute_nbody_pair_potential(system.graph, positions)
    force = compute_nbody_pair_forces(system.graph, positions)

    assert potential == pytest.approx(-amplitude / 1e154, rel=1e-14, abs=0.0)
    assert force[0, 0] == pytest.approx(amplitude / 1e308, rel=1e-12, abs=0.0)
    assert force[1] == pytest.approx(-force[0], abs=0.0)
    assert np.all(np.isfinite(force))


def test_newtonian_products_avoid_intermediate_overflow() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1e150, 0.0, 0.0]])
    masses = np.array([1e300, 1e300])

    potential = gravitational_potential(positions, masses, G=1e-300)
    force = gravitational_force(positions, masses, G=1e-300)

    assert potential == pytest.approx(-1e150, rel=1e-14)
    assert force[0, 0] == pytest.approx(1.0, rel=1e-14)
    assert force[1, 0] == pytest.approx(-1.0, rel=1e-14)


def test_both_adapters_scale_kinetic_products_before_squaring() -> None:
    velocity = np.array([[1e155, 0.0, 0.0]])
    expected = 5e306
    reference = NBodySystem(1, masses=[1e-3])
    reference.set_state(np.zeros((1, 3)), velocity)
    adapter = TNFRNBodySystem(
        n_bodies=1,
        masses=[1e-3],
        positions=np.zeros((1, 3)),
        velocities=velocity,
    )

    assert reference.compute_energy() == pytest.approx((expected, 0.0, expected))
    assert adapter.compute_energy() == pytest.approx((expected, 0.0, expected))


@pytest.mark.parametrize("n_bodies", [True, 1.5, np.bool_(True)])
def test_tnfr_adapter_rejects_noninteger_body_count(n_bodies) -> None:
    with pytest.raises(NetworkConfigError, match="n_bodies"):
        TNFRNBodySystem(
            n_bodies=n_bodies,
            masses=[1.0],
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
        )


def test_tnfr_adapter_normalizes_invalid_mass_shape_error() -> None:
    with pytest.raises(NetworkConfigError, match="masses"):
        TNFRNBodySystem(
            n_bodies=2,
            masses=np.ones((2, 1)),
            positions=np.zeros((2, 3)),
            velocities=np.zeros((2, 3)),
        )


@pytest.mark.parametrize(
    ("function", "positions", "masses", "message"),
    [
        (gravitational_force, np.zeros((2, 2)), [1.0, 1.0], "positions"),
        (
            gravitational_potential,
            np.zeros((2, 3)),
            [1.0, float("nan")],
            "masses",
        ),
    ],
)
def test_newtonian_helpers_reject_invalid_state(
    function: object,
    positions: np.ndarray,
    masses: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(positions, masses)  # type: ignore[operator]


def test_newtonian_helpers_reject_unsoftened_collision() -> None:
    positions = np.zeros((2, 3))
    masses = np.ones(2)

    with pytest.raises(ValueError, match="coincident positions"):
        gravitational_force(positions, masses)
    with pytest.raises(ValueError, match="coincident positions"):
        gravitational_potential(positions, masses)


@pytest.mark.parametrize("value", [True, np.bool_(True), 0.0, -0.1, float("nan")])
def test_newtonian_step_rejects_invalid_time_step(value: float) -> None:
    with pytest.raises(ValueError, match="dt"):
        NBodySystem(1, masses=[1.0]).step(value)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan")])
def test_distance_regularization_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(NetworkConfigError, match="distance_regularization"):
        TNFRNBodySystem(
            n_bodies=1,
            masses=[1.0],
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
            distance_regularization=value,
        )


@pytest.mark.parametrize("dt", [True, np.bool_(True), 0.0, -0.1, float("nan")])
def test_step_rejects_nonpositive_or_nonfinite_time_step(dt: float) -> None:
    with pytest.raises(NetworkConfigError, match="dt"):
        _two_body_system().step(dt)


def test_newtonian_failed_step_does_not_partially_commit_state() -> None:
    system = NBodySystem(2, masses=[1.0, 1.0])
    positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.875, 0.0, 0.0], [-0.875, 0.0, 0.0]])
    system.set_state(positions, velocities)

    with pytest.raises(ValueError, match="coincident positions"):
        system.step(1.0)

    assert system.positions == pytest.approx(positions, abs=0.0)
    assert system.velocities == pytest.approx(velocities, abs=0.0)
    assert system.time == 0.0


def test_evolution_rejects_unrepresentable_step_count_before_allocation() -> None:
    with pytest.raises(ValueError, match="too many time steps"):
        NBodySystem(1, masses=[1.0]).evolve(t_final=1e308, dt=1.0)
    with pytest.raises(NetworkConfigError, match="too many time steps"):
        TNFRNBodySystem(
            n_bodies=1,
            masses=[1.0],
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
        ).evolve(t_final=1e308, dt=1.0)
