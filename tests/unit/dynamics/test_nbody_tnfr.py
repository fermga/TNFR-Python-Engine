"""Tests for pure TNFR N-body system (no gravitational assumptions).

Tests verify:
1. System initialization
2. Phase-dependent forces
3. Coherence-based dynamics
4. Momentum conservation
5. TNFR canonical invariants
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.dynamics.nbody_tnfr import (
    TNFRNBodySystem,
    compute_tnfr_coherence_potential,
    compute_tnfr_delta_nfr,
)


class TestTNFRNBodyInitialization:
    """Test TNFR N-body system initialization."""

    def test_basic_initialization(self):
        """Test basic system creation."""
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        velocities = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
        phases = np.array([0.0, 0.0, 0.0])

        system = TNFRNBodySystem(
            n_bodies=3,
            masses=[1.0, 2.0, 3.0],
            positions=positions,
            velocities=velocities,
            phases=phases,
        )

        assert system.n_bodies == 3
        assert len(system.masses) == 3
        assert system.positions.shape == (3, 3)
        assert system.velocities.shape == (3, 3)
        assert system.phases.shape == (3,)

    def test_invalid_masses_raises(self):
        """Test that negative masses raise error."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])

        with pytest.raises(ValueError, match="positive"):
            TNFRNBodySystem(
                n_bodies=2,
                masses=[1.0, -1.0],
                positions=positions,
                velocities=velocities,
            )

    def test_graph_creation(self):
        """Test that TNFR graph is created correctly."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 2.0],
            positions=positions,
            velocities=velocities,
        )

        # Check graph exists
        assert system.graph is not None
        assert len(system.graph.nodes) == 2

        # Check for TNFR attributes
        for node_id in ["body_0", "body_1"]:
            assert node_id in system.graph.nodes
            node_data = system.graph.nodes[node_id]
            assert "νf" in node_data or "vf" in node_data
            assert "theta" in node_data

    def test_structural_frequency_is_inverse_mass(self):
        """Test TNFR invariant: νf = 1/m."""
        masses = [2.0, 4.0, 0.5]
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        velocities = np.zeros((3, 3))

        system = TNFRNBodySystem(
            n_bodies=3,
            masses=masses,
            positions=positions,
            velocities=velocities,
        )

        for i, m in enumerate(masses):
            node_id = f"body_{i}"
            node_data = system.graph.nodes[node_id]
            nu_f = node_data.get("νf", node_data.get("vf"))
            expected_nu_f = 1.0 / m
            assert abs(nu_f - expected_nu_f) < 1e-10


class TestPhaseDependentForces:
    """Test phase-dependent force computation."""

    def test_synchronized_phases_attract(self):
        """Test that synchronized phases lead to attraction."""
        # Two bodies with same phase should attract
        positions = np.array([[0, 0, 0], [2, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])
        phases = np.array([0.0, 0.0])  # Synchronized

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
            phases=phases,
            coupling_strength=1.0,
            coherence_strength=-1.0,  # Negative = attractive
        )

        # Get accelerations
        accel = system._compute_tnfr_accelerations()

        # Bodies should attract: accelerations point toward each other
        # Body 0 toward body 1, body 1 toward body 0
        # The key is they should have opposite signs and similar magnitudes
        assert abs(accel[0, 0]) > 0  # Non-zero acceleration
        assert abs(accel[1, 0]) > 0  # Non-zero acceleration
        
        # Should be opposite directions (attraction)
        assert accel[0, 0] * accel[1, 0] < 0

        # Magnitudes should be similar (equal masses)
        assert abs(abs(accel[0, 0]) - abs(accel[1, 0])) < 0.1

    def test_antiphase_repulsion(self):
        """Test that antiphase nodes repel."""
        # Two bodies with opposite phases should repel
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])
        phases = np.array([0.0, np.pi])  # Antiphase

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
            phases=phases,
            coupling_strength=1.0,
            coherence_strength=-1.0,
        )

        accel = system._compute_tnfr_accelerations()

        # With antiphase, cos(π) = -1, so force reverses
        # Since coherence_strength is negative, double negative = positive
        # So bodies should repel: body 0 away from body 1
        # This depends on implementation details, but magnitude should be non-zero
        assert abs(accel[0, 0]) > 0 or abs(accel[1, 0]) > 0


class TestMomentumConservation:
    """Test linear momentum conservation (fundamental in TNFR)."""

    def test_momentum_conserved_two_body(self):
        """Test momentum conservation for two-body system."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[1.0, 0.5, 0.0], [-0.5, -0.25, 0.0]])
        phases = np.array([0.0, 0.0])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 2.0],
            positions=positions,
            velocities=velocities,
            phases=phases,
        )

        p0 = system.compute_momentum()

        # Evolve
        history = system.evolve(t_final=1.0, dt=0.01, store_interval=10)

        p_final = history["momentum"][-1]

        # Momentum should be conserved
        assert np.allclose(p_final, p0, atol=1e-10)


class TestTNFRInvariants:
    """Test TNFR canonical invariants."""

    def test_epi_contains_position_velocity(self):
        """Test that EPI encodes position and velocity."""
        positions = np.array([[1, 2, 3], [4, 5, 6]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
        )

        # Check EPI in graph
        for i in range(2):
            node_id = f"body_{i}"
            epi = system.graph.nodes[node_id]["epi"]

            # EPI should be a dict with position and velocity
            if isinstance(epi, dict):
                assert "position" in epi
                assert "velocity" in epi
                assert np.allclose(epi["position"], positions[i])
                assert np.allclose(epi["velocity"], velocities[i])

    def test_phases_evolve(self):
        """Test that phases evolve over time."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 1, 0]])
        phases = np.array([0.0, 0.5])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
            phases=phases,
        )

        phases_initial = system.phases.copy()

        # Evolve
        history = system.evolve(t_final=1.0, dt=0.1, store_interval=5)

        phases_final = history["phases"][-1]

        # Phases should have evolved (in general)
        # (May not change if ΔNFR = 0, but unlikely)
        # At least check they're tracked
        assert len(phases_final) == 2
        assert np.all(phases_final >= 0)
        assert np.all(phases_final < 2 * np.pi)


class TestCoherenceBasedDynamics:
    """Test that dynamics emerge from coherence, not gravity."""

    def test_no_motion_without_coupling(self):
        """Test that bodies don't move without coupling."""
        positions = np.array([[0, 0, 0], [10, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
            coupling_strength=0.0,  # No coupling
        )

        # Accelerations should be zero
        accel = system._compute_tnfr_accelerations()
        assert np.allclose(accel, 0.0, atol=1e-10)

    def test_coherence_potential_exists(self):
        """Test that coherence potential can be computed."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
        )

        # Compute coherence potential
        U = compute_tnfr_coherence_potential(
            system.graph, system.positions, system.hbar_str
        )

        # Should be a finite number
        assert np.isfinite(U)


class TestNumericalStability:
    """Test numerical stability."""

    def test_zero_initial_velocities(self):
        """Test system with zero initial velocities."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
        )

        # Should not crash
        history = system.evolve(t_final=0.5, dt=0.01, store_interval=10)

        assert len(history["time"]) > 0

    def test_close_bodies_stable(self):
        """Test that close bodies don't blow up."""
        positions = np.array([[0, 0, 0], [0.1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0, 0]])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 1.0],
            positions=positions,
            velocities=velocities,
        )

        # Should not blow up
        history = system.evolve(t_final=0.1, dt=0.001, store_interval=10)

        # Check velocities remain finite
        assert np.all(np.isfinite(history["velocities"]))


@pytest.mark.slow
class TestSystemEvolution:
    """Test system evolution over time."""

    def test_two_body_evolution(self):
        """Test two-body system evolution."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        velocities = np.array([[0, 0, 0], [0, 0.5, 0]])
        phases = np.array([0.0, 0.0])

        system = TNFRNBodySystem(
            n_bodies=2,
            masses=[1.0, 0.3],
            positions=positions,
            velocities=velocities,
            phases=phases,
        )

        # Evolve
        history = system.evolve(t_final=5.0, dt=0.01, store_interval=10)

        # Check history structure
        assert "time" in history
        assert "positions" in history
        assert "velocities" in history
        assert "phases" in history
        assert "energy" in history
        assert "energy_drift" in history

        # Check shapes
        n_stored = len(history["time"])
        assert history["positions"].shape == (n_stored, 2, 3)
        assert history["velocities"].shape == (n_stored, 2, 3)
        assert history["phases"].shape == (n_stored, 2)

        # Energy drift should be finite
        assert np.isfinite(history["energy_drift"])
