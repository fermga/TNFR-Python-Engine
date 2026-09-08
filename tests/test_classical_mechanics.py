"""Validation tests for the explicit classical-mechanics adapters.

Orbit agreement checks the declared force laws and symplectic integrator. It is
not treated as evidence that the first-order nodal equation derives mechanics.
"""

import math
import unittest

import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics.nbody import (
    NBodySystem,
    compute_gravitational_dnfr,
    compute_newtonian_acceleration,
)
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.errors import TNFRValueError
from tnfr.physics.classical_mechanics import (
    ClassicalForceTranslator,
    ClassicalMechanicsMapper,
    GeneralizedCoordinateSystem,
)
from tnfr.types import TNFRNode
from tnfr.validation import validate_sequence


class TestClassicalAdapter(unittest.TestCase):

    def setUp(self):
        # Create a mock node
        self.node: TNFRNode = {
            EPI_PRIMARY: np.zeros(2),  # 1D system: [q, q_dot]
            VF_PRIMARY: 1.0,  # Mass = 1.0
            DNFR_PRIMARY: np.zeros(2),  # Force = 0
        }

    def test_force_presence_maps_to_complete_grammar_words(self):
        masses = np.array([1.0, 2.0])
        active = ClassicalMechanicsMapper.equations_of_motion_to_operators(
            np.array([[1.0, 0.0], [0.0, 0.0]]), masses
        )
        inactive = ClassicalMechanicsMapper.equations_of_motion_to_operators(
            np.zeros((2, 2)), masses
        )

        self.assertEqual(
            active, ["emission", "dissonance", "coherence", "silence"]
        )
        self.assertEqual(inactive, ["emission", "coherence", "silence"])
        self.assertTrue(validate_sequence(active).passed)
        self.assertTrue(validate_sequence(inactive).passed)

        with self.assertRaises(TNFRValueError):
            ClassicalMechanicsMapper.equations_of_motion_to_operators(
                np.zeros((2, 2)), np.array([1.0])
            )
        for forces, masses in (
            (np.array([[1.0 + 1.0j]]), np.array([1.0])),
            (np.array([[True]]), np.array([1.0])),
        ):
            with self.subTest(forces=forces):
                with self.assertRaises(TNFRValueError):
                    ClassicalMechanicsMapper.equations_of_motion_to_operators(
                        forces, masses
                    )

    def test_mapper_exposes_adapter_scope_without_certifying_coherence(self):
        system = GeneralizedCoordinateSystem(
            q=np.array([1.0, -1.0]),
            q_dot=np.array([0.25, -0.5]),
            masses=np.array([1.0, 3.0]),
        )

        mapped = ClassicalMechanicsMapper.lagrangian_to_tnfr(
            lambda q, q_dot, t: float(np.sum(q_dot**2) - np.sum(q**2)),
            system,
            t=2.0,
        )

        self.assertEqual(mapped[VF_PRIMARY], 0.5)
        np.testing.assert_array_equal(mapped[DNFR_PRIMARY], np.zeros(4))
        self.assertIsInstance(mapped["classical_L"], float)
        metadata = mapped["classical_adapter"]
        self.assertEqual(
            metadata["nu_f_semantics"], "inverse_mean_mass_adapter_only"
        )
        self.assertEqual(
            metadata["mass_reduction"],
            "arithmetic_mean_single_node_summary",
        )
        self.assertEqual(metadata["mass_reference"], 2.0)
        self.assertEqual(
            metadata["dnfr_semantics"], "zero_placeholder_no_force_bridge"
        )
        self.assertFalse(metadata["force_bridge_supplied"])
        self.assertFalse(metadata["canonical_coherence_available"])

    def test_hamiltonian_mapping_decodes_velocity_only_by_adapter_convention(self):
        system = GeneralizedCoordinateSystem(
            q=np.array([1.0, 2.0]),
            p=np.array([2.0, 6.0]),
            masses=np.array([2.0, 3.0]),
        )

        mapped = ClassicalMechanicsMapper.hamiltonian_to_tnfr(
            lambda q, p, t: float(np.sum(q**2) + np.sum(p**2)), system
        )

        np.testing.assert_allclose(mapped[EPI_PRIMARY], [1.0, 2.0, 1.0, 2.0])
        self.assertEqual(mapped[VF_PRIMARY], 0.4)
        self.assertFalse(
            mapped["classical_adapter"]["canonical_coherence_available"]
        )

    def test_mapper_rejects_ambiguous_or_nonfinite_adapter_values(self):
        lagrangian_system = GeneralizedCoordinateSystem(
            q=np.array([1.0]), q_dot=np.array([0.0])
        )
        hamiltonian_system = GeneralizedCoordinateSystem(
            q=np.array([1.0]), p=np.array([0.0])
        )

        with self.assertRaises(TNFRValueError):
            ClassicalMechanicsMapper.lagrangian_to_tnfr(
                lambda q, q_dot, t: np.array([1.0]), lagrangian_system
            )
        with self.assertRaises(TNFRValueError):
            ClassicalMechanicsMapper.hamiltonian_to_tnfr(
                lambda q, p, t: float("nan"), hamiltonian_system
            )
        with self.assertRaises(TNFRValueError):
            ClassicalMechanicsMapper.lagrangian_to_tnfr(
                lambda q, q_dot, t: 0.0, lagrangian_system, t=True
            )

    def test_poisson_bracket_restores_state_after_observable_failure(self):
        system = GeneralizedCoordinateSystem(
            q=np.array([1.0]),
            p=np.array([2.0]),
            masses=np.array([1.0]),
        )
        q_before = system.q.copy()
        p_before = system.p.copy()

        def failing_observable(_system):
            raise RuntimeError("observable failed")

        with self.assertRaises(RuntimeError):
            ClassicalForceTranslator.compute_poisson_bracket(
                failing_observable,
                lambda state: float(state.p[0]),
                system,
            )

        np.testing.assert_array_equal(system.q, q_before)
        np.testing.assert_array_equal(system.p, p_before)
        with self.assertRaises(TNFRValueError):
            ClassicalForceTranslator.compute_poisson_bracket(
                lambda state: float(state.q[0]),
                lambda state: float(state.p[0]),
                system,
                epsilon=0.0,
            )

        bracket = ClassicalForceTranslator.compute_poisson_bracket(
            lambda state: float(state.q[0]),
            lambda state: float(state.p[0]),
            system,
        )
        self.assertAlmostEqual(bracket, 1.0)
        with self.assertRaises(TNFRValueError):
            ClassicalForceTranslator.compute_poisson_bracket(
                lambda state: complex(state.q[0], 1.0),
                lambda state: float(state.p[0]),
                system,
            )

    def test_newtonian_acceleration_name_preserves_legacy_api_and_scope(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        masses = np.array([1.0, 2.0])

        canonical_name = compute_newtonian_acceleration(positions, masses)
        legacy_name = compute_gravitational_dnfr(positions, masses)
        np.testing.assert_array_equal(canonical_name, legacy_name)
        np.testing.assert_allclose(
            canonical_name,
            [[0.5, 0.0, 0.0], [-0.25, 0.0, 0.0]],
        )

        adapter = NBodySystem(2, masses=masses)
        self.assertEqual(
            adapter.graph.graph["MODEL_SCOPE"],
            "external_newtonian_nbody_adapter",
        )
        self.assertEqual(
            adapter.graph.graph["NU_F_SEMANTICS"],
            "inverse_mass_adapter_metadata_only",
        )
        self.assertFalse(adapter.graph.graph["FORCE_BRIDGE_MATERIALIZED"])
        self.assertFalse(adapter.graph.graph["CANONICAL_C_T_AVAILABLE"])
        with self.assertRaises(ValueError):
            compute_newtonian_acceleration(positions.astype(complex), masses)
        with self.assertRaises(ValueError):
            NBodySystem(1, masses=np.array([1.0 + 0.0j]))

    def test_harmonic_oscillator(self):
        """
        Verify that the declared harmonic adapter reproduces
        simple harmonic motion.

        System: Mass m=1, k=1.
        Hamiltonian: H = p^2/2m + k*q^2/2
        Force: F = -k*q
        Period: T = 2*pi * sqrt(m/k) = 2*pi
        """
        k = 1.0
        m = 1.0
        omega = math.sqrt(k / m)
        period = 2 * math.pi / omega

        # Initial State: q=1, q_dot=0
        q_init = np.array([1.0])
        q_dot_init = np.array([0.0])

        # Setup Node
        system = GeneralizedCoordinateSystem(
            q=q_init, q_dot=q_dot_init, masses=np.array([m])
        )
        mapped_state = ClassicalMechanicsMapper.lagrangian_to_tnfr(
            lambda q, qd, t: float(
                0.5 * m * np.sum(qd**2) - 0.5 * k * np.sum(q**2)
            ),
            system,
        )

        self.node.update(mapped_state)

        # Explicit harmonic force evaluator for the mechanical adapter.
        def harmonic_force(n: TNFRNode) -> np.ndarray:
            epi = n[EPI_PRIMARY]
            q = epi[:1]
            # The full adapter force slot is [0, F], with F = -k*q.
            f = -k * q
            return np.concatenate([np.zeros_like(f), f])

        self.node[DNFR_PRIMARY] = harmonic_force(self.node)

        # Integrate for one period
        dt = 0.01
        steps = int(period / dt)

        for _ in range(steps):
            TNFRSymplecticIntegrator.velocity_verlet(self.node, dt, harmonic_force)

        # Check final position (should be close to 1.0)
        q_final = self.node[EPI_PRIMARY][:1]
        self.assertAlmostEqual(q_final[0], 1.0, delta=0.1)

        # Check classical energy drift.
        # E = 0.5*q^2 + 0.5*q_dot^2
        q_end = self.node[EPI_PRIMARY][:1]
        q_dot_end = self.node[EPI_PRIMARY][1:]
        energy = 0.5 * k * q_end[0] ** 2 + 0.5 * m * q_dot_end[0] ** 2
        expected_energy = 0.5 * k * 1.0**2
        self.assertAlmostEqual(energy, expected_energy, delta=0.01)

    def test_kepler_orbit(self):
        """
        Verify one-period closure for the explicit circular-orbit adapter.
        """
        # 2D System
        # Sun at origin (fixed), Planet orbiting
        # G*M = 1
        gm = 1.0

        # Initial State: Circular orbit at r=1
        # v = sqrt(GM/r) = 1
        q_init = np.array([1.0, 0.0])
        q_dot_init = np.array([0.0, 1.0])

        # Setup Node
        system = GeneralizedCoordinateSystem(q=q_init, q_dot=q_dot_init)
        mapped_state = ClassicalMechanicsMapper.lagrangian_to_tnfr(
            lambda q, qd, t: 0.5 * np.sum(qd**2)
            + gm / np.linalg.norm(q),  # L = T - V (V = -GM/r)
            system,
        )
        self.node.update(mapped_state)

        # Force Evaluator (Central Gravity)
        def gravity_force(n: TNFRNode) -> np.ndarray:
            epi = n[EPI_PRIMARY]
            q = epi[:2]
            r = np.linalg.norm(q)
            # F = -GM/r^3 * q
            f = -gm / (r**3) * q
            return np.concatenate([np.zeros_like(f), f])

        self.node[DNFR_PRIMARY] = gravity_force(self.node)

        # Integrate for one period T = 2*pi*sqrt(r^3/GM) = 2*pi
        period = 2 * math.pi
        dt = 0.01
        steps = int(period / dt)

        for _ in range(steps):
            TNFRSymplecticIntegrator.velocity_verlet(self.node, dt, gravity_force)

        # Check return to start
        q_final = self.node[EPI_PRIMARY][:2]
        dist = np.linalg.norm(q_final - q_init)
        self.assertLess(dist, 0.1)


if __name__ == "__main__":
    unittest.main()
