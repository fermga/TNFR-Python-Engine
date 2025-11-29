"""Validation tests for Classical Mechanics emergence from TNFR Dynamics.

These tests verify that the ClassicalMechanicsMapper and TNFRSymplecticIntegrator
correctly reproduce standard classical phenomena (Harmonic Oscillator, Kepler Problem)
as limiting cases of Nodal Dynamics.
"""

import math
import unittest

import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.physics.classical_mechanics import ClassicalMechanicsMapper, GeneralizedCoordinateSystem
from tnfr.types import TNFRNode


class TestClassicalEmergence(unittest.TestCase):

    def setUp(self):
        # Create a mock node
        self.node: TNFRNode = {
            EPI_PRIMARY: np.zeros(2),  # 1D system: [q, q_dot]
            VF_PRIMARY: 1.0,           # Mass = 1.0
            DNFR_PRIMARY: np.zeros(2)  # Force = 0
        }

    def test_harmonic_oscillator(self):
        """
        Verify that a TNFR node under harmonic confinement reproduces
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
        system = GeneralizedCoordinateSystem(q=q_init, q_dot=q_dot_init, masses=np.array([m]))
        mapped_state = ClassicalMechanicsMapper.lagrangian_to_tnfr(
            lambda q, qd, t: 0.5*m*qd**2 - 0.5*k*q**2, # L = T - V
            system
        )
        
        self.node.update(mapped_state)
        
        # Force Evaluator (Harmonic Confinement)
        def harmonic_force(n: TNFRNode) -> np.ndarray:
            epi = n[EPI_PRIMARY]
            q = epi[:1]
            # F = -k*q
            # ΔNFR = F (since νf=1/m=1, a = F)
            # We return full ΔNFR vector [0, F]
            f = -k * q
            return np.concatenate([np.zeros_like(f), f])
            
        # Integrate for one period
        dt = 0.01
        steps = int(period / dt)
        
        history_q = []
        for _ in range(steps):
            TNFRSymplecticIntegrator.velocity_verlet(self.node, dt, harmonic_force)
            q_curr = self.node[EPI_PRIMARY][:1]
            history_q.append(q_curr[0])
            
        # Check final position (should be close to 1.0)
        q_final = self.node[EPI_PRIMARY][:1]
        self.assertAlmostEqual(q_final[0], 1.0, delta=0.1)
        
        # Check energy conservation?
        # E = 0.5*q^2 + 0.5*q_dot^2
        q_end = self.node[EPI_PRIMARY][:1]
        q_dot_end = self.node[EPI_PRIMARY][1:]
        energy = 0.5 * k * q_end[0]**2 + 0.5 * m * q_dot_end[0]**2
        expected_energy = 0.5 * k * 1.0**2
        self.assertAlmostEqual(energy, expected_energy, delta=0.01)

    def test_kepler_orbit(self):
        """
        Verify that a TNFR node under central coherence gradient (gravity)
        reproduces an elliptical orbit.
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
            lambda q, qd, t: 0.5*np.sum(qd**2) + gm/np.linalg.norm(q), # L = T - V (V = -GM/r)
            system
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

if __name__ == '__main__':
    unittest.main()
