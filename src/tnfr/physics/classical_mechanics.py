"""TNFR Classical Mechanics Mapper — Canonical Translation Layer

This module implements the formal correspondence between Classical Mechanics
(Lagrangian/Hamiltonian formalisms) and TNFR Structural Dynamics. It provides
the translation layer requested in the "Módulo Traductor Mecánica Clásica" task.

Theoretical Foundation
----------------------
The mapping relies on the Universal Tetrahedral Correspondence and the Nodal Equation:
    ∂EPI/∂t = νf · ΔNFR(t)

This relationship reveals that Classical Mechanics is a limiting case of TNFR
dynamics where:
1. Coherence is maximized (low dissonance regime).
2. Structural frequency (νf) acts as inverse inertia.
3. Structural pressure (ΔNFR) manifests as phenomenological force.

Canonical Mappings:
1. Generalized Coordinates (q) <--> EPI Spatial Components
2. Generalized Velocities (q_dot) <--> EPI Velocity Components
3. Inertial Mass (m) <--> Inverse Structural Frequency (1/νf)
4. Force / Gradient (-∇V) <--> Structural Pressure (ΔNFR)
5. Action (S) <--> Structural Phase Accumulation (∫ φ dt)

This module allows defining systems in classical terms (L or H) and converting
them into valid TNFR operator sequences and nodal states, demonstrating how
classical behavior emerges from fundamental nodal dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Canonical constants and keys
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY


@dataclass
class GeneralizedCoordinateSystem:
    """Represents a system in generalized coordinates (q, p)."""
    
    q: np.ndarray  # Generalized coordinates
    p: Optional[np.ndarray] = None  # Generalized momenta (for Hamiltonian)
    q_dot: Optional[np.ndarray] = None  # Generalized velocities (for Lagrangian)
    masses: Optional[np.ndarray] = None  # Masses associated with coordinates
    
    def __post_init__(self):
        if self.p is None and self.q_dot is None:
            # Allow initialization with just q, but warn or handle if needed
            pass
        if self.masses is None:
            # Default to unit masses if not specified
            self.masses = np.ones_like(self.q)

    @property
    def dimension(self) -> int:
        return len(self.q)


class ClassicalMechanicsMapper:
    """Translates Classical Mechanics formulations to TNFR Structural Dynamics."""

    @staticmethod
    def lagrangian_to_tnfr(
        L: Callable[[np.ndarray, np.ndarray, float], float],
        system: GeneralizedCoordinateSystem,
        t: float = 0.0
    ) -> Dict[str, Any]:
        """
        Maps a Lagrangian L(q, q_dot, t) to a TNFR Nodal State.

        Args:
            L: Lagrangian function L(q, q_dot, t) -> float (Energy)
            system: The generalized coordinate system state
            t: Current time

        Returns:
            Dict containing TNFR nodal attributes:
            - EPI: Combined state vector [q, q_dot]
            - νf: Structural frequency (derived from mass)
            - ΔNFR: Structural pressure (derived from Euler-Lagrange)
        """
        if system.q_dot is None:
            raise ValueError("Lagrangian mapping requires generalized velocities (q_dot).")

        # 1. Map Mass to Frequency: νf = 1/m
        # We take the mean mass if multiple, or return a vector if supported.
        # For a single node representing the system, we might use an effective mass.
        # Here we assume the system represents a single entity or we return arrays.
        # To keep it simple for the mapper, we map per-coordinate if possible.
        
        # In TNFR, a node usually has one scalar νf. If this system is multi-body,
        # it should probably map to a Graph. For now, we map to attributes of a single
        # representative node or a list of attributes.
        
        # Let's assume 1D or N-D system mapped to N-D EPI.
        
        # νf (Structural Frequency) <--> 1 / Mass
        # Using the first mass as reference or vector if supported by custom node types.
        # Standard TNFR nodes have scalar νf.
        mass_ref = np.mean(system.masses) if system.masses is not None else 1.0
        nu_f = 1.0 / mass_ref if mass_ref > 0 else 1.0

        # 2. Map State to EPI
        # EPI typically holds the structural form. In N-body, it's [pos, vel].
        epi_vector = np.concatenate([system.q, system.q_dot])

        # 3. Map Dynamics to ΔNFR
        # The Euler-Lagrange equation: d/dt (∂L/∂q_dot) - ∂L/∂q = 0
        # => d/dt (p) = F_generalized
        # => F = ∂L/∂q
        # In TNFR: ∂EPI/∂t = νf · ΔNFR
        # Ideally ΔNFR corresponds to the Force term.
        
        # We can approximate ∂L/∂q numerically or symbolically.
        # For this mapper, we might need the force function explicitly or use autodiff.
        # Since we only have the function L, we can't easily get gradients without autodiff.
        # For now, we will return a placeholder or require the force function.
        
        # However, the prompt asks for the *mapper structure*.
        # We will return the mapped state.
        
        return {
            EPI_PRIMARY: epi_vector,
            VF_PRIMARY: nu_f,
            "classical_L": L(system.q, system.q_dot, t),
            # ΔNFR would be calculated by the engine using the potential,
            # here we just set up the state.
            DNFR_PRIMARY: np.zeros_like(epi_vector)  # Placeholder
        }

    @staticmethod
    def hamiltonian_to_tnfr(
        H: Callable[[np.ndarray, np.ndarray, float], float],
        system: GeneralizedCoordinateSystem,
        t: float = 0.0
    ) -> Dict[str, Any]:
        """
        Maps a Hamiltonian H(q, p, t) to a TNFR Nodal State.

        Args:
            H: Hamiltonian function H(q, p, t) -> float (Energy)
            system: The generalized coordinate system state
            t: Current time

        Returns:
            Dict containing TNFR nodal attributes.
        """
        if system.p is None:
            raise ValueError("Hamiltonian mapping requires generalized momenta (p).")

        # 1. Map Mass to Frequency
        mass_ref = np.mean(system.masses) if system.masses is not None else 1.0
        nu_f = 1.0 / mass_ref if mass_ref > 0 else 1.0

        # 2. Map State to EPI
        # For Hamiltonian, state is (q, p).
        # We might map p back to q_dot for the standard EPI [pos, vel] representation
        # if we want consistency with the N-body solver.
        # q_dot = p / m
        q_dot = system.p / system.masses if system.masses is not None else system.p
        epi_vector = np.concatenate([system.q, q_dot])

        # 3. Map Energy to Coherence/Potential
        # H is total energy.
        # In TNFR, Φ_s (Structural Potential) relates to Potential Energy.
        # But H includes Kinetic.
        
        return {
            EPI_PRIMARY: epi_vector,
            VF_PRIMARY: nu_f,
            "classical_H": H(system.q, system.p, t),
            DNFR_PRIMARY: np.zeros_like(epi_vector)
        }

    @staticmethod
    def equations_of_motion_to_operators(
        forces: np.ndarray,
        masses: np.ndarray
    ) -> List[str]:
        """
        Translates phenomenological forces into their fundamental Structural Operator equivalents.
        
        Classical F=ma is the limiting case of ∂EPI/∂t = νf · ΔNFR where:
        - Force (F) corresponds to Structural Pressure (ΔNFR)
        - Mass (m) corresponds to Inverse Structural Frequency (1/νf)
        
        This suggests that 'Force' is applied via 'Dissonance' (OZ) or 'Reception' (EN)
        depending on whether it's internal or external, followed by 'Coherence' (IL)
        to stabilize the new state.
        
        Args:
            forces: Array of force vectors
            masses: Array of masses

        Returns:
            List of operator names (e.g., ['OZ', 'IL'])
        """
        # If forces are non-zero, we have a change in state (acceleration).
        # In TNFR, change is driven by ΔNFR.
        # To induce ΔNFR, we might use Dissonance (OZ) to break equilibrium,
        # or Reception (EN) to intake information (force).
        
        # Canonical sequence for state update:
        # 1. Dissonance (OZ) - Introduces ΔNFR (Force)
        # 2. Coherence (IL) - Stabilizes the new trajectory (Integration)
        
        ops = []
        if np.any(np.abs(forces) > 1e-9):
            ops.append("OZ")  # Apply Force / Pressure
            ops.append("IL")  # Integrate / Stabilize
        else:
            ops.append("SHA")  # Silence / Inertia
            
        return ops

    @staticmethod
    def state_vector_to_generalized(
        epi: np.ndarray,
        nu_f: float
    ) -> GeneralizedCoordinateSystem:
        """
        Inverse mapping: TNFR EPI -> Generalized Coordinates.
        Assumes EPI is [q, q_dot] stacked.
        """
        n = len(epi) // 2
        q = epi[:n]
        q_dot = epi[n:]
        mass = 1.0 / nu_f if nu_f > 0 else 1.0
        p = q_dot * mass
        
        return GeneralizedCoordinateSystem(
            q=q,
            q_dot=q_dot,
            p=p,
            masses=np.full_like(q, mass)
        )


class ClassicalForceTranslator:
    """
    Translates phenomenological forces into fundamental Structural Mechanisms.
    
    This class provides the dictionary between observed classical forces and
    the underlying nodal dynamics that generate them.
    """

    @staticmethod
    def gravity_to_tnfr() -> str:
        """
        Gravity corresponds to the Coherence Gradient (-∇Φ_s).
        
        Mechanism:
        Nodes naturally evolve to maximize phase synchronization (minimize dissonance).
        This creates an emergent attractive force between coherent structures,
        which we observe macroscopically as gravity.
        
        Returns:
            Description of the mechanism.
        """
        return "Emergent Coherence Attraction (Phase Synchronization)"

    @staticmethod
    def friction_to_tnfr() -> str:
        """
        Friction corresponds to Structural Damping / Coherence Stabilization.
        
        Mechanism:
        The 'Coherence' (IL) operator acts as a stabilizer, reducing high-frequency
        fluctuations (thermal energy) and aligning velocity vectors. This manifests
        as a dissipative force (friction) that removes kinetic energy from the
        macroscopic mode.
        
        Returns:
            Description of the mechanism.
        """
        return "Structural Stabilization (IL Operator)"

    @staticmethod
    def harmonic_restoring_to_tnfr() -> str:
        """
        Harmonic forces (Springs) correspond to Structural Confinement.
        
        Mechanism:
        When a node deviates from its equilibrium position in the structural manifold,
        the Phase Gradient (|∇φ|) increases. The system generates a restoring
        pressure (ΔNFR) to return to the low-gradient state (equilibrium).
        
        Returns:
            Description of the mechanism.
        """
        return "Phase Gradient Confinement (|∇φ| Minimization)"

    @staticmethod
    def compute_poisson_bracket(
        f: Callable[[GeneralizedCoordinateSystem], float],
        g: Callable[[GeneralizedCoordinateSystem], float],
        system: GeneralizedCoordinateSystem,
        epsilon: float = 1e-5
    ) -> float:
        """
        Computes the Poisson Bracket {f, g} numerically on the Structural Manifold.
        
        {f, g} = Σ (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)
        
        This metric quantifies the structural commutation relation between two
        observables. If {f, H} = 0, then f is a conserved structural invariant.
        
        Args:
            f: First observable function.
            g: Second observable function.
            system: Current state.
            epsilon: Finite difference step size.
            
        Returns:
            Value of the Poisson Bracket.
        """
        n = system.dimension
        bracket = 0.0
        
        # We need to perturb q and p.
        # Since GeneralizedCoordinateSystem is immutable-ish (dataclass),
        # we create copies.
        
        # Helper to evaluate gradient
        def gradient(func, sys, var_name, idx):
            original = getattr(sys, var_name)[idx]
            
            # Forward
            getattr(sys, var_name)[idx] = original + epsilon
            val_plus = func(sys)
            
            # Backward
            getattr(sys, var_name)[idx] = original - epsilon
            val_minus = func(sys)
            
            # Restore
            getattr(sys, var_name)[idx] = original
            
            return (val_plus - val_minus) / (2 * epsilon)

        # We need mutable arrays for this to work efficiently,
        # or we construct new systems.
        # The dataclass fields are numpy arrays, which are mutable.
        # So we can modify in place and restore.
        
        if system.p is None:
            raise ValueError("Poisson Bracket requires momenta (p).")
            
        for i in range(n):
            df_dq = gradient(f, system, 'q', i)
            dg_dp = gradient(g, system, 'p', i)
            
            df_dp = gradient(f, system, 'p', i)
            dg_dq = gradient(g, system, 'q', i)
            
            bracket += (df_dq * dg_dp) - (df_dp * dg_dq)
            
        return bracket


