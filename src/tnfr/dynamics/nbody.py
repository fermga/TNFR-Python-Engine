"""Classical N-body problem implementation in TNFR structural framework.

⚠️ **IMPORTANT LIMITATION**: This module ASSUMES Newtonian gravitational potential:
   U(q) = -Σ_{i<j} G * m_i * m_j / |r_i - r_j|

This is an **external assumption**, NOT derived from TNFR first principles!

For a PURE TNFR implementation (no gravitational assumption), see:
   tnfr.dynamics.nbody_tnfr

That module derives dynamics from coherence potential and Hamiltonian
commutator, with NO classical force law assumptions.

Purpose of This Module
-----------------------

This module demonstrates how TNFR can **reproduce** classical mechanics
when we explicitly map classical potentials into the TNFR framework.
It shows the correspondence:

Classical Mechanics   ←→   TNFR Framework
-------------------        ---------------
Position q            ←→   EPI spatial component
Velocity v            ←→   EPI velocity component
Mass m                ←→   1/νf (structural inertia)
Force F = -∇U         ←→   ΔNFR (ASSUMED from classical U)
Newton's 2nd law      ←→   Nodal equation ∂EPI/∂t = νf·ΔNFR

Comparison:
-----------

**This module** (nbody.py):
```python
# Assumes gravitational potential
U = -Σ G*m_i*m_j/r_ij
F = -∇U  # Classical force
ΔNFR = F/m  # External assumption
```

**Pure TNFR** (nbody_tnfr.py):
```python
# NO assumed potential
H_int = H_coh + H_freq + H_coupling
ΔNFR = i[H_int, ·]/ℏ_str  # From Hamiltonian
# Forces emerge from coherence/phase sync
```

Theoretical Foundation
----------------------

The classical N-body problem emerges from TNFR as the **low-dissonance
coherence regime** where:

1. **Mass as inverse frequency**: m_i = 1/νf_i
   High mass → low structural reorganization rate (inertia)
   Low mass → high structural reorganization rate (responsiveness)

2. **Gravitational potential as coherence potential** (ASSUMED):
   U(q) = -Σ_{i<j} G * m_i * m_j / |r_i - r_j|

   This potential encodes structural stability landscape. Nodes
   naturally evolve toward configurations of higher coherence
   (lower potential energy).

3. **Nodal equation integration**:
   ∂EPI/∂t = νf · ΔNFR(t)

   Where EPI encodes position and velocity, and ΔNFR is computed
   from the gravitational coherence gradient (ASSUMED).

Mathematical Correspondence
---------------------------

Classical mechanics:     TNFR structural dynamics:
- Position q_i          → EPI spatial component
- Velocity v_i          → EPI velocity component
- Mass m_i              → 1/νf_i (structural inertia)
- Force F_i = -∇U       → ΔNFR (coherence gradient, ASSUMED)
- Newton's 2nd law      → Nodal equation ∂EPI/∂t = νf·ΔNFR

Conservation Laws
-----------------

The implementation preserves:
- Total energy (H_int = T + U)
- Linear momentum (Σ m_i * v_i)
- Angular momentum (Σ r_i × m_i * v_i)

These emerge naturally from the Hamiltonian structure and
translational/rotational symmetry of the coherence potential.

References
----------
- tnfr.dynamics.nbody_tnfr: Pure TNFR n-body (no assumptions)
- docs/source/theory/07_emergence_classical_mechanics.md
- docs/source/theory/08_classical_mechanics_euler_lagrange.md
- TNFR.pdf: Canonical nodal equation (§2.3)
- AGENTS.md: Canonical invariants (§3)

Examples
--------
Two-body orbit (Earth-Moon system) with ASSUMED gravity:

>>> from tnfr.dynamics.nbody import NBodySystem
>>> import numpy as np
>>>
>>> # Create 2-body system (dimensionless units)
>>> system = NBodySystem(
...     n_bodies=2,
...     masses=[1.0, 0.012],  # Mass ratio ~ Earth/Moon
...     G=1.0  # Gravitational constant (ASSUMED)
... )
>>>
>>> # Initialize circular orbit
>>> positions = np.array([
...     [0.0, 0.0, 0.0],      # Earth at origin
...     [1.0, 0.0, 0.0]       # Moon at distance 1
... ])
>>> velocities = np.array([
...     [0.0, 0.0, 0.0],      # Earth at rest (CM frame)
...     [0.0, 1.0, 0.0]       # Moon with tangential velocity
... ])
>>>
>>> system.set_state(positions, velocities)
>>>
>>> # Evolve system (structural time)
>>> history = system.evolve(t_final=10.0, dt=0.01)
>>>
>>> # Check energy conservation
>>> E0 = history['energy'][0]
>>> E_final = history['energy'][-1]
>>> print(f"Energy drift: {abs(E_final - E0) / abs(E0):.2e}")

Three-body system (Figure-8 orbit):

>>> system = NBodySystem(n_bodies=3, masses=[1.0, 1.0, 1.0], G=1.0)
>>> # Use known figure-8 initial conditions
>>> # (See Chenciner & Montgomery, 2000)
>>> history = system.evolve(t_final=6.3, dt=0.001)
>>> system.plot_trajectories(history)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from ..structural import create_nfr
from ..types import TNFRGraph

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = (
    "NBodySystem",
    "gravitational_potential",
    "gravitational_force",
    "compute_gravitational_dnfr",
)


def gravitational_potential(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> float:
    """Compute total Newtonian gravitational potential energy.

    U(q) = -Σ_{i<j} G * m_i * m_j / |r_i - r_j|

    This is the coherence potential in TNFR language: lower U means
    higher structural stability (attractive gravitational well).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Positions of N bodies in 3D space
    masses : ndarray, shape (N,)
        Masses of N bodies
    G : float, default=1.0
        Gravitational constant (in appropriate units)
    softening : float, default=0.0
        Softening length to avoid singularities at r=0.
        Effective distance: r_eff = sqrt(r² + ε²)

    Returns
    -------
    U : float
        Total gravitational potential energy (negative)

    Notes
    -----
    The negative sign ensures bound states have U < 0, matching
    the TNFR convention that coherent states minimize the potential.
    """
    N = len(positions)
    U = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r_ij = positions[j] - positions[i]
            dist = np.sqrt(np.sum(r_ij**2) + softening**2)
            U -= G * masses[i] * masses[j] / dist

    return U


def gravitational_force(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> NDArray[np.floating]:
    """Compute gravitational forces on all bodies.

    F_i = -∇_i U = Σ_{j≠i} G * m_i * m_j * (r_j - r_i) / |r_j - r_i|³

    In TNFR language: force is the coherence gradient pointing toward
    higher stability (lower potential).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Positions of N bodies
    masses : ndarray, shape (N,)
        Masses of N bodies
    G : float, default=1.0
        Gravitational constant
    softening : float, default=0.0
        Softening length for numerical stability

    Returns
    -------
    forces : ndarray, shape (N, 3)
        Gravitational forces on each body

    Notes
    -----
    Force points from lower to higher coherence (lower potential).
    Newton's 3rd law (F_ij = -F_ji) emerges from symmetry of U(q).
    """
    N = len(positions)
    forces = np.zeros_like(positions)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            r_ij = positions[j] - positions[i]
            dist_sq = np.sum(r_ij**2) + softening**2
            dist = np.sqrt(dist_sq)
            dist_cubed = dist_sq * dist

            # F_i = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|³
            forces[i] += G * masses[i] * masses[j] * r_ij / dist_cubed

    return forces


def compute_gravitational_dnfr(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> NDArray[np.floating]:
    """Compute ΔNFR from gravitational coherence gradient.

    ΔNFR_i = F_i / m_i = a_i (acceleration)

    This is the structural reorganization operator that drives evolution
    via the nodal equation: ∂EPI/∂t = νf · ΔNFR

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Positions of N bodies
    masses : ndarray, shape (N,)
        Masses (or inverse frequencies: m = 1/νf)
    G : float, default=1.0
        Gravitational constant
    softening : float, default=0.0
        Softening length

    Returns
    -------
    dnfr : ndarray, shape (N, 3)
        ΔNFR values (accelerations) for each body

    Notes
    -----
    ΔNFR = a = F/m is independent of mass by equivalence principle.
    This is the reorganization "pressure" that drives structural change.
    """
    forces = gravitational_force(positions, masses, G, softening)

    # ΔNFR = F/m (acceleration)
    # Broadcast division: (N, 3) / (N, 1) -> (N, 3)
    dnfr = forces / masses[:, np.newaxis]

    return dnfr


class NBodySystem:
    """Classical N-body gravitational system in TNFR framework.

    Implements N particles (resonant nodes) coupled through Newtonian
    gravitational potential. Positions and velocities are encoded as
    EPI components, masses as inverse frequencies (m = 1/νf), and
    evolution follows the canonical nodal equation.

    Attributes
    ----------
    n_bodies : int
        Number of bodies in the system
    masses : ndarray, shape (N,)
        Masses of bodies (m_i = 1/νf_i)
    G : float
        Gravitational constant
    softening : float
        Softening length for numerical stability
    positions : ndarray, shape (N, 3)
        Current positions
    velocities : ndarray, shape (N, 3)
        Current velocities
    time : float
        Current structural time
    graph : TNFRGraph
        NetworkX graph storing nodes as NFRs

    Notes
    -----
    The system maintains TNFR canonical invariants:
    - EPI encodes (position, velocity) pairs
    - νf = 1/m (structural frequency from mass)
    - ΔNFR computed from gravitational gradient
    - Evolution via ∂EPI/∂t = νf · ΔNFR

    Conservation laws emerge naturally from Hamiltonian structure.
    """

    def __init__(
        self,
        n_bodies: int,
        masses: List[float] | NDArray[np.floating],
        G: float = 1.0,
        softening: float = 0.0,
    ):
        """Initialize N-body system.

        Parameters
        ----------
        n_bodies : int
            Number of bodies
        masses : array_like, shape (N,)
            Masses of bodies (must be positive)
        G : float, default=1.0
            Gravitational constant
        softening : float, default=0.0
            Softening length (ε) for numerical stability.
            Prevents singularities at r=0.

        Raises
        ------
        ValueError
            If masses are non-positive or dimensions mismatch
        """
        if n_bodies < 1:
            raise ValueError(f"n_bodies must be >= 1, got {n_bodies}")

        self.n_bodies = n_bodies
        self.masses = np.array(masses, dtype=float)

        if len(self.masses) != n_bodies:
            raise ValueError(f"masses length {len(self.masses)} != n_bodies {n_bodies}")

        if np.any(self.masses <= 0):
            raise ValueError("All masses must be positive")

        self.G = float(G)
        self.softening = float(softening)

        # State vectors
        self.positions = np.zeros((n_bodies, 3), dtype=float)
        self.velocities = np.zeros((n_bodies, 3), dtype=float)
        self.time = 0.0

        # Create TNFR graph representation
        self._build_graph()

    def _build_graph(self) -> None:
        """Build TNFR graph representation of the N-body system.

        Each body becomes a resonant node with:
        - νf = 1/m (structural frequency)
        - EPI encoding (position, velocity)
        - Phase initialized to 0 (can be set for rotation)
        - Fully connected topology (all-to-all gravitational coupling)
        """
        # Create empty graph (will add nodes manually)
        import networkx as nx

        self.graph: TNFRGraph = nx.Graph()
        self.graph.graph["name"] = "nbody_system"

        # Add nodes with TNFR attributes
        for i in range(self.n_bodies):
            node_id = f"body_{i}"

            # Structural frequency: νf = 1/m
            nu_f = 1.0 / self.masses[i]

            # Create NFR node
            _, _ = create_nfr(
                node_id,
                epi=1.0,  # Will be overwritten by set_state
                vf=nu_f,
                theta=0.0,  # Phase (for rotating systems)
                graph=self.graph,
            )

        # Add edges (all-to-all coupling for gravitational interaction)
        # Edge weight represents gravitational coupling strength
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                node_i = f"body_{i}"
                node_j = f"body_{j}"
                # Coupling weight: G * m_i * m_j
                weight = self.G * self.masses[i] * self.masses[j]
                self.graph.add_edge(node_i, node_j, weight=weight)

    def set_state(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> None:
        """Set system state (positions and velocities).

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Positions of N bodies
        velocities : ndarray, shape (N, 3)
            Velocities of N bodies

        Raises
        ------
        ValueError
            If shapes don't match (N, 3)
        """
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)

        expected_shape = (self.n_bodies, 3)
        if positions.shape != expected_shape:
            raise ValueError(f"positions shape {positions.shape} != {expected_shape}")
        if velocities.shape != expected_shape:
            raise ValueError(f"velocities shape {velocities.shape} != {expected_shape}")

        self.positions = positions.copy()
        self.velocities = velocities.copy()

        # Update EPI in graph nodes
        # EPI encodes state as dictionary with position/velocity
        for i in range(self.n_bodies):
            node_id = f"body_{i}"
            # Store as structured EPI
            epi_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }
            self.graph.nodes[node_id]["epi"] = epi_state

    def get_state(self) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get current state (positions and velocities).

        Returns
        -------
        positions : ndarray, shape (N, 3)
            Current positions
        velocities : ndarray, shape (N, 3)
            Current velocities
        """
        return self.positions.copy(), self.velocities.copy()

    def compute_energy(self) -> Tuple[float, float, float]:
        """Compute system energy (kinetic + potential).

        Returns
        -------
        kinetic : float
            Total kinetic energy T = Σ (1/2) m_i v_i²
        potential : float
            Total potential energy U (negative for bound systems)
        total : float
            Total energy H = T + U

        Notes
        -----
        Energy conservation is a fundamental check of integrator accuracy.
        For Hamiltonian systems, H should be constant over time.
        """
        # Kinetic energy: T = Σ (1/2) m_i v_i²
        v_squared = np.sum(self.velocities**2, axis=1)
        kinetic = 0.5 * np.sum(self.masses * v_squared)

        # Potential energy: U = -Σ_{i<j} G m_i m_j / r_ij
        potential = gravitational_potential(self.positions, self.masses, self.G, self.softening)

        total = kinetic + potential

        return kinetic, potential, total

    def compute_momentum(self) -> NDArray[np.floating]:
        """Compute total linear momentum.

        Returns
        -------
        momentum : ndarray, shape (3,)
            Total momentum P = Σ m_i v_i

        Notes
        -----
        For isolated systems, momentum should be conserved (constant).
        """
        momentum = np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0)
        return momentum

    def compute_angular_momentum(self) -> NDArray[np.floating]:
        """Compute total angular momentum about origin.

        Returns
        -------
        angular_momentum : ndarray, shape (3,)
            Total angular momentum L = Σ r_i × m_i v_i

        Notes
        -----
        For central force systems, angular momentum is conserved.
        """
        L = np.zeros(3)
        for i in range(self.n_bodies):
            L += self.masses[i] * np.cross(self.positions[i], self.velocities[i])
        return L

    def step(self, dt: float) -> None:
        """Advance system by one time step using velocity Verlet.

        The velocity Verlet integrator is symplectic (preserves phase space
        volume) and provides excellent long-term energy conservation.

        Algorithm:
        1. r(t+dt) = r(t) + v(t)*dt + (1/2)*a(t)*dt²
        2. a(t+dt) = compute acceleration at new positions
        3. v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt

        Parameters
        ----------
        dt : float
            Time step (in structural time units)

        Notes
        -----
        This integrator is equivalent to applying the nodal equation:
        ∂EPI/∂t = νf · ΔNFR with νf = 1/m and ΔNFR = acceleration.
        """
        # Compute acceleration at current time: a(t) = ΔNFR
        accel_t = compute_gravitational_dnfr(self.positions, self.masses, self.G, self.softening)

        # Update positions: r(t+dt) = r(t) + v(t)*dt + (1/2)*a(t)*dt²
        self.positions += self.velocities * dt + 0.5 * accel_t * dt**2

        # Compute acceleration at new time: a(t+dt)
        accel_t_plus_dt = compute_gravitational_dnfr(
            self.positions, self.masses, self.G, self.softening
        )

        # Update velocities: v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt
        self.velocities += 0.5 * (accel_t + accel_t_plus_dt) * dt

        # Update structural time
        self.time += dt

        # Update graph representation
        for i in range(self.n_bodies):
            node_id = f"body_{i}"
            epi_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }
            self.graph.nodes[node_id]["epi"] = epi_state

    def evolve(
        self,
        t_final: float,
        dt: float,
        store_interval: int = 1,
    ) -> Dict[str, Any]:
        """Evolve system from current time to t_final.

        Parameters
        ----------
        t_final : float
            Final structural time
        dt : float
            Time step for integration
        store_interval : int, default=1
            Store state every N steps (for memory efficiency)

        Returns
        -------
        history : dict
            Dictionary containing:
            - 'time': array of time points
            - 'positions': array of positions (n_steps, N, 3)
            - 'velocities': array of velocities (n_steps, N, 3)
            - 'energy': array of total energies
            - 'kinetic': array of kinetic energies
            - 'potential': array of potential energies
            - 'momentum': array of momentum vectors (n_steps, 3)
            - 'angular_momentum': array of L vectors (n_steps, 3)

        Notes
        -----
        The evolution implements the nodal equation iteratively.
        Conservation laws are tracked for validation.
        """
        n_steps = int((t_final - self.time) / dt)

        if n_steps < 1:
            raise ValueError(f"t_final {t_final} <= current time {self.time}")

        # Pre-allocate storage
        n_stored = (n_steps // store_interval) + 1
        times = np.zeros(n_stored)
        positions_hist = np.zeros((n_stored, self.n_bodies, 3))
        velocities_hist = np.zeros((n_stored, self.n_bodies, 3))
        energies = np.zeros(n_stored)
        kinetic_energies = np.zeros(n_stored)
        potential_energies = np.zeros(n_stored)
        momenta = np.zeros((n_stored, 3))
        angular_momenta = np.zeros((n_stored, 3))

        # Store initial state
        store_idx = 0
        times[store_idx] = self.time
        positions_hist[store_idx] = self.positions.copy()
        velocities_hist[store_idx] = self.velocities.copy()
        K, U, E = self.compute_energy()
        kinetic_energies[store_idx] = K
        potential_energies[store_idx] = U
        energies[store_idx] = E
        momenta[store_idx] = self.compute_momentum()
        angular_momenta[store_idx] = self.compute_angular_momentum()
        store_idx += 1

        # Evolution loop
        for step in range(n_steps):
            self.step(dt)

            # Store state if needed
            if (step + 1) % store_interval == 0 and store_idx < n_stored:
                times[store_idx] = self.time
                positions_hist[store_idx] = self.positions.copy()
                velocities_hist[store_idx] = self.velocities.copy()
                K, U, E = self.compute_energy()
                kinetic_energies[store_idx] = K
                potential_energies[store_idx] = U
                energies[store_idx] = E
                momenta[store_idx] = self.compute_momentum()
                angular_momenta[store_idx] = self.compute_angular_momentum()
                store_idx += 1

        return {
            "time": times[:store_idx],
            "positions": positions_hist[:store_idx],
            "velocities": velocities_hist[:store_idx],
            "energy": energies[:store_idx],
            "kinetic": kinetic_energies[:store_idx],
            "potential": potential_energies[:store_idx],
            "momentum": momenta[:store_idx],
            "angular_momentum": angular_momenta[:store_idx],
        }

    def plot_trajectories(
        self,
        history: Dict[str, Any],
        ax: Optional[Any] = None,
        show_energy: bool = True,
    ) -> Figure:
        """Plot trajectories and energy evolution.

        Parameters
        ----------
        history : dict
            Result from evolve() method
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates new figure.
        show_energy : bool, default=True
            If True, also plot energy conservation

        Returns
        -------
        fig : matplotlib Figure
            Figure object containing plots

        Raises
        ------
        ImportError
            If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install 'tnfr[viz-basic]'"
            ) from exc

        if show_energy:
            fig = plt.figure(figsize=(14, 6))
            ax_3d = fig.add_subplot(121, projection="3d")
            ax_energy = fig.add_subplot(122)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax_3d = fig.add_subplot(111, projection="3d")

        # Plot 3D trajectories
        positions = history["positions"]
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n_bodies))

        for i in range(self.n_bodies):
            traj = positions[:, i, :]
            ax_3d.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=colors[i],
                label=f"Body {i+1} (m={self.masses[i]:.2f})",
                alpha=0.7,
            )
            # Mark initial position
            ax_3d.scatter(
                traj[0, 0],
                traj[0, 1],
                traj[0, 2],
                color=colors[i],
                s=100,
                marker="o",
            )
            # Mark final position
            ax_3d.scatter(
                traj[-1, 0],
                traj[-1, 1],
                traj[-1, 2],
                color=colors[i],
                s=50,
                marker="x",
            )

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("N-Body Trajectories (TNFR Framework)")
        ax_3d.legend()

        if show_energy:
            # Plot energy conservation
            time = history["time"]
            E = history["energy"]
            E0 = E[0]

            ax_energy.plot(
                time,
                (E - E0) / abs(E0) * 100,
                label="Relative energy error (%)",
                color="red",
            )
            ax_energy.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax_energy.set_xlabel("Structural Time")
            ax_energy.set_ylabel("ΔE/E₀ (%)")
            ax_energy.set_title("Energy Conservation Check")
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
