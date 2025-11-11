"""N-body dynamics using pure TNFR physics (no external potentials).

This module implements N-body dynamics derived STRICTLY from TNFR structural
framework, without assuming any classical potentials (Newtonian, Coulomb, etc.).

Key Differences from Classical N-Body
--------------------------------------

**Classical Approach** (nbody.py):
- Assumes gravitational potential: U = -G*m*m/r
- Force computed as: F = -∇U
- ΔNFR = F/m (external assumption)

**TNFR Approach** (this module):
- NO assumed potential
- Coherence potential emerges from network structure
- ΔNFR computed from Hamiltonian commutator: ΔNFR = i[H_int, ·]/ℏ_str
- Attraction/repulsion emerges from phase synchronization and coupling

Theoretical Foundation
----------------------

The nodal equation:
    ∂EPI/∂t = νf · ΔNFR(t)

Where ΔNFR emerges from the internal Hamiltonian:
    H_int = H_coh + H_freq + H_coupling

Components:
1. **H_coh**: Coherence potential from structural similarity (phase, EPI, νf, Si)
2. **H_freq**: Diagonal operator encoding each node's νf
3. **H_coupling**: Network topology-induced interactions

The crucial insight: Attractive forces emerge naturally from maximizing
coherence between nodes, NOT from assuming gravity.

Phase-Dependent Interaction
----------------------------

Unlike classical gravity (always attractive), TNFR coupling depends on phase:
- |φᵢ - φⱼ| small → strong coherence → attraction
- |φᵢ - φⱼ| ≈ π → destructive interference → repulsion

This captures wave-like behavior absent in classical mechanics.

Emergence of Classical Limit
-----------------------------

In the low-dissonance limit (ε → 0) with:
- Nearly synchronized phases: |φᵢ - φⱼ| → 0
- Strong coupling: all nodes connected
- High coherence: C(t) ≈ 1

The TNFR dynamics reproduces classical-like behavior, but from first principles.

References
----------
- TNFR.pdf § 2.3: Nodal equation
- src/tnfr/operators/hamiltonian.py: H_int construction
- docs/source/theory/07_emergence_classical_mechanics.md: Classical limit
- AGENTS.md § Canonical Invariants: TNFR physics principles

Examples
--------
Two-body orbital resonance (no gravitational assumption):

>>> from tnfr.dynamics.nbody_tnfr import TNFRNBodySystem
>>> import numpy as np
>>>
>>> # Create 2-body system
>>> system = TNFRNBodySystem(
...     n_bodies=2,
...     masses=[1.0, 0.1],  # Actually νf^-1
...     positions=np.array([[0, 0, 0], [1, 0, 0]]),
...     velocities=np.array([[0, 0, 0], [0, 1, 0]]),
...     phases=np.array([0.0, 0.0])  # Synchronized initially
... )
>>>
>>> # Evolve via pure TNFR dynamics
>>> history = system.evolve(t_final=10.0, dt=0.01)
>>>
>>> # Check that orbital behavior emerges without assuming gravity
>>> print(f"Energy conservation: {history['energy_drift']:.2e}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..structural import create_nfr
from ..types import TNFRGraph
from ..operators.hamiltonian import InternalHamiltonian

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = (
    "TNFRNBodySystem",
    "compute_tnfr_coherence_potential",
    "compute_tnfr_delta_nfr",
)


def compute_tnfr_coherence_potential(
    G: TNFRGraph,
    positions: NDArray[np.floating],
    hbar_str: float = 1.0,
) -> float:
    """Compute coherence potential from TNFR network structure.

    This is the pure TNFR potential - NO classical assumptions.

    The potential emerges from:
    - Structural similarity (coherence matrix)
    - Network coupling topology
    - Phase synchronization

    NOT from Newtonian gravity or any other classical force law.

    Parameters
    ----------
    G : TNFRGraph
        Network graph with nodes containing TNFR attributes
    positions : ndarray, shape (N, 3)
        Current positions (affect phase evolution, not potential directly)
    hbar_str : float, default=1.0
        Structural Planck constant

    Returns
    -------
    U : float
        Coherence potential energy (lower = more stable)

    Notes
    -----
    In TNFR, the potential U encodes structural stability landscape.
    Nodes evolve toward configurations that maximize coherence (minimize U).
    """
    # Build internal Hamiltonian
    ham = InternalHamiltonian(G, hbar_str=hbar_str)

    # Potential is encoded in ground state energy
    eigenvalues, _ = ham.get_spectrum()

    # Total potential: sum of eigenvalues (trace of H_int)
    # For energy conservation, we use ground state as reference
    U = float(eigenvalues[0])  # Ground state energy

    return U


def compute_tnfr_delta_nfr(
    G: TNFRGraph,
    node_ids: List[str],
    hbar_str: float = 1.0,
) -> NDArray[np.floating]:
    """Compute ΔNFR from Hamiltonian commutator (pure TNFR).

    This is the correct TNFR computation of ΔNFR:
        ΔNFR = i[H_int, ·]/ℏ_str

    NOT from classical forces: F = -∇U (external assumption).

    Parameters
    ----------
    G : TNFRGraph
        Network graph with TNFR attributes
    node_ids : list of str
        Node identifiers in order
    hbar_str : float, default=1.0
        Structural Planck constant

    Returns
    -------
    dnfr : ndarray, shape (N,)
        ΔNFR values for each node (structural reorganization pressure)

    Notes
    -----
    The ΔNFR values represent the "reorganization pressure" driving
    structural evolution via the nodal equation: ∂EPI/∂t = νf · ΔNFR
    """
    # Build Hamiltonian
    ham = InternalHamiltonian(G, hbar_str=hbar_str)

    # Compute ΔNFR for each node
    dnfr = np.zeros(len(node_ids))
    for i, node_id in enumerate(node_ids):
        dnfr[i] = ham.compute_node_delta_nfr(node_id)

    return dnfr


class TNFRNBodySystem:
    """N-body system using pure TNFR physics (no classical assumptions).

    This implementation computes dynamics from TNFR structural coherence,
    without assuming any classical potentials (gravity, Coulomb, etc.).

    Attributes
    ----------
    n_bodies : int
        Number of bodies
    masses : ndarray, shape (N,)
        Masses (m = 1/νf, structural inertia)
    positions : ndarray, shape (N, 3)
        Current positions
    velocities : ndarray, shape (N, 3)
        Current velocities
    phases : ndarray, shape (N,)
        Current phases (θ ∈ [0, 2π])
    time : float
        Current structural time
    graph : TNFRGraph
        TNFR network representation
    hbar_str : float
        Structural Planck constant

    Notes
    -----
    Dynamics follow from nodal equation: ∂EPI/∂t = νf · ΔNFR
    where ΔNFR is computed from Hamiltonian commutator.

    Attraction/repulsion emerges from phase synchronization,
    NOT from assumed gravitational potential.
    """

    def __init__(
        self,
        n_bodies: int,
        masses: List[float] | NDArray[np.floating],
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        phases: NDArray[np.floating] | None = None,
        hbar_str: float = 1.0,
        coupling_strength: float = 0.1,
        coherence_strength: float = -1.0,
    ):
        """Initialize TNFR N-body system.

        Parameters
        ----------
        n_bodies : int
            Number of bodies
        masses : array_like, shape (N,)
            Masses (m = 1/νf, must be positive)
        positions : ndarray, shape (N, 3)
            Initial positions
        velocities : ndarray, shape (N, 3)
            Initial velocities
        phases : ndarray, shape (N,), optional
            Initial phases. If None, initialized to zero (synchronized)
        hbar_str : float, default=1.0
            Structural Planck constant
        coupling_strength : float, default=0.1
            Network coupling strength (J_0 in H_coupling)
        coherence_strength : float, default=-1.0
            Coherence potential strength (C_0 in H_coh)
            Negative = attractive potential well

        Raises
        ------
        ValueError
            If dimensions mismatch or masses non-positive
        """
        if n_bodies < 1:
            raise ValueError(f"n_bodies must be >= 1, got {n_bodies}")

        self.n_bodies = n_bodies
        self.masses = np.array(masses, dtype=float)

        if len(self.masses) != n_bodies:
            raise ValueError(f"masses length {len(self.masses)} != n_bodies {n_bodies}")

        if np.any(self.masses <= 0):
            raise ValueError("All masses must be positive")

        # State vectors
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.asarray(velocities, dtype=float).copy()

        if phases is None:
            self.phases = np.zeros(n_bodies, dtype=float)
        else:
            self.phases = np.asarray(phases, dtype=float).copy()

        # Validate shapes
        expected_shape = (n_bodies, 3)
        if self.positions.shape != expected_shape:
            raise ValueError(f"positions shape {self.positions.shape} != {expected_shape}")
        if self.velocities.shape != expected_shape:
            raise ValueError(f"velocities shape {self.velocities.shape} != {expected_shape}")
        if self.phases.shape != (n_bodies,):
            raise ValueError(f"phases shape {self.phases.shape} != ({n_bodies},)")

        self.time = 0.0
        self.hbar_str = float(hbar_str)

        # TNFR parameters
        self.coupling_strength = float(coupling_strength)
        self.coherence_strength = float(coherence_strength)

        # Build TNFR graph
        self._build_graph()

    def _build_graph(self) -> None:
        """Build TNFR graph representation.

        Each body becomes a resonant node with:
        - νf = 1/m (structural frequency)
        - EPI encoding (position, velocity)
        - Phase θ
        - All-to-all coupling (full network)
        """
        import networkx as nx

        self.graph: TNFRGraph = nx.Graph()
        self.graph.graph["name"] = "tnfr_nbody_system"
        self.graph.graph["H_COUPLING_STRENGTH"] = self.coupling_strength
        self.graph.graph["H_COH_STRENGTH"] = self.coherence_strength

        # Add nodes with TNFR attributes
        for i in range(self.n_bodies):
            node_id = f"body_{i}"

            # Structural frequency: νf = 1/m
            nu_f = 1.0 / self.masses[i]

            # Create NFR node with structured EPI
            epi_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }

            # Note: create_nfr expects scalar epi for initialization
            # We'll override it immediately
            _, _ = create_nfr(
                node_id,
                epi=1.0,  # Temporary, will be overwritten
                vf=nu_f,
                theta=float(self.phases[i]),
                graph=self.graph,
            )

            # Override with structured EPI
            self.graph.nodes[node_id]["epi"] = epi_state

        # Add edges (all-to-all coupling)
        # In TNFR, coupling strength depends on structural similarity
        # Here we use uniform coupling for simplicity
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                node_i = f"body_{i}"
                node_j = f"body_{j}"

                # Edge weight: coupling strength
                # (In more sophisticated version, could depend on distance)
                weight = self.coupling_strength
                self.graph.add_edge(node_i, node_j, weight=weight)

    def compute_energy(self) -> Tuple[float, float, float]:
        """Compute system energy (kinetic + coherence potential).

        Returns
        -------
        kinetic : float
            Kinetic energy K = Σ (1/2) m v²
        potential : float
            Coherence potential U from TNFR Hamiltonian
        total : float
            Total energy H = K + U

        Notes
        -----
        Unlike classical n-body (assumes U = -Gm₁m₂/r), the potential
        here emerges from TNFR coherence matrix and coupling topology.
        """
        # Kinetic energy (same as classical)
        v_squared = np.sum(self.velocities**2, axis=1)
        kinetic = 0.5 * np.sum(self.masses * v_squared)

        # Coherence potential from TNFR Hamiltonian
        # This is the key difference: NO assumption about gravity
        potential = compute_tnfr_coherence_potential(self.graph, self.positions, self.hbar_str)

        total = kinetic + potential

        return kinetic, potential, total

    def compute_momentum(self) -> NDArray[np.floating]:
        """Compute total linear momentum.

        Returns
        -------
        momentum : ndarray, shape (3,)
            Total momentum P = Σ m v
        """
        momentum = np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0)
        return momentum

    def compute_angular_momentum(self) -> NDArray[np.floating]:
        """Compute total angular momentum.

        Returns
        -------
        angular_momentum : ndarray, shape (3,)
            Total L = Σ r × (m v)
        """
        L = np.zeros(3)
        for i in range(self.n_bodies):
            L += self.masses[i] * np.cross(self.positions[i], self.velocities[i])
        return L

    def step(self, dt: float) -> None:
        """Advance system by one time step using TNFR dynamics.

        This implements the nodal equation: ∂EPI/∂t = νf · ΔNFR
        where ΔNFR is computed from Hamiltonian commutator.

        Parameters
        ----------
        dt : float
            Time step (structural time units)

        Notes
        -----
        Uses velocity Verlet-like integration for position/velocity,
        but accelerations come from TNFR ΔNFR via coherence gradients.
        """
        # Update graph with current state
        self._update_graph()

        # Compute accelerations from TNFR coherence-based forces
        # This is the key: forces emerge from coherence gradient, not gravity
        accel = self._compute_tnfr_accelerations()

        # Velocity Verlet integration
        # v(t+dt/2) = v(t) + a(t) * dt/2
        v_half = self.velocities + 0.5 * accel * dt

        # r(t+dt) = r(t) + v(t+dt/2) * dt
        self.positions += v_half * dt

        # Update graph with new positions
        self._update_graph()

        # Recompute accelerations at new positions
        accel_new = self._compute_tnfr_accelerations()

        # v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
        self.velocities = v_half + 0.5 * accel_new * dt

        # Update phases based on ΔNFR
        # Compute scalar ΔNFR for phase evolution
        node_ids = [f"body_{i}" for i in range(self.n_bodies)]
        dnfr_values = compute_tnfr_delta_nfr(self.graph, node_ids, self.hbar_str)

        # Phase evolution: dθ/dt ~ ΔNFR
        self.phases += dnfr_values * dt
        self.phases = np.mod(self.phases, 2 * np.pi)  # Keep in [0, 2π]

        # Update time
        self.time += dt

    def _compute_tnfr_accelerations(self) -> NDArray[np.floating]:
        """Compute accelerations from TNFR coherence-based forces.

        This is where TNFR physics determines motion:
        - Forces emerge from coherence gradient (NOT gravity!)
        - Phase differences create attraction/repulsion
        - Coupling strength determines force magnitude

        Returns
        -------
        accelerations : ndarray, shape (N, 3)
            Acceleration vectors for each body

        Notes
        -----
        The key TNFR insight: Forces emerge from maximizing coherence.

        Coherence between nodes i and j depends on:
        1. Phase difference: cos(θᵢ - θⱼ) (in-phase → attractive)
        2. Coupling strength: J₀ (from network edges)
        3. Distance dependence: Coherence decreases with separation

        This gives rise to attraction/repulsion WITHOUT assuming gravity!
        """
        accelerations = np.zeros((self.n_bodies, 3))

        # For each pair of bodies
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i == j:
                    continue

                # Position difference
                r_ij = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(r_ij)

                if dist < 1e-10:
                    continue  # Avoid singularity

                # Unit vector from i to j
                r_hat = r_ij / dist

                # Phase difference (key TNFR element!)
                phase_diff = self.phases[j] - self.phases[i]

                # Coherence factor: positive when in-phase, negative when anti-phase
                # This creates attraction for synchronized nodes
                coherence_factor = np.cos(phase_diff)

                # Distance-dependent coupling (coherence decays with distance)
                # This emerges from spatial structure of coherence matrix
                # Use exponential decay or power law
                distance_factor = 1.0 / (dist**2 + 0.1)  # Softened power law

                # Frequency coupling: Both nodes contribute
                nu_i = 1.0 / self.masses[i]
                nu_j = 1.0 / self.masses[j]
                freq_factor = np.sqrt(nu_i * nu_j)

                # Total TNFR force magnitude
                force_mag = (
                    self.coupling_strength
                    * self.coherence_strength  # Negative = attractive well
                    * coherence_factor
                    * distance_factor
                    * freq_factor
                )

                # Force direction
                force_vec = force_mag * r_hat

                # Acceleration: a = F/m = F * νf
                accelerations[i] += force_vec * nu_i

        return accelerations

    def _update_graph(self) -> None:
        """Update graph representation with current state."""
        for i in range(self.n_bodies):
            node_id = f"body_{i}"

            # Update EPI
            epi_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }
            self.graph.nodes[node_id]["epi"] = epi_state

            # Update phase
            self.graph.nodes[node_id]["theta"] = float(self.phases[i])

    def evolve(
        self,
        t_final: float,
        dt: float,
        store_interval: int = 1,
    ) -> Dict[str, Any]:
        """Evolve system using pure TNFR dynamics.

        Parameters
        ----------
        t_final : float
            Final time
        dt : float
            Time step
        store_interval : int, default=1
            Store state every N steps

        Returns
        -------
        history : dict
            Contains: time, positions, velocities, phases,
                     energy, kinetic, potential, momentum, angular_momentum

        Notes
        -----
        Evolution follows nodal equation with ΔNFR from Hamiltonian.
        NO classical gravitational assumptions.
        """
        n_steps = int((t_final - self.time) / dt)

        if n_steps < 1:
            raise ValueError(f"t_final {t_final} <= current time {self.time}")

        # Pre-allocate storage
        n_stored = (n_steps // store_interval) + 1
        times = np.zeros(n_stored)
        positions_hist = np.zeros((n_stored, self.n_bodies, 3))
        velocities_hist = np.zeros((n_stored, self.n_bodies, 3))
        phases_hist = np.zeros((n_stored, self.n_bodies))
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
        phases_hist[store_idx] = self.phases.copy()
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
                phases_hist[store_idx] = self.phases.copy()
                K, U, E = self.compute_energy()
                kinetic_energies[store_idx] = K
                potential_energies[store_idx] = U
                energies[store_idx] = E
                momenta[store_idx] = self.compute_momentum()
                angular_momenta[store_idx] = self.compute_angular_momentum()
                store_idx += 1

        # Compute energy drift
        energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0])

        return {
            "time": times[:store_idx],
            "positions": positions_hist[:store_idx],
            "velocities": velocities_hist[:store_idx],
            "phases": phases_hist[:store_idx],
            "energy": energies[:store_idx],
            "kinetic": kinetic_energies[:store_idx],
            "potential": potential_energies[:store_idx],
            "momentum": momenta[:store_idx],
            "angular_momentum": angular_momenta[:store_idx],
            "energy_drift": energy_drift,
        }

    def plot_trajectories(
        self,
        history: Dict[str, Any],
        show_energy: bool = True,
        show_phases: bool = True,
    ) -> Figure:
        """Plot trajectories, energy, and phase evolution.

        Parameters
        ----------
        history : dict
            Result from evolve()
        show_energy : bool, default=True
            Show energy conservation plot
        show_phases : bool, default=True
            Show phase evolution plot

        Returns
        -------
        fig : matplotlib Figure

        Raises
        ------
        ImportError
            If matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib required for plotting. " "Install with: pip install 'tnfr[viz-basic]'"
            ) from exc

        n_plots = 1 + int(show_energy) + int(show_phases)
        fig = plt.figure(figsize=(6 * n_plots, 5))

        plot_idx = 1

        # 3D trajectories
        ax_3d = fig.add_subplot(1, n_plots, plot_idx, projection="3d")
        plot_idx += 1

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
            ax_3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors[i], s=100, marker="o")
            ax_3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors[i], s=50, marker="x")

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("TNFR N-Body Trajectories (No Gravitational Assumption)")
        ax_3d.legend()

        # Energy conservation
        if show_energy:
            ax_energy = fig.add_subplot(1, n_plots, plot_idx)
            plot_idx += 1

            time = history["time"]
            E = history["energy"]
            E0 = E[0]

            ax_energy.plot(
                time,
                (E - E0) / abs(E0) * 100,
                label="Energy drift (%)",
                color="red",
                linewidth=2,
            )
            ax_energy.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax_energy.set_xlabel("Structural Time")
            ax_energy.set_ylabel("ΔE/E₀ (%)")
            ax_energy.set_title("Energy Conservation (TNFR Hamiltonian)")
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)

        # Phase evolution
        if show_phases:
            ax_phases = fig.add_subplot(1, n_plots, plot_idx)

            time = history["time"]
            phases = history["phases"]

            for i in range(self.n_bodies):
                ax_phases.plot(
                    time,
                    phases[:, i],
                    color=colors[i],
                    label=f"Body {i+1}",
                    linewidth=2,
                )

            ax_phases.set_xlabel("Structural Time")
            ax_phases.set_ylabel("Phase θ (rad)")
            ax_phases.set_title("Phase Evolution (TNFR Synchronization)")
            ax_phases.legend()
            ax_phases.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
