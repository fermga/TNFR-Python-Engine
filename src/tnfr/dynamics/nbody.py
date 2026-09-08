"""Newtonian N-body solver with explicit TNFR-shaped adapter metadata.

The force law in this module is the externally assumed Newtonian potential
``U = -sum(G*m_i*m_j/r_ij)``. Velocity Verlet advances that classical model.
For compatibility, graph nodes store position/velocity dictionaries, record
the optional adapter assignment ``nu_f = 1/m``, and expose Newtonian
acceleration through the legacy name ``compute_gravitational_dnfr``.

The solver does not execute the scalar nodal equation and its acceleration is
not the graph pressure computed by :mod:`tnfr.dynamics.dnfr`. Agreement with
Newtonian orbits validates the stated force law and integrator only; it does not
derive gravity, mechanical energy or increasing structural C(t) from TNFR.

For a different declared auxiliary pair law, see
:mod:`tnfr.dynamics.nbody_tnfr`.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from ..constants.canonical import EPI_MAX_CANONICAL
from ..mathematics.unified_numerical import np
from ..structural import create_nfr
from ..types import TNFRGraph

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = (
    "NBodySystem",
    "gravitational_potential",
    "gravitational_force",
    "compute_newtonian_acceleration",
    "compute_gravitational_dnfr",
)


def _integration_schedule(
    current_time: float, t_final: float, dt: float
) -> tuple[int, float | None]:
    """Return step count and an optional short final step.

    The caller validates finite times and positive ``dt``.  Keeping the
    remainder explicit prevents truncation from ending before ``t_final``.
    """
    remaining = t_final - current_time
    if remaining <= 0.0:
        return 0, None
    ratio = remaining / dt
    if (
        not math.isfinite(ratio)
        or ratio >= float(np.iinfo(np.intp).max)
    ):
        raise ValueError("requested integration requires too many time steps")
    full_steps = int(math.floor(ratio))
    remainder = remaining - full_steps * dt
    roundoff = 16.0 * np.finfo(float).eps * max(
        1.0, abs(current_time), abs(t_final), abs(full_steps * dt)
    )
    if remainder <= roundoff:
        return full_steps, None
    return full_steps + 1, remainder


def _pair_direction_and_inverse_distance(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    extra_dimension: float = 0.0,
) -> tuple[NDArray[np.floating], float]:
    """Return a stable pair direction and inverse Euclidean distance.

    Scaling is used only when direct subtraction overflows.  It preserves a
    representable reciprocal even when the distance itself exceeds the float
    range (for example, points near opposite ends of that range).
    ``extra_dimension`` supplies the softening coordinate when needed.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = np.asarray(target, dtype=float) - np.asarray(
            source, dtype=float
        )
    distance = math.hypot(
        *(float(value) for value in displacement), float(extra_dimension)
    )
    if math.isfinite(distance):
        if distance == 0.0:
            return np.zeros_like(displacement), math.inf
        inverse_distance = 1.0 / distance
        return displacement * inverse_distance, inverse_distance

    scale = max(
        *(abs(float(value)) for value in source),
        *(abs(float(value)) for value in target),
        abs(float(extra_dimension)),
    )
    if scale == 0.0:
        return np.zeros_like(displacement), math.inf
    scaled_displacement = (
        np.asarray(target, dtype=float) / scale
        - np.asarray(source, dtype=float) / scale
    )
    scaled_extra = float(extra_dimension) / scale
    scaled_distance = math.hypot(
        *(float(value) for value in scaled_displacement), scaled_extra
    )
    if scaled_distance == 0.0 or not math.isfinite(scaled_distance):
        raise ValueError("pair separation cannot be represented reliably")
    inverse_distance = (1.0 / scale) / scaled_distance
    return scaled_displacement / scaled_distance, inverse_distance


def _finite_float(value: Any, name: str) -> float:
    """Normalize a finite scalar while rejecting boolean sentinels."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _finite_real_array(value: Any, name: str) -> NDArray[np.floating]:
    """Normalize a finite real array without discarding imaginary parts."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real array")
    array = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real array")
    return array


def _scaled_product(*factors: float) -> float:
    """Multiply finite floats without avoidable intermediate range loss."""
    sign = 1.0
    mantissa = 1.0
    exponent = 0
    for raw in factors:
        factor = float(raw)
        if not math.isfinite(factor):
            raise ValueError("interaction magnitude must remain finite")
        if factor == 0.0:
            return 0.0
        if factor < 0.0:
            sign = -sign
            factor = -factor
        part, part_exponent = math.frexp(factor)
        mantissa *= part
        exponent += part_exponent
        mantissa, adjustment = math.frexp(mantissa)
        exponent += adjustment
    try:
        result = math.ldexp(sign * mantissa, exponent)
    except OverflowError as exc:
        raise ValueError("interaction magnitude exceeds finite range") from exc
    if not math.isfinite(result):
        raise ValueError("interaction magnitude exceeds finite range")
    return result


def _kinetic_energy(
    masses: NDArray[np.floating], velocities: NDArray[np.floating]
) -> float:
    """Return positive kinetic energy without squaring before scaling."""
    terms = [
        _scaled_product(0.5, mass, component, component)
        for mass, velocity in zip(masses, velocities)
        for component in velocity
    ]
    try:
        return math.fsum(terms)
    except OverflowError as exc:
        raise ValueError("kinetic energy exceeds finite range") from exc


def _validate_newtonian_inputs(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float,
    softening: float,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    float,
    float,
]:
    """Return finite arrays and scalar parameters for the 3D adapter."""
    position_array = _finite_real_array(positions, "positions")
    mass_array = _finite_real_array(masses, "masses")
    gravitational_constant = _finite_float(G, "G")
    softening_length = _finite_float(softening, "softening")

    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if mass_array.ndim != 1 or len(mass_array) != len(position_array):
        raise ValueError("masses must have shape (N,) matching positions")
    if np.any(mass_array <= 0.0):
        raise ValueError("All masses must be finite and positive")
    if gravitational_constant < 0.0:
        raise ValueError("G must be finite and non-negative")
    if softening_length < 0.0:
        raise ValueError("softening must be finite and non-negative")

    return position_array, mass_array, gravitational_constant, softening_length


def gravitational_potential(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> float:
    """Compute total Newtonian gravitational potential energy.

    U(q) = -Σ_{i<j} G * m_i * m_j / |r_i - r_j|

    This is the externally supplied Newtonian potential used by this adapter.

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
    The negative sign is the conventional Newtonian choice of energy origin.
    """
    positions, masses, G, softening = _validate_newtonian_inputs(
        positions, masses, G, softening
    )
    N = len(positions)
    pair_potentials: list[float] = []

    for i in range(N):
        for j in range(i + 1, N):
            _, inverse_distance = _pair_direction_and_inverse_distance(
                positions[i], positions[j], softening
            )
            if math.isinf(inverse_distance):
                raise ValueError(
                    "coincident positions require strictly positive softening"
                )
            pair_potentials.append(
                -_scaled_product(G, masses[i], masses[j], inverse_distance)
            )

    try:
        return math.fsum(pair_potentials)
    except OverflowError as exc:
        raise ValueError("gravitational potential exceeds finite range") from exc


def gravitational_force(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> NDArray[np.floating]:
    """Compute gravitational forces on all bodies.

    F_i = -∇_i U = Σ_{j≠i} G * m_i * m_j * (r_j - r_i) / |r_j - r_i|³

    The force is the gradient of the externally supplied Newtonian potential.

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
    Newton's third-law pair symmetry follows from the symmetric potential.
    """
    positions, masses, G, softening = _validate_newtonian_inputs(
        positions, masses, G, softening
    )
    N = len(positions)
    forces = np.zeros_like(positions)

    for i in range(N):
        for j in range(i + 1, N):
            direction, inverse_distance = _pair_direction_and_inverse_distance(
                positions[i], positions[j], softening
            )
            if math.isinf(inverse_distance):
                raise ValueError(
                    "coincident positions require strictly positive softening"
                )
            magnitude = _scaled_product(
                G,
                masses[i],
                masses[j],
                inverse_distance,
                inverse_distance,
            )
            pair_force = magnitude * direction
            forces[i] += pair_force
            forces[j] -= pair_force

    if not np.all(np.isfinite(forces)):
        raise ValueError("gravitational force exceeds finite range")

    return forces


def compute_newtonian_acceleration(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> NDArray[np.floating]:
    """Return acceleration from the externally supplied Newtonian pair law.

    ``a_i = F_i / m_i``

    The returned array is used directly as acceleration by velocity Verlet. It
    is not the scalar graph pressure computed by ``tnfr.dynamics.dnfr`` and is
    not multiplied by ``nu_f`` in this solver.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Positions of N bodies
    masses : ndarray, shape (N,)
        Classical masses; the graph records ``nu_f=1/m`` as adapter metadata
    G : float, default=1.0
        Gravitational constant
    softening : float, default=0.0
        Softening length

    Returns
    -------
    accelerations : ndarray, shape (N, 3)
        Newtonian accelerations for each body

    Notes
    -----
    This is an auxiliary classical-model value. It is not canonical scalar
    graph pressure and cannot be used to compute structural ``C(t)`` without a
    separately declared mapping to the graph pressure and EPI-rate channels.
    """
    positions, masses, G, softening = _validate_newtonian_inputs(
        positions, masses, G, softening
    )
    forces = gravitational_force(positions, masses, G, softening)

    # Broadcast division: (N, 3) / (N, 1) -> (N, 3).
    with np.errstate(over="ignore", invalid="ignore"):
        accelerations = forces / masses[:, np.newaxis]
    if not np.all(np.isfinite(accelerations)):
        raise ValueError("gravitational acceleration exceeds finite range")

    return accelerations


def compute_gravitational_dnfr(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    G: float = 1.0,
    softening: float = 0.0,
) -> NDArray[np.floating]:
    """Compatibility alias for :func:`compute_newtonian_acceleration`.

    The historical name does not make the returned vector a canonical TNFR
    pressure. New code should use the model-specific acceleration name.
    """

    return compute_newtonian_acceleration(positions, masses, G, softening)


class NBodySystem:
    """Newtonian N-body solver with TNFR-shaped graph metadata.

    Implements N particles coupled through the externally specified Newtonian
    potential. Graph nodes retain adapter-shaped position/velocity metadata and
    the optional assignment ``nu_f=1/m``; evolution follows velocity Verlet.

    Attributes
    ----------
    n_bodies : int
        Number of bodies in the system
    masses : ndarray, shape (N,)
        Masses of bodies. This adapter declares the embedding ``νf_i=1/m_i``;
        the identity is not a general consequence of the nodal equation.
    G : float
        Gravitational constant
    softening : float
        Softening length for numerical stability
    positions : ndarray, shape (N, 3)
        Current positions
    velocities : ndarray, shape (N, 3)
        Current velocities
    time : float
        Current time in the declared Newtonian adapter units
    graph : TNFRGraph
        Adapter graph storing NFR-shaped nodes and explicit scope metadata

    Notes
    -----
    Position/velocity dictionaries and ``nu_f=1/m`` are adapter metadata. They
    do not constitute a canonical scalar-EPI operator trajectory. Continuous
    Newtonian invariants follow from the central potential; the discrete solver
    reports their numerical drift.
    """

    def __init__(
        self,
        n_bodies: int,
        masses: list[float] | NDArray[np.floating],
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
        if (
            isinstance(n_bodies, bool)
            or not isinstance(n_bodies, Integral)
            or n_bodies < 1
        ):
            raise ValueError(f"n_bodies must be >= 1, got {n_bodies}")

        self.n_bodies = int(n_bodies)
        self.masses = _finite_real_array(masses, "masses")

        if self.masses.ndim != 1 or self.masses.shape != (n_bodies,):
            raise ValueError(
                f"masses shape {self.masses.shape} != ({n_bodies},)"
            )

        if np.any(self.masses <= 0):
            raise ValueError("All masses must be finite and positive")

        self.G = _finite_float(G, "G")
        self.softening = _finite_float(softening, "softening")
        if self.G < 0.0:
            raise ValueError("G must be finite and non-negative")
        if self.softening < 0.0:
            raise ValueError("softening must be finite and non-negative")

        # State vectors
        self.positions: NDArray[np.floating] = np.zeros((n_bodies, 3), dtype=float)
        self.velocities: NDArray[np.floating] = np.zeros((n_bodies, 3), dtype=float)
        self.time = 0.0

        # Create TNFR graph representation
        self._build_graph()

    def _build_graph(self) -> None:
        """Build TNFR graph representation of the N-body system.

        Each body becomes a resonant node with:
        - νf = 1/m under this adapter's explicit classical embedding
        - EPI encoding (position, velocity)
        - Phase initialized to 0 (can be set for rotation)
        - Fully connected metadata for the externally computed pair law
        """
        # Create empty graph (will add nodes manually)
        import networkx as nx

        self.graph: TNFRGraph = nx.Graph()
        self.graph.graph["name"] = "nbody_system"
        self.graph.graph.update(
            {
                "MODEL_SCOPE": "external_newtonian_nbody_adapter",
                "DYNAMICS_LAW": "velocity_verlet_newtonian_pair_force",
                "NU_F_SEMANTICS": "inverse_mass_adapter_metadata_only",
                "FORCE_BRIDGE_MATERIALIZED": False,
                "ACCELERATION_API_COMPATIBILITY": (
                    "compute_gravitational_dnfr_is_legacy_name"
                ),
                "CANONICAL_C_T_AVAILABLE": False,
            }
        )

        # Canonical EPI seed stays below the EPI_MAX = 1.0 validation bound.
        epi_seed = min(0.5, EPI_MAX_CANONICAL * 0.95)

        # Add nodes with TNFR attributes
        for i in range(self.n_bodies):
            node_id = f"body_{i}"

            # Adapter convention for this classical embedding. In the bare
            # first-order nodal equation νf has mobility semantics.
            nu_f = 1.0 / float(self.masses[i])
            if not math.isfinite(nu_f) or nu_f <= 0.0:
                raise ValueError(
                    "inverse-mass adapter frequency is not representable"
                )

            # Create NFR node
            _, _ = create_nfr(
                node_id,
                epi=epi_seed,  # Will be overwritten by set_state
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
                # Metadata coefficient for the external Newtonian pair law.
                weight = _scaled_product(self.G, self.masses[i], self.masses[j])
                self.graph.add_edge(node_i, node_j, weight=weight)

    def set_state(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> None:
        """set system state (positions and velocities).

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
        positions = _finite_real_array(positions, "positions")
        velocities = _finite_real_array(velocities, "velocities")

        expected_shape = (self.n_bodies, 3)
        if positions.shape != expected_shape:
            raise ValueError(f"positions shape {positions.shape} != {expected_shape}")
        if velocities.shape != expected_shape:
            raise ValueError(
                f"velocities shape {velocities.shape} != {expected_shape}"
            )
        self.positions = positions.copy()
        self.velocities = velocities.copy()

        # Update EPI in graph nodes
        # EPI encodes state as dictionary with position/velocity
        for i in range(self.n_bodies):
            node_id = f"body_{i}"
            # Store the adapter payload under the legacy EPI key. Canonical
            # scalar-EPI operators are not applied to this graph.
            epi_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }
            self.graph.nodes[node_id]["epi"] = epi_state

    def get_state(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get current state (positions and velocities).

        Returns
        -------
        positions : ndarray, shape (N, 3)
            Current positions
        velocities : ndarray, shape (N, 3)
            Current velocities
        """
        return self.positions.copy(), self.velocities.copy()

    def compute_energy(self) -> tuple[float, float, float]:
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
        The continuous declared Newtonian model conserves energy. Reported
        drift measures the finite-step integrator error.
        """
        # Kinetic energy: T = Σ (1/2) m_i v_i²
        kinetic = _kinetic_energy(self.masses, self.velocities)

        # Potential energy: U = -Σ_{i<j} G m_i m_j / r_ij
        potential = gravitational_potential(
            self.positions, self.masses, self.G, self.softening
        )

        try:
            total = math.fsum((kinetic, potential))
        except OverflowError as exc:
            raise ValueError("total energy exceeds finite range") from exc

        return kinetic, potential, total

    def compute_momentum(self) -> NDArray[np.floating]:
        """Compute total linear momentum.

        Returns
        -------
        momentum : ndarray, shape (3,)
            Total momentum P = Σ m_i v_i

        Notes
        -----
        The continuous declared Newtonian central-force model conserves
        total linear momentum; a finite trajectory can show numerical drift.
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
        The continuous declared Newtonian central-force model conserves
        angular momentum; a finite trajectory can show numerical drift.
        """
        L = np.zeros(3)
        for i in range(self.n_bodies):
            L += self.masses[i] * np.cross(self.positions[i], self.velocities[i])
        return L

    def step(self, dt: float) -> None:
        """Advance system by one time step using velocity Verlet.

        Velocity Verlet is symplectic for this autonomous separable
        Hamiltonian model. Its finite-step energy drift depends on the timestep.

        Algorithm:
        1. r(t+dt) = r(t) + v(t)*dt + (1/2)*a(t)*dt²
        2. a(t+dt) = compute acceleration at new positions
        3. v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt

        Parameters
        ----------
        dt : float
            Time step in the Newtonian adapter's units

        Notes
        -----
        This second-order update is a classical adapter. It is not equivalent
        to the first-order nodal equation merely because the graph also stores
        ``nu_f=1/m``.
        """
        dt = _finite_float(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")

        # Compute Newtonian acceleration at the current time.
        accel_t = compute_newtonian_acceleration(
            self.positions, self.masses, self.G, self.softening
        )

        # Update positions: r(t+dt) = r(t) + v(t)*dt + (1/2)*a(t)*dt²
        new_positions = (
            self.positions + self.velocities * dt + 0.5 * accel_t * dt**2
        )

        # Compute acceleration at new time: a(t+dt)
        accel_t_plus_dt = compute_newtonian_acceleration(
            new_positions, self.masses, self.G, self.softening
        )

        # Update velocities: v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt
        new_velocities = (
            self.velocities + 0.5 * (accel_t + accel_t_plus_dt) * dt
        )
        if not np.all(np.isfinite(new_velocities)):
            raise ValueError("time step produced non-finite velocities")
        new_time = self.time + dt
        if not math.isfinite(new_time):
            raise ValueError("time step produced non-finite time")

        self.positions = new_positions
        self.velocities = new_velocities

        # Update time in the Newtonian adapter units.
        self.time = new_time

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
    ) -> dict[str, Any]:
        """Evolve system from current time to t_final.

        Parameters
        ----------
        t_final : float
            Final adapter time
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
        The evolution applies velocity Verlet to the Newtonian force law.
        Classical invariant diagnostics are tracked for numerical validation.
        """
        t_final = _finite_float(t_final, "t_final")
        dt = _finite_float(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        if (
            isinstance(store_interval, bool)
            or not isinstance(store_interval, Integral)
            or store_interval <= 0
        ):
            raise ValueError("store_interval must be a strictly positive integer")
        store_interval = int(store_interval)

        n_steps, final_step = _integration_schedule(self.time, t_final, dt)

        if n_steps < 1:
            raise ValueError(f"t_final {t_final} <= current time {self.time}")

        # Pre-allocate storage
        n_stored = 1 + (n_steps // store_interval) + int(
            n_steps % store_interval != 0
        )
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
            step_dt = (
                final_step
                if final_step is not None and step + 1 == n_steps
                else dt
            )
            self.step(step_dt)

            # Store state if needed
            should_store = (step + 1) % store_interval == 0 or step + 1 == n_steps
            if should_store and store_idx < n_stored:
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

        self.time = float(t_final)
        times[store_idx - 1] = self.time

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
        history: dict[str, Any],
        ax: Any | None = None,
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
                label=f"Body {i + 1} (m={self.masses[i]:.2f})",
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
        ax_3d.set_title("Newtonian N-Body Adapter Trajectories")
        ax_3d.legend()

        if show_energy:
            # Plot energy conservation
            time = history["time"]
            E = history["energy"]
            E0 = E[0]
            energy_scale = max(abs(float(E0)), float(np.finfo(float).eps))

            ax_energy.plot(
                time,
                (E - E0) / energy_scale * 100,
                label="Relative energy error (%)",
                color="red",
            )
            ax_energy.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax_energy.set_xlabel("Adapter Time")
            ax_energy.set_ylabel("ΔE/E₀ (%)")
            ax_energy.set_title("Newtonian Energy Drift")
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
