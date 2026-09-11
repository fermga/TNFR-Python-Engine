"""Experimental phase-coupled N-body adapter expressed with TNFR variables.

The adapter declares a second-order pair-interaction model and mirrors its state
alongside a TNFR graph.  It is useful for controlled comparisons with classical
N-body solvers, but it is not a derivation of mechanics from the first-order
nodal equation and it does not execute the canonical operator grammar.

The declared embedding is

``nu_f[i] = 1 / mass[i]``

and the pair force on body ``i`` from body ``j`` is

``F_ij = A_ij r_hat_ij / (r_ij**2 + epsilon)``

with

``A_ij = -C_0 w_ij cos(theta_j - theta_i) sqrt(nu_i nu_j)``.

Here ``C_0`` is ``coherence_strength``, ``w_ij`` is the graph edge weight and
``epsilon`` is an explicitly selected squared-distance regularizer.  With the
default negative ``C_0``, synchronized phases attract and antiphase nodes repel.
The matching position-dependent pair potential is implemented alongside the
force so its energy diagnostic has a well-defined meaning.

The module also retains two historical Hamiltonian-named read-outs for API
compatibility.  The ground-state read-out of :class:`InternalHamiltonian` does
not depend explicitly on body positions.  The diagonal localized-projector
commutator is identically zero and therefore cannot provide an autonomous phase
law.  Phases remain fixed during this adapter's evolution.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from ..alias import get_attr, get_theta_attr
from ..constants.canonical import EPI_MAX_CANONICAL
from ..constants.aliases import ALIAS_VF
from ..errors.contextual import NetworkConfigError
from ..mathematics.unified_numerical import np
from ..operators.hamiltonian import InternalHamiltonian
from ..structural import create_nfr
from ..types import TNFRGraph
from ..utils.cache import cached_node_list
from .nbody import (
    _integration_schedule,
    _kinetic_energy,
    _pair_direction_and_inverse_distance,
    _scaled_product,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = (
    "NBODY_DISTANCE_REGULARIZATION_DEFAULT",
    "TNFRNBodySystem",
    "compute_internal_hamiltonian_ground_state",
    "compute_nbody_pair_forces",
    "compute_nbody_pair_potential",
    "compute_node_projector_commutator_rates",
    "compute_tnfr_coherence_potential",
    "compute_tnfr_delta_nfr",
)


# Selected adapter parameter in squared position units.  It regularizes the
# pair law at short range and is not a canonical TNFR constant.
NBODY_DISTANCE_REGULARIZATION_DEFAULT = 0.1


def _finite_scalar(value: Any, parameter: str) -> float:
    """Normalize one finite real adapter parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise NetworkConfigError(
            parameter=parameter,
            value=value,
            reason="Must be a finite real number",
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NetworkConfigError(
            parameter=parameter,
            value="[not a finite real number]",
            reason="Must be a finite real number",
        ) from exc
    if not math.isfinite(normalized):
        raise NetworkConfigError(
            parameter=parameter,
            value=value,
            reason="Must be a finite real number",
        )
    return normalized


def _validate_positions(
    G: TNFRGraph,
    positions: NDArray[np.floating],
) -> tuple[list[Any], NDArray[np.floating]]:
    """Return graph node order and a finite position matrix."""
    nodes = list(cached_node_list(G))
    try:
        position_array = np.asarray(positions, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NetworkConfigError(
            parameter="positions",
            value="[not a finite real matrix]",
            reason="Positions must be a finite real matrix",
        ) from exc
    if position_array.ndim != 2 or position_array.shape[0] != len(nodes):
        raise NetworkConfigError(
            parameter="positions",
            value=str(position_array.shape),
            reason=f"Expected one position row for each of {len(nodes)} graph nodes",
        )
    if position_array.shape[1] < 1:
        raise NetworkConfigError(
            parameter="positions",
            value=str(position_array.shape),
            reason="Position vectors must have at least one component",
        )
    if not np.all(np.isfinite(position_array)):
        raise NetworkConfigError(
            parameter="positions",
            value="[contains non-finite values]",
            reason="All positions must be finite",
        )
    return nodes, position_array


def _validate_distance_regularization(value: float) -> float:
    regularization = _finite_scalar(value, "distance_regularization")
    if regularization <= 0.0:
        raise NetworkConfigError(
            parameter="distance_regularization",
            value=value,
            reason="Must be finite and strictly positive",
        )
    return regularization


def _pair_amplitudes(G: TNFRGraph, nodes: list[Any]) -> NDArray[np.floating]:
    """Build the symmetric amplitude matrix for the declared pair law."""
    if G.is_directed():
        raise NetworkConfigError(
            parameter="graph",
            value="directed",
            reason="The N-body pair adapter requires an undirected graph",
        )

    node_to_index = {node: index for index, node in enumerate(nodes)}
    try:
        frequencies = np.asarray(
            [float(get_attr(G.nodes[node], ALIAS_VF, 0.0)) for node in nodes],
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise NetworkConfigError(
            parameter="nu_f",
            value="[contains non-real values]",
            reason="Pair-law structural frequencies must be finite and non-negative",
        ) from exc
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise NetworkConfigError(
            parameter="nu_f",
            value="[contains invalid values]",
            reason="Pair-law structural frequencies must be finite and non-negative",
        )
    try:
        phases = np.asarray(
            [float(get_theta_attr(G.nodes[node], 0.0)) for node in nodes],
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise NetworkConfigError(
            parameter="phases",
            value="[contains non-real values]",
            reason="Pair-law phases must be finite",
        ) from exc
    if not np.all(np.isfinite(phases)):
        raise NetworkConfigError(
            parameter="phases",
            value="[contains non-finite values]",
            reason="Pair-law phases must be finite",
        )

    phases = np.remainder(phases, math.tau)

    default_weight = _finite_scalar(
        G.graph.get("H_COUPLING_STRENGTH", 0.1), "coupling_strength"
    )
    edge_weights = np.zeros((len(nodes), len(nodes)), dtype=float)
    for source, target, data in G.edges(data=True):
        i = node_to_index[source]
        j = node_to_index[target]
        weight = _finite_scalar(data.get("weight", default_weight), "edge weight")
        edge_weights[i, j] += weight
        edge_weights[j, i] += weight

    coherence_strength = _finite_scalar(
        G.graph.get("H_COH_STRENGTH", -1.0), "coherence_strength"
    )
    phase_difference = phases[None, :] - phases[:, None]
    root_frequencies = np.sqrt(frequencies)
    frequency_factor = root_frequencies[:, None] * root_frequencies[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        amplitudes = (
            -coherence_strength
            * edge_weights
            * np.cos(phase_difference)
            * frequency_factor
        )
    if not np.all(np.isfinite(amplitudes)):
        raise NetworkConfigError(
            parameter="pair amplitude",
            value="[outside finite range]",
            reason="Pair-law amplitudes must remain finite",
        )
    np.fill_diagonal(amplitudes, 0.0)
    return amplitudes


def compute_nbody_pair_forces(
    G: TNFRGraph,
    positions: NDArray[np.floating],
    distance_regularization: float = NBODY_DISTANCE_REGULARIZATION_DEFAULT,
) -> NDArray[np.floating]:
    r"""Compute forces for the adapter's declared phase-coupled pair law.

    Rows in ``positions`` follow :func:`cached_node_list` order.  The returned
    force is central and antisymmetric for every undirected edge, so its sum is
    zero up to floating-point roundoff.  This is an auxiliary N-body law, not a
    canonical ``DeltaNFR`` computation.
    """
    nodes, position_array = _validate_positions(G, positions)
    regularization = _validate_distance_regularization(distance_regularization)
    amplitudes = _pair_amplitudes(G, nodes)

    root_regularization = math.sqrt(regularization)
    forces = np.zeros_like(position_array)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            amplitude = float(amplitudes[i, j])
            if amplitude == 0.0:
                continue
            direction, _ = _pair_direction_and_inverse_distance(
                position_array[i], position_array[j]
            )
            _, inverse_effective_distance = _pair_direction_and_inverse_distance(
                position_array[i], position_array[j], root_regularization
            )
            try:
                magnitude = _scaled_product(
                    amplitude,
                    inverse_effective_distance,
                    inverse_effective_distance,
                )
            except ValueError as exc:
                raise NetworkConfigError(
                    parameter="pair force",
                    value="[outside finite range]",
                    reason="Pair-law force must remain finite",
                ) from exc
            pair_force = magnitude * direction
            if not np.all(np.isfinite(pair_force)):
                raise NetworkConfigError(
                    parameter="pair force",
                    value="[outside finite range]",
                    reason="Pair-law force must remain finite",
                )
            forces[i] += pair_force
            forces[j] -= pair_force
    if not np.all(np.isfinite(forces)):
        raise NetworkConfigError(
            parameter="pair force",
            value="[outside finite range]",
            reason="Pair-law force must remain finite",
        )
    return forces


def compute_nbody_pair_potential(
    G: TNFRGraph,
    positions: NDArray[np.floating],
    distance_regularization: float = NBODY_DISTANCE_REGULARIZATION_DEFAULT,
) -> float:
    r"""Return the position-dependent potential matching the pair force.

    For pair amplitude ``A`` and separation ``r``, the potential with zero at
    infinite separation is

    ``U_ij = A/sqrt(epsilon) * (atan(r/sqrt(epsilon)) - pi/2)``.

    Differentiating gives ``F_ij = A r_hat/(r**2 + epsilon)``.
    """
    nodes, position_array = _validate_positions(G, positions)
    regularization = _validate_distance_regularization(distance_regularization)
    amplitudes = _pair_amplitudes(G, nodes)
    root_regularization = float(np.sqrt(regularization))

    pair_potentials: list[float] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            amplitude = float(amplitudes[i, j])
            if amplitude == 0.0:
                continue
            _, inverse_distance = _pair_direction_and_inverse_distance(
                position_array[i], position_array[j]
            )
            angle = (
                math.pi / 2.0
                if math.isinf(inverse_distance)
                else math.atan(root_regularization * inverse_distance)
            )
            try:
                contribution = -_scaled_product(
                    amplitude, angle, 1.0 / root_regularization
                )
            except ValueError as exc:
                raise NetworkConfigError(
                    parameter="pair potential",
                    value="[outside finite range]",
                    reason="Pair-law potential must remain finite",
                ) from exc
            if not math.isfinite(contribution):
                raise NetworkConfigError(
                    parameter="pair potential",
                    value="[outside finite range]",
                    reason="Pair-law potential must remain finite",
                )
            pair_potentials.append(contribution)
    try:
        return math.fsum(pair_potentials)
    except OverflowError as exc:
        raise NetworkConfigError(
            parameter="pair potential",
            value="[outside finite range]",
            reason="Pair-law potential must remain finite",
        ) from exc


def compute_internal_hamiltonian_ground_state(
    G: TNFRGraph,
    hbar_str: float = 1.0,
) -> float:
    """Return the auxiliary internal-Hamiltonian ground-state read-out.

    This spectral quantity is computed from graph attributes and topology.  It
    is not the potential whose gradient drives body positions in this module.
    ``hbar_str`` is retained by :class:`InternalHamiltonian` for its evolution
    APIs but does not enter the matrix spectrum itself.
    """
    hbar_str = _finite_scalar(hbar_str, "hbar_str")
    if hbar_str == 0.0:
        raise NetworkConfigError(
            parameter="hbar_str",
            value=hbar_str,
            reason="Must be finite and non-zero",
        )
    if len(G) == 0:
        return 0.0
    eigenvalues, _ = InternalHamiltonian(G, hbar_str=hbar_str).get_spectrum()
    return float(eigenvalues[0])


def compute_tnfr_coherence_potential(
    G: TNFRGraph,
    positions: NDArray[np.floating],
    hbar_str: float = 1.0,
) -> float:
    """Return the legacy-named Hamiltonian ground-state read-out.

    This function retains its public name and call signature for compatibility.
    ``positions`` is validated for row alignment but does not enter the
    :class:`InternalHamiltonian` construction.  Use
    :func:`compute_nbody_pair_potential` for the position-dependent potential
    matching this module's force law.

    Parameters
    ----------
    G : TNFRGraph
        Network graph with nodes containing TNFR attributes
    positions : ndarray, shape (N, 3)
        Position rows in graph-node order. Validated but otherwise unused.
    hbar_str : float, default=1.0
        Compatibility scale passed to the auxiliary Hamiltonian. It does not
        affect this eigenvalue read-out.

    Returns
    -------
    U : float
        Smallest eigenvalue of the auxiliary internal Hamiltonian.

    Notes
    -----
    The returned scalar is a structural spectral diagnostic.  It is not a
    mechanical potential and no force is obtained by differentiating it with
    respect to ``positions``.
    """
    _validate_positions(G, positions)
    return compute_internal_hamiltonian_ground_state(G, hbar_str=hbar_str)


def compute_node_projector_commutator_rates(
    G: TNFRGraph,
    node_ids: list[str],
    hbar_str: float = 1.0,
) -> NDArray[np.floating]:
    r"""Evaluate the diagonal localized-projector commutator read-out.

    For ``P_n = |n><n|``, the selected diagonal matrix element obeys

    ``<n|[H, P_n]|n> = H_nn - H_nn = 0``.

    The result is therefore identically zero for every finite Hamiltonian.  It
    is exposed to document the historical calculation and must not be used as
    an autonomous phase law or as the canonical multichannel ``DeltaNFR``.
    """
    hbar_str = _finite_scalar(hbar_str, "hbar_str")
    if hbar_str == 0.0:
        raise NetworkConfigError(
            parameter="hbar_str",
            value=hbar_str,
            reason="Must be finite and non-zero",
        )
    hamiltonian = InternalHamiltonian(G, hbar_str=hbar_str)
    unknown = [node for node in node_ids if node not in hamiltonian.nodes]
    if unknown:
        raise NetworkConfigError(
            parameter="node_ids",
            value=unknown,
            reason="Every requested node must be present in the graph",
        )
    return np.zeros(len(node_ids), dtype=float)


def compute_tnfr_delta_nfr(
    G: TNFRGraph,
    node_ids: list[str],
    hbar_str: float = 1.0,
) -> NDArray[np.floating]:
    """Return the legacy localized-projector commutator read-out.

    The public name is retained for compatibility.  The implemented diagonal
    projection is algebraically zero; it is not the canonical multichannel
    ``DeltaNFR`` used by the engine.

    Parameters
    ----------
    G : TNFRGraph
        Network graph with TNFR attributes
    node_ids : list of str
        Node identifiers in order
    hbar_str : float, default=1.0
        Nonzero scale in the historical commutator expression.

    Returns
    -------
    dnfr : ndarray, shape (N,)
        Zero for every requested node.

    Notes
    -----
    Use :func:`compute_node_projector_commutator_rates` when the exact scope of
    this diagnostic should be explicit in new code.
    """
    return compute_node_projector_commutator_rates(G, node_ids, hbar_str)


class TNFRNBodySystem:
    """Second-order N-body adapter with a declared phase-coupled pair law.

    The class stores position and velocity in a separate per-node adapter
    payload while retaining the node's scalar canonical EPI seed.  It uses
    ``nu_f = 1 / mass`` as an adapter convention.  Direct mechanical stepping
    and graph mirroring are outside the canonical operator executor.

    Attributes
    ----------
    n_bodies : int
        Number of bodies
    masses : ndarray, shape (N,)
        Mechanical masses; mapped to ``nu_f`` by the adapter convention.
    positions : ndarray, shape (N, 3)
        Current positions
    velocities : ndarray, shape (N, 3)
        Current velocities
    phases : ndarray, shape (N,)
        Current phases (θ ∈ [0, 2π])
    time : float
        Current adapter time
    graph : TNFRGraph
        TNFR network representation
    hbar_str : float
        Parameter retained for auxiliary internal-Hamiltonian APIs; it does not
        enter the pair force.
    distance_regularization : float
        Positive squared-distance regularizer in the pair law.

    Notes
    -----
    This model does not assert a classical limit of the nodal equation.  It
    provides a reproducible adapter whose force and matching potential are
    written explicitly in the module documentation.
    """

    def __init__(
        self,
        n_bodies: int,
        masses: list[float] | NDArray[np.floating],
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        phases: NDArray[np.floating] | None = None,
        hbar_str: float = 1.0,
        coupling_strength: float = 0.1,
        coherence_strength: float = -1.0,
        distance_regularization: float = NBODY_DISTANCE_REGULARIZATION_DEFAULT,
    ):
        """Initialize TNFR N-body system.

        Parameters
        ----------
        n_bodies : int
            Number of bodies
        masses : array_like, shape (N,)
            Positive masses. The adapter stores ``nu_f = 1 / mass``.
        positions : ndarray, shape (N, 3)
            Initial positions
        velocities : ndarray, shape (N, 3)
            Initial velocities
        phases : ndarray, shape (N,), optional
            Initial phases. If None, initialized to zero (synchronized)
        hbar_str : float, default=1.0
            Parameter retained for auxiliary internal-Hamiltonian APIs. It does
            not enter the declared position force.
        coupling_strength : float, default=0.1
            Network coupling strength (J_0 in H_coupling)
        coherence_strength : float, default=-1.0
            Pair amplitude parameter ``C_0``. Negative values make
            synchronized positive-weight edges attractive.
        distance_regularization : float, default=0.1
            Positive short-range regularizer added to squared separation. This
            is an adapter parameter, not a canonical TNFR constant.

        Raises
        ------
        ValueError
            If dimensions mismatch or masses non-positive
        """
        if (
            isinstance(n_bodies, (bool, np.bool_))
            or not isinstance(n_bodies, Integral)
            or n_bodies < 1
        ):
            raise NetworkConfigError(
                parameter="n_bodies",
                value=n_bodies,
                reason="Must have at least one body",
            )

        self.n_bodies = int(n_bodies)
        try:
            self.masses = np.asarray(masses, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NetworkConfigError(
                parameter="masses",
                value="[not a finite real vector]",
                reason=f"Masses must have shape ({self.n_bodies},)",
            ) from exc

        if self.masses.shape != (self.n_bodies,):
            raise NetworkConfigError(
                parameter="masses",
                value=str(self.masses.shape),
                reason=f"Masses must have shape ({self.n_bodies},)",
            )

        if not np.all(np.isfinite(self.masses)) or np.any(self.masses <= 0):
            raise NetworkConfigError(
                parameter="masses",
                value="[contains invalid values]",
                reason="All masses must be finite and positive",
            )

        # State vectors
        try:
            self.positions = np.asarray(positions, dtype=float).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise NetworkConfigError(
                parameter="positions",
                value="[not a finite real matrix]",
                reason=f"Shape mismatch, expected ({self.n_bodies}, 3)",
            ) from exc
        try:
            self.velocities = np.asarray(velocities, dtype=float).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise NetworkConfigError(
                parameter="velocities",
                value="[not a finite real matrix]",
                reason=f"Shape mismatch, expected ({self.n_bodies}, 3)",
            ) from exc

        if phases is None:
            self.phases = np.zeros(self.n_bodies, dtype=float)
        else:
            try:
                self.phases = np.asarray(phases, dtype=float).copy()
            except (TypeError, ValueError, OverflowError) as exc:
                raise NetworkConfigError(
                    parameter="phases",
                    value="[not a finite real vector]",
                    reason=f"Shape mismatch, expected ({self.n_bodies},)",
                ) from exc

        # Validate shapes
        expected_shape = (self.n_bodies, 3)
        if self.positions.shape != expected_shape:
            raise NetworkConfigError(
                parameter="positions",
                value=str(self.positions.shape),
                reason=f"Shape mismatch, expected {expected_shape}",
            )
        if self.velocities.shape != expected_shape:
            raise NetworkConfigError(
                parameter="velocities",
                value=str(self.velocities.shape),
                reason=f"Shape mismatch, expected {expected_shape}",
            )
        if self.phases.shape != (self.n_bodies,):
            raise NetworkConfigError(
                parameter="phases",
                value=str(self.phases.shape),
                reason=f"Shape mismatch, expected ({self.n_bodies},)",
            )
        if not np.all(np.isfinite(self.positions)):
            raise NetworkConfigError(
                parameter="positions",
                value="[contains non-finite values]",
                reason="All positions must be finite",
            )
        if not np.all(np.isfinite(self.velocities)):
            raise NetworkConfigError(
                parameter="velocities",
                value="[contains non-finite values]",
                reason="All velocities must be finite",
            )
        if not np.all(np.isfinite(self.phases)):
            raise NetworkConfigError(
                parameter="phases",
                value="[contains non-finite values]",
                reason="All phases must be finite",
            )
        self.phases = np.mod(self.phases, 2.0 * np.pi)

        self.time = 0.0
        self.hbar_str = _finite_scalar(hbar_str, "hbar_str")
        if self.hbar_str == 0.0:
            raise NetworkConfigError(
                parameter="hbar_str",
                value=hbar_str,
                reason="Must be finite and non-zero",
            )

        # Adapter parameters
        self.coupling_strength = _finite_scalar(
            coupling_strength, "coupling_strength"
        )
        self.coherence_strength = _finite_scalar(
            coherence_strength, "coherence_strength"
        )
        self.distance_regularization = _validate_distance_regularization(
            distance_regularization
        )

        # Build TNFR graph
        self._build_graph()

    def _build_graph(self) -> None:
        """Build TNFR graph representation.

        Each body becomes a resonant node with:
        - νf = 1/m under this adapter's explicit N-body embedding
        - a separate adapter payload for position and velocity
        - a fixed scalar EPI seed for Hamiltonian compatibility
        - Phase θ
        - All-to-all coupling (full network)
        """
        import networkx as nx

        self.graph: TNFRGraph = nx.Graph()
        self.graph.graph["name"] = "tnfr_nbody_system"
        self.graph.graph["MODEL_SCOPE"] = "declared_phase_coupled_nbody_adapter"
        self.graph.graph["PHASE_DYNAMICS"] = "fixed"
        self.graph.graph["H_COUPLING_STRENGTH"] = self.coupling_strength
        self.graph.graph["H_COH_STRENGTH"] = self.coherence_strength
        self.graph.graph["NBODY_DISTANCE_REGULARIZATION"] = (
            self.distance_regularization
        )

        epi_seed = min(0.5, EPI_MAX_CANONICAL * 0.95)

        # Add nodes with TNFR attributes
        for i in range(self.n_bodies):
            node_id = f"body_{i}"

            # Adapter convention; it is not a general mass identity of the
            # first-order nodal equation, where νf acts as mobility.
            nu_f = 1.0 / self.masses[i]

            # Mechanical state is mirrored separately from canonical scalar EPI.
            epi_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }

            _, _ = create_nfr(
                node_id,
                epi=epi_seed,
                vf=nu_f,
                theta=float(self.phases[i]),
                graph=self.graph,
            )

            # ``epi`` is the historical compatibility key.  ``nbody_state``
            # states its scope without shadowing the canonical scalar ``EPI``.
            self.graph.nodes[node_id]["epi"] = epi_state
            self.graph.nodes[node_id]["nbody_state"] = epi_state

        # The declared adapter starts from uniform all-to-all coupling.  The
        # force reader honors subsequent edge removal or weight changes.
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                node_i = f"body_{i}"
                node_j = f"body_{j}"

                # Edge weight: coupling strength
                # (In more sophisticated version, could depend on distance)
                weight = self.coupling_strength
                self.graph.add_edge(node_i, node_j, weight=weight)

    def compute_energy(self) -> tuple[float, float, float]:
        """Compute kinetic plus matching pair-interaction potential energy.

        Returns
        -------
        kinetic : float
            Kinetic energy K = Σ (1/2) m v²
        potential : float
            Position-dependent potential whose gradient gives the pair force.
        total : float
            Pair-law mechanical energy E = K + U

        Notes
        -----
        This is the energy of the declared auxiliary pair law.  The internal
        Hamiltonian ground-state read-out is available separately through
        :meth:`compute_hamiltonian_readout`.
        """
        try:
            kinetic = _kinetic_energy(self.masses, self.velocities)
        except ValueError as exc:
            raise NetworkConfigError(
                parameter="energy",
                value="[outside finite range]",
                reason="Kinetic energy must remain finite",
            ) from exc

        potential = compute_nbody_pair_potential(
            self.graph,
            self.positions,
            self.distance_regularization,
        )

        try:
            total = math.fsum((kinetic, potential))
        except OverflowError as exc:
            raise NetworkConfigError(
                parameter="energy",
                value="[outside finite range]",
                reason="Total pair-law energy must remain finite",
            ) from exc

        return kinetic, potential, total

    def compute_hamiltonian_readout(self) -> float:
        """Return the separate auxiliary internal-Hamiltonian ground state."""
        return compute_internal_hamiltonian_ground_state(
            self.graph, hbar_str=self.hbar_str
        )

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
        """Advance the declared second-order pair model by one time step.

        Parameters
        ----------
        dt : float
            Positive time step in adapter time units.

        Notes
        -----
        Velocity Verlet integrates the explicit pair force.  The current model
        has no autonomous phase law, so phases stay fixed.  Mechanical state is
        mirrored to the graph's separate adapter payload after the update.
        """
        dt = _finite_scalar(dt, "dt")
        if dt <= 0.0:
            raise NetworkConfigError(
                parameter="dt",
                value=dt,
                reason="Must be finite and strictly positive",
            )

        # Synchronize graph-owned phase and coupling metadata before reading it.
        self._update_graph()

        # Compute acceleration from the explicitly declared pair law.
        accel = self._compute_tnfr_accelerations()

        # Velocity Verlet integration
        # v(t+dt/2) = v(t) + a(t) * dt/2
        v_half = self.velocities + 0.5 * accel * dt

        # r(t+dt) = r(t) + v(t+dt/2) * dt
        new_positions = self.positions + v_half * dt

        # Recompute accelerations at new positions
        total_force_new = compute_nbody_pair_forces(
            self.graph,
            new_positions,
            self.distance_regularization,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            accel_new = total_force_new / self.masses[:, None]
        if not np.all(np.isfinite(accel_new)):
            raise NetworkConfigError(
                parameter="acceleration",
                value="[outside finite range]",
                reason="Pair-law acceleration must remain finite",
            )

        # v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
        new_velocities = v_half + 0.5 * accel_new * dt
        if not np.all(np.isfinite(new_velocities)):
            raise NetworkConfigError(
                parameter="dt",
                value=dt,
                reason="Time step produced non-finite velocities",
            )
        new_time = self.time + dt
        if not math.isfinite(new_time):
            raise NetworkConfigError(
                parameter="dt",
                value=dt,
                reason="Time step produced non-finite time",
            )

        self.positions = new_positions
        self.velocities = new_velocities

        # Update time
        self.time = new_time

        # Mirror the completed position/velocity state.  The graph is an
        # adapter view and this assignment is not a canonical operator step.
        self._update_graph()

    def _compute_tnfr_accelerations(self) -> NDArray[np.floating]:
        """Compute accelerations from the declared phase-coupled force.

        Returns
        -------
        accelerations : ndarray, shape (N, 3)
            Acceleration vectors for each body

        Notes
        -----
        Multiplication by ``nu_f = 1 / mass`` is part of the adapter mapping
        from central pair force to acceleration.
        """
        total_force = compute_nbody_pair_forces(
            self.graph,
            self.positions,
            self.distance_regularization,
        )
        nu = 1.0 / self.masses
        with np.errstate(over="ignore", invalid="ignore"):
            accelerations = total_force * nu[:, None]
        if not np.all(np.isfinite(accelerations)):
            raise NetworkConfigError(
                parameter="acceleration",
                value="[outside finite range]",
                reason="Pair-law acceleration must remain finite",
            )
        return accelerations

    def _update_graph(self) -> None:
        """Update graph representation with current state."""
        for i in range(self.n_bodies):
            node_id = f"body_{i}"

            adapter_state = {
                "position": self.positions[i].copy(),
                "velocity": self.velocities[i].copy(),
            }
            self.graph.nodes[node_id]["epi"] = adapter_state
            self.graph.nodes[node_id]["nbody_state"] = adapter_state

            # Update phase
            self.graph.nodes[node_id]["theta"] = float(self.phases[i])

    def evolve(
        self,
        t_final: float,
        dt: float,
        store_interval: int = 1,
    ) -> dict[str, Any]:
        """Evolve the declared phase-coupled N-body adapter.

        Parameters
        ----------
        t_final : float
            Final time
        dt : float
            Positive adapter time step.
        store_interval : int, default=1
            Store state every N steps

        Returns
        -------
        history : dict
            Contains: time, positions, velocities, phases,
                     energy, kinetic, potential, hamiltonian_ground_state,
                     momentum, angular_momentum

        Notes
        -----
        The history's ``potential`` and ``energy`` fields refer to the explicit
        pair potential and its sum with mechanical kinetic energy.  They do not
        refer to the auxiliary internal-Hamiltonian spectrum, which is retained
        under the explicit ``hamiltonian_ground_state`` key.
        """
        dt = _finite_scalar(dt, "dt")
        if dt <= 0.0:
            raise NetworkConfigError(
                parameter="dt",
                value=dt,
                reason="Must be finite and strictly positive",
            )
        if (
            isinstance(store_interval, (bool, np.bool_))
            or not isinstance(store_interval, Integral)
            or store_interval <= 0
        ):
            raise NetworkConfigError(
                parameter="store_interval",
                value=store_interval,
                reason="Must be a strictly positive integer",
            )
        store_interval = int(store_interval)
        t_final = _finite_scalar(t_final, "t_final")
        try:
            n_steps, final_step = _integration_schedule(self.time, t_final, dt)
        except ValueError as exc:
            raise NetworkConfigError(
                parameter="dt", value=dt, reason=str(exc)
            ) from exc

        if n_steps < 1:
            raise NetworkConfigError(
                parameter="t_final",
                value=t_final,
                reason=f"Must be greater than current time {self.time}",
            )

        # Pre-allocate storage
        n_stored = 1 + (n_steps // store_interval) + int(
            n_steps % store_interval != 0
        )
        times = np.zeros(n_stored)
        positions_hist = np.zeros((n_stored, self.n_bodies, 3))
        velocities_hist = np.zeros((n_stored, self.n_bodies, 3))
        phases_hist = np.zeros((n_stored, self.n_bodies))
        energies = np.zeros(n_stored)
        kinetic_energies = np.zeros(n_stored)
        potential_energies = np.zeros(n_stored)
        hamiltonian_readouts = np.zeros(n_stored)
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
        hamiltonian_readouts[store_idx] = self.compute_hamiltonian_readout()
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
                phases_hist[store_idx] = self.phases.copy()
                K, U, E = self.compute_energy()
                kinetic_energies[store_idx] = K
                potential_energies[store_idx] = U
                energies[store_idx] = E
                hamiltonian_readouts[store_idx] = self.compute_hamiltonian_readout()
                momenta[store_idx] = self.compute_momentum()
                angular_momenta[store_idx] = self.compute_angular_momentum()
                store_idx += 1

        self.time = float(t_final)
        times[store_idx - 1] = self.time

        # Relative drift of the stored mechanical pair-law energy.  The scale
        # floor keeps the diagnostic finite when the selected energy origin is
        # numerically zero.
        energy_scale = max(abs(float(energies[0])), float(np.finfo(float).eps))
        energy_drift = abs(float(energies[store_idx - 1] - energies[0])) / energy_scale

        return {
            "time": times[:store_idx],
            "positions": positions_hist[:store_idx],
            "velocities": velocities_hist[:store_idx],
            "phases": phases_hist[:store_idx],
            "energy": energies[:store_idx],
            "kinetic": kinetic_energies[:store_idx],
            "potential": potential_energies[:store_idx],
            "hamiltonian_ground_state": hamiltonian_readouts[:store_idx],
            "momentum": momenta[:store_idx],
            "angular_momentum": angular_momenta[:store_idx],
            "energy_drift": energy_drift,
        }

    def plot_trajectories(
        self,
        history: dict[str, Any],
        show_energy: bool = True,
        show_phases: bool = True,
    ) -> Figure:
        """Plot trajectories, energy, and phase evolution.

        Parameters
        ----------
        history : dict
            Result from evolve()
        show_energy : bool, default=True
            Show pair-law energy drift plot
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
                "matplotlib required for plotting. "
                "Install with: pip install 'tnfr[viz-basic]'"
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
                label=f"Body {i + 1} (m={self.masses[i]:.2f})",
                alpha=0.7,
            )
            ax_3d.scatter(
                traj[0, 0], traj[0, 1], traj[0, 2], color=colors[i], s=100, marker="o"
            )
            ax_3d.scatter(
                traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors[i], s=50, marker="x"
            )

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("Declared Phase-Coupled N-Body Adapter")
        ax_3d.legend()

        # Pair-law energy drift
        if show_energy:
            ax_energy = fig.add_subplot(1, n_plots, plot_idx)
            plot_idx += 1

            time = history["time"]
            E = history["energy"]
            E0 = E[0]
            energy_scale = max(abs(float(E0)), float(np.finfo(float).eps))

            ax_energy.plot(
                time,
                (E - E0) / energy_scale * 100,
                label="Energy drift (%)",
                color="red",
                linewidth=2,
            )
            ax_energy.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax_energy.set_xlabel("Adapter Time")
            ax_energy.set_ylabel("ΔE/E₀ (%)")
            ax_energy.set_title("Pair-Interaction Energy Drift")
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
                    label=f"Body {i + 1}",
                    linewidth=2,
                )

            ax_phases.set_xlabel("Adapter Time")
            ax_phases.set_ylabel("Phase θ (rad)")
            ax_phases.set_title("Fixed Phase Parameters")
            ax_phases.legend()
            ax_phases.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
