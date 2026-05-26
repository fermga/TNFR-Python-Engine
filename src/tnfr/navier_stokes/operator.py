"""N2: discrete TNFR-Navier-Stokes operator on a periodic torus graph.

Implements the per-component phase-field translation of the linearised
(viscous) incompressible Navier-Stokes equation on a 2D periodic grid graph.
The non-linear advection (u . grad) u is deliberately switched off in this
baseline so that the Taylor-Green vortex admits a closed-form analytical
energy decay rate against which the discrete TNFR flow can be validated.

Canonical translation (a in {1, 2} for 2D, {1, 2, 3} for 3D)
------------------------------------------------------------
Continuum velocity component u_a(x, t) is represented by a per-node phase
field phi^(a)_i. The viscous half of the momentum equation,

    d u_a / dt = nu * laplacian(u_a),

becomes, on the discrete graph G with combinatorial Laplacian L = D - A,

    d phi^(a) / dt = - (nu / h^2) * L phi^(a),

where h is the physical grid spacing. The minus sign reflects the fact that
the graph Laplacian is positive semi-definite while the continuum Laplacian
in the heat / diffusion equation appears with a plus sign once expressed via
second differences ((phi[i+1] - 2 phi[i] + phi[i-1]) / h^2 = -L phi / h^2 in
the 1D periodic case).

Pressure / incompressibility (Phi_s, INCOMP) and the advective (u . grad) u
term are NOT included in N2; they are scheduled for N3 (energy inequality)
and N4 (BKM criterion). See theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md.

Validation (Taylor-Green vortex, 2D)
------------------------------------
Initial condition on [0, 2 pi]^2:

    u(x, y, 0) =   sin(x) * cos(y)
    v(x, y, 0) = - cos(x) * sin(y)

Both components are eigenfunctions of -laplacian with eigenvalue 2, so each
amplitude decays as exp(-2 nu t) and the kinetic energy

    E(t) = (1/2) * integral (u^2 + v^2) dA

decays as E(t) = E(0) * exp(-4 nu t). The discrete flow must reproduce this
rate to within O(h^2) finite-difference error for sufficiently smooth modes.

Honest scope
------------
N2 establishes a baseline. The Clay problem is about the full non-linear 3D
flow, where the open question is whether the advection term can amplify
gradients fast enough to produce finite-time blow-up. N2 does NOT address
that question; it only certifies the viscous half of the discrete operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np


def build_torus_graph(n: int) -> nx.Graph:
    """Build a periodic ``n x n`` grid graph (2-torus topology).

    Nodes are integer tuples ``(i, j)`` with ``0 <= i, j < n`` and carry a
    ``pos`` attribute giving the physical coordinate on ``[0, 2 pi)^2``.
    Edges connect each node to its four nearest neighbours with wrap-around.

    Parameters
    ----------
    n : int
        Linear grid resolution. Total number of nodes is ``n * n``.

    Returns
    -------
    networkx.Graph
        Undirected periodic grid graph with ``pos`` node attribute.
    """
    if n < 2:
        raise ValueError("torus grid must have at least 2 nodes per side")

    G = nx.grid_2d_graph(n, n, periodic=True)
    h = 2.0 * math.pi / n
    for (i, j) in G.nodes:
        G.nodes[(i, j)]["pos"] = (i * h, j * h)
    G.graph["spacing"] = h
    G.graph["resolution"] = n
    return G


def taylor_green_initial_condition(
    G: nx.Graph,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 2D Taylor-Green velocity components sampled on ``G``.

    Parameters
    ----------
    G : networkx.Graph
        Periodic torus graph produced by :func:`build_torus_graph`.
    amplitude : float, optional
        Velocity amplitude ``U_0``. Defaults to ``1.0``.

    Returns
    -------
    (u, v) : tuple of numpy.ndarray
        Arrays of length ``len(G.nodes)`` indexed by the canonical
        ``list(G.nodes)`` order. ``u`` carries phi^(1), ``v`` carries phi^(2).
    """
    nodes = list(G.nodes)
    u = np.empty(len(nodes), dtype=float)
    v = np.empty(len(nodes), dtype=float)
    for k, node in enumerate(nodes):
        x, y = G.nodes[node]["pos"]
        u[k] = amplitude * math.sin(x) * math.cos(y)
        v[k] = -amplitude * math.cos(x) * math.sin(y)
    return u, v


@dataclass
class TNFRNavierStokesOperator:
    """Discrete TNFR-NS operator on a periodic graph (linear viscous regime).

    Parameters
    ----------
    graph : networkx.Graph
        Periodic torus graph (see :func:`build_torus_graph`).
    viscosity : float
        Kinematic viscosity ``nu`` (>= 0).
    dimension : int, optional
        Number of velocity components (2 or 3). Defaults to 2.

    Notes
    -----
    Per-component phase fields are stored in ``self.phi`` with shape
    ``(dimension, n_nodes)`` indexed by ``list(graph.nodes)`` order. The
    combinatorial Laplacian and physical spacing are cached on construction.
    """

    graph: nx.Graph
    viscosity: float
    dimension: int = 2
    phi: np.ndarray = field(init=False)
    _laplacian: np.ndarray = field(init=False, repr=False)
    _spacing: float = field(init=False, repr=False)
    _nodes: list = field(init=False, repr=False)
    time: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.viscosity < 0:
            raise ValueError("viscosity must be non-negative")
        if self.dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if "spacing" not in self.graph.graph:
            raise ValueError(
                "graph must carry a 'spacing' attribute; "
                "use build_torus_graph() to construct it"
            )

        self._nodes = list(self.graph.nodes)
        self._spacing = float(self.graph.graph["spacing"])
        # Dense Laplacian is fine for the resolutions targeted in N2
        # (n <= 64 -> 4096 nodes). Sparse variant deferred to N3.
        self._laplacian = nx.laplacian_matrix(
            self.graph, nodelist=self._nodes
        ).toarray().astype(float)
        self.phi = np.zeros((self.dimension, len(self._nodes)), dtype=float)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    def set_taylor_green(self, amplitude: float = 1.0) -> None:
        """Initialise phi from the 2D Taylor-Green vortex.

        For ``dimension == 3`` the third component is initialised to zero
        (the classical 2D Taylor-Green has no w component).
        """
        u, v = taylor_green_initial_condition(self.graph, amplitude=amplitude)
        self.phi[0] = u
        self.phi[1] = v
        if self.dimension == 3:
            self.phi[2] = 0.0
        self.time = 0.0

    def set_components(self, components: list[np.ndarray] | np.ndarray) -> None:
        """Inject arbitrary per-component phase fields."""
        arr = np.asarray(components, dtype=float)
        if arr.shape != self.phi.shape:
            raise ValueError(
                f"components shape {arr.shape} does not match "
                f"({self.dimension}, {len(self._nodes)})"
            )
        self.phi = arr.copy()
        self.time = 0.0

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the viscous TNFR-NS flow by one Crank-Nicolson half-step.

        The discrete update is the implicit-explicit midpoint
        ``(I + (nu dt / 2 h^2) L) phi_{n+1} = (I - (nu dt / 2 h^2) L) phi_n``,
        which is unconditionally stable and second-order accurate in dt.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        coeff = self.viscosity * dt / (2.0 * self._spacing ** 2)
        n = self._laplacian.shape[0]
        identity = np.eye(n)
        lhs = identity + coeff * self._laplacian
        rhs_matrix = identity - coeff * self._laplacian
        # Solve component-wise. dimension is small (2 or 3), so the explicit
        # loop is clearer than a tensor solve.
        for a in range(self.dimension):
            self.phi[a] = np.linalg.solve(lhs, rhs_matrix @ self.phi[a])
        self.time += dt

    def run(self, dt: float, steps: int) -> np.ndarray:
        """Run ``steps`` Crank-Nicolson updates and return the energy history.

        Returns
        -------
        numpy.ndarray
            Array of length ``steps + 1`` with the discrete kinetic energy
            (sum over nodes of ``0.5 * phi^2 * h^d``) at each step including
            the initial state.
        """
        history = np.empty(steps + 1, dtype=float)
        history[0] = self.kinetic_energy()
        for k in range(1, steps + 1):
            self.step(dt)
            history[k] = self.kinetic_energy()
        return history

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def kinetic_energy(self) -> float:
        """Discrete kinetic energy (1/2) * sum_a sum_i phi^(a)_i^2 * h^d."""
        h_d = self._spacing ** self.dimension
        return 0.5 * float(np.sum(self.phi ** 2)) * h_d

    def enstrophy(self) -> float:
        """Discrete enstrophy sum_a sum_i (L phi^(a))_i^2 / h^4 * h^d.

        Uses the squared discrete Laplacian as a proxy for ``|omega|^2`` in
        the linear-viscous regime, where vorticity inherits the same decay
        rate as the velocity components for each Laplacian eigenmode.
        """
        h_d = self._spacing ** self.dimension
        h4 = self._spacing ** 4
        total = 0.0
        for a in range(self.dimension):
            laplaced = self._laplacian @ self.phi[a]
            total += float(np.sum(laplaced ** 2))
        return total * h_d / h4

    def divergence_residual(self) -> float:
        """L2 norm of the discrete divergence sum_a (D_a phi^(a))_i.

        Uses central differences along the canonical grid axes. For the
        Taylor-Green initial condition this should be zero up to round-off.
        """
        if "resolution" not in self.graph.graph:
            return float("nan")

        n = int(self.graph.graph["resolution"])
        h = self._spacing
        # Reshape each component onto the grid via the canonical node order
        grids = [
            self._component_grid(a, n) for a in range(min(self.dimension, 2))
        ]
        du_dx = (np.roll(grids[0], -1, axis=0) - np.roll(grids[0], 1, axis=0)) / (2.0 * h)
        dv_dy = (np.roll(grids[1], -1, axis=1) - np.roll(grids[1], 1, axis=1)) / (2.0 * h)
        div = du_dx + dv_dy
        return float(np.sqrt(np.mean(div ** 2)))

    def _component_grid(self, a: int, n: int) -> np.ndarray:
        """Return phi^(a) reshaped onto the (n, n) Cartesian grid."""
        grid = np.empty((n, n), dtype=float)
        for k, node in enumerate(self._nodes):
            i, j = node
            grid[i, j] = self.phi[a, k]
        return grid

    # ------------------------------------------------------------------
    # Analytical reference
    # ------------------------------------------------------------------
    def taylor_green_reference(
        self,
        times: np.ndarray,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Closed-form kinetic-energy decay of the 2D Taylor-Green vortex.

        On the continuum [0, 2 pi]^2 with the convention used by
        :func:`taylor_green_initial_condition`,

            integral_[0, 2pi]^2 sin^2(x) cos^2(y) dA = pi^2,

        and the same value for ``cos^2(x) sin^2(y)``. Hence

            E(t) = (1/2) * (pi^2 + pi^2) * amplitude^2 * exp(-4 nu t)
                 = pi^2 * amplitude^2 * exp(-4 nu t).
        """
        prefactor = (math.pi ** 2) * amplitude ** 2
        return prefactor * np.exp(-4.0 * self.viscosity * np.asarray(times))


__all__ = [
    "TNFRNavierStokesOperator",
    "build_torus_graph",
    "taylor_green_initial_condition",
]
