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
    def step(self, dt: float, advection: bool = False) -> None:
        """Advance the TNFR-NS flow by one time step.

        If ``advection`` is ``False`` (default, N2 baseline) the update is
        the pure Crank-Nicolson viscous step
        ``(I + (nu dt / 2 h^2) L) phi_{n+1} = (I - (nu dt / 2 h^2) L) phi_n``,
        unconditionally stable and second-order accurate in dt.

        If ``advection`` is ``True`` (N3) the non-linear term
        ``-(u . grad) u`` is included via Strang splitting: half-step explicit
        advection in skew-symmetric form, full Crank-Nicolson viscous step,
        half-step explicit advection. The skew-symmetric form

            A_a = -(1/2) [ u_b d_b phi_a + d_b (u_b phi_a) ]

        annihilates ``<A, phi>`` to within round-off on the periodic grid, so
        the discrete kinetic energy is dissipated solely by viscosity. This
        is the prerequisite for the discrete Leray energy inequality.

        Pressure (and the candidate INCOMP operator) is NOT enforced here;
        divergence may drift on the order of ``dt`` per step, which is
        tracked explicitly by :meth:`divergence_residual`.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        if advection:
            if self.dimension != 2:
                raise NotImplementedError(
                    "advection currently implemented for dimension == 2 only"
                )
            adv1 = self._advection_term_2d()
            self.phi = self.phi + 0.5 * dt * adv1
            self._viscous_substep(dt)
            adv2 = self._advection_term_2d()
            self.phi = self.phi + 0.5 * dt * adv2
        else:
            self._viscous_substep(dt)

        self.time += dt

    def _viscous_substep(self, dt: float) -> None:
        """Single Crank-Nicolson viscous update (does not advance ``time``)."""
        coeff = self.viscosity * dt / (2.0 * self._spacing ** 2)
        n = self._laplacian.shape[0]
        identity = np.eye(n)
        lhs = identity + coeff * self._laplacian
        rhs_matrix = identity - coeff * self._laplacian
        for a in range(self.dimension):
            self.phi[a] = np.linalg.solve(lhs, rhs_matrix @ self.phi[a])

    def _advection_term_2d(self) -> np.ndarray:
        """Skew-symmetric discrete advection ``-(u . grad) u`` on the 2-torus.

        Uses central differences on the canonical ``(n, n)`` grid layout.
        Returns an array of shape ``(dimension, n_nodes)``.
        """
        n = int(self.graph.graph["resolution"])
        h = self._spacing
        u_grid = self._component_grid(0, n)
        v_grid = self._component_grid(1, n)
        result = np.zeros_like(self.phi)
        for a in range(2):
            phi_grid = self._component_grid(a, n)
            dphi_dx = (np.roll(phi_grid, -1, axis=0) - np.roll(phi_grid, 1, axis=0)) / (2.0 * h)
            dphi_dy = (np.roll(phi_grid, -1, axis=1) - np.roll(phi_grid, 1, axis=1)) / (2.0 * h)
            conv = u_grid * dphi_dx + v_grid * dphi_dy

            ux_phi = u_grid * phi_grid
            vy_phi = v_grid * phi_grid
            d_ux_phi = (np.roll(ux_phi, -1, axis=0) - np.roll(ux_phi, 1, axis=0)) / (2.0 * h)
            d_vy_phi = (np.roll(vy_phi, -1, axis=1) - np.roll(vy_phi, 1, axis=1)) / (2.0 * h)
            div_form = d_ux_phi + d_vy_phi

            adv_grid = -0.5 * (conv + div_form)
            for k, node in enumerate(self._nodes):
                i, j = node
                result[a, k] = adv_grid[i, j]
        return result

    def run(self, dt: float, steps: int, advection: bool = False) -> np.ndarray:
        """Run ``steps`` updates and return the energy history.

        Returns
        -------
        numpy.ndarray
            Array of length ``steps + 1`` with the discrete kinetic energy at
            each step including the initial state.
        """
        history = np.empty(steps + 1, dtype=float)
        history[0] = self.kinetic_energy()
        for k in range(1, steps + 1):
            self.step(dt, advection=advection)
            history[k] = self.kinetic_energy()
        return history

    # ------------------------------------------------------------------
    # Energy / dissipation diagnostics (N3)
    # ------------------------------------------------------------------
    def dissipation_rate(self) -> float:
        """Instantaneous viscous dissipation rate ``nu * sum_a <phi_a, L phi_a>``.

        On a periodic 2D grid the quadratic form ``<phi, L phi>`` equals
        ``sum_edges (phi_i - phi_j)^2``, which approximates the continuum
        ``integral |grad phi|^2 dA`` (the graph Laplacian carries an
        implicit ``h^2`` factor that exactly cancels the Riemann area
        element ``h^d`` in 2D). With ``E = (1/2) ||u||_{L^2}^2`` the energy
        identity reads ``dE/dt = -nu * ||grad u||_{L^2}^2``, so the
        discrete analogue of the instantaneous dissipation that enters the
        Leray inequality is ``nu * sum_a <phi_a, L phi_a>``.
        """
        total = 0.0
        for a in range(self.dimension):
            total += float(self.phi[a] @ (self._laplacian @ self.phi[a]))
        return self.viscosity * total

    def leray_budget(
        self,
        dt: float,
        steps: int,
        advection: bool = True,
    ) -> dict[str, np.ndarray]:
        """Run the flow and return the discrete Leray energy-inequality budget.

        Records, at every step ``n = 0 .. steps``,

            - ``time``       : t_n = n * dt
            - ``energy``     : E(t_n) = (1/2) ||phi||^2 * h^d
            - ``dissipation``: D(t_n) = nu * sum_a <phi_a, L phi_a>
            - ``cumulative`` : E(0) - E(t_n) - integral_0^{t_n} D(tau) dtau
            - ``divergence`` : L2 residual of the discrete divergence

        For the continuum Leray weak solution the inequality

            E(t) + nu * integral_0^t ||grad u||^2 dtau  <=  E(0)

        translates discretely to ``cumulative >= 0`` at every step (energy
        dissipated viscously up to time t cannot exceed the energy actually
        lost). A monotone non-decreasing ``cumulative`` sequence certifies
        that the advection step does not inject spurious energy.
        """
        time_history = np.empty(steps + 1, dtype=float)
        energy_history = np.empty(steps + 1, dtype=float)
        dissipation_history = np.empty(steps + 1, dtype=float)
        divergence_history = np.empty(steps + 1, dtype=float)

        time_history[0] = self.time
        energy_history[0] = self.kinetic_energy()
        dissipation_history[0] = self.dissipation_rate()
        divergence_history[0] = self.divergence_residual()

        for k in range(1, steps + 1):
            self.step(dt, advection=advection)
            time_history[k] = self.time
            energy_history[k] = self.kinetic_energy()
            dissipation_history[k] = self.dissipation_rate()
            divergence_history[k] = self.divergence_residual()

        # Trapezoidal integral of dissipation (continuous-time analogue).
        cumulative_dissipated = np.zeros(steps + 1, dtype=float)
        for k in range(1, steps + 1):
            cumulative_dissipated[k] = cumulative_dissipated[k - 1] + 0.5 * (
                dissipation_history[k] + dissipation_history[k - 1]
            ) * dt
        cumulative_budget = (
            energy_history[0] - energy_history - cumulative_dissipated
        )

        return {
            "time": time_history,
            "energy": energy_history,
            "dissipation": dissipation_history,
            "cumulative_dissipated": cumulative_dissipated,
            "cumulative_budget": cumulative_budget,
            "divergence": divergence_history,
        }

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
    # N4: vorticity, enstrophy, discrete BKM criterion (2D)
    # ------------------------------------------------------------------
    def vorticity_2d(self) -> np.ndarray:
        """Discrete scalar vorticity ``omega = d_x v - d_y u`` on the 2D torus.

        Uses central differences on the canonical ``(n, n)`` periodic grid
        with spacing ``h = 2 pi / n``. Returns an ``(n, n)`` array. Only
        defined for ``self.dimension == 2``.
        """
        if self.dimension != 2:
            raise NotImplementedError(
                "vorticity_2d() requires self.dimension == 2"
            )
        if "resolution" not in self.graph.graph:
            raise RuntimeError("operator graph lacks 'resolution' metadata")
        n = int(self.graph.graph["resolution"])
        h = self._spacing
        u = self._component_grid(0, n)
        v = self._component_grid(1, n)
        dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * h)
        du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * h)
        return dv_dx - du_dy

    def vorticity_sup_norm(self) -> float:
        """Discrete ``|| omega ||_{L^inf}`` (2D only).

        The Beale-Kato-Majda (BKM) criterion controls finite-time
        continuation of smooth solutions by the time integral
        ``int_0^T || omega(tau) ||_{L^inf} dtau``.
        """
        return float(np.max(np.abs(self.vorticity_2d())))

    def enstrophy_curl(self) -> float:
        """Discrete enstrophy ``Omega = (1/2) * integral omega^2 dA`` (2D).

        Built from the true curl ``vorticity_2d()`` rather than the
        Laplacian-squared proxy used by :meth:`enstrophy`. The integral
        is approximated by Riemann sum with area element ``h^2``.
        """
        omega = self.vorticity_2d()
        h2 = self._spacing ** 2
        return 0.5 * float(np.sum(omega ** 2)) * h2

    def bkm_budget(
        self,
        dt: float,
        steps: int,
        advection: bool = True,
    ) -> dict[str, np.ndarray]:
        """Track the discrete BKM integral and enstrophy along the flow.

        Beale-Kato-Majda (BKM, 1984): a smooth solution on ``[0, T*)`` of
        the incompressible Navier-Stokes equations extends past ``T*`` if
        and only if

            int_0^{T*} || omega(tau) ||_{L^inf} dtau  <  infinity.

        In 2D the vorticity equation has no stretching term, so this
        integral stays bounded for all time and the flow is globally
        regular. The demo therefore certifies the *consistency* of the
        discrete operator with the well-known 2D global-regularity
        result, and provides the infrastructure that will be reused in
        3D (where blow-up via vortex stretching is the open Clay
        question, NS-G5).

        Returns a dict with keys ``time``, ``vorticity_sup``,
        ``enstrophy``, ``bkm_integral`` (trapezoidal cumulative sum) and
        ``divergence``. Lengths are ``steps + 1``.
        """
        time_axis = np.zeros(steps + 1, dtype=float)
        vort_sup = np.zeros(steps + 1, dtype=float)
        enstrophy_hist = np.zeros(steps + 1, dtype=float)
        bkm_int = np.zeros(steps + 1, dtype=float)
        div_hist = np.zeros(steps + 1, dtype=float)

        time_axis[0] = self.time
        vort_sup[0] = self.vorticity_sup_norm()
        enstrophy_hist[0] = self.enstrophy_curl()
        div_hist[0] = self.divergence_residual()

        for n_step in range(1, steps + 1):
            self.step(dt, advection=advection)
            time_axis[n_step] = self.time
            vort_sup[n_step] = self.vorticity_sup_norm()
            enstrophy_hist[n_step] = self.enstrophy_curl()
            div_hist[n_step] = self.divergence_residual()
            # Trapezoidal accumulation of int_0^{t_n} ||omega||_inf dtau
            bkm_int[n_step] = bkm_int[n_step - 1] + 0.5 * dt * (
                vort_sup[n_step - 1] + vort_sup[n_step]
            )

        return {
            "time": time_axis,
            "vorticity_sup": vort_sup,
            "enstrophy": enstrophy_hist,
            "bkm_integral": bkm_int,
            "divergence": div_hist,
        }


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
