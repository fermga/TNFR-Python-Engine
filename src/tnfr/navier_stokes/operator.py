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
    for i, j in G.nodes:
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


def build_torus_graph_3d(n: int) -> nx.Graph:
    """Build a periodic ``n x n x n`` grid graph (3-torus topology).

    Nodes are integer triples ``(i, j, k)`` with ``0 <= i, j, k < n`` and
    carry a ``pos`` attribute giving the physical coordinate on
    ``[0, 2 pi)^3``. Edges connect each node to its six nearest neighbours
    with wrap-around on every axis.

    Parameters
    ----------
    n : int
        Linear grid resolution. Total number of nodes is ``n^3``.

    Returns
    -------
    networkx.Graph
        Undirected periodic 3D grid graph with ``pos`` node attribute,
        ``spacing = 2*pi/n``, ``resolution = n`` and ``ndim = 3``.

    Notes
    -----
    Used by the N6 milestone (vortex stretching / 3D NS). The viscous
    Crank-Nicolson step uses FFT diagonalisation rather than the dense
    Laplacian so that runtime is O(n^3 log n) instead of O(n^9).
    """
    if n < 2:
        raise ValueError("torus grid must have at least 2 nodes per side")

    G = nx.grid_graph(dim=(n, n, n), periodic=True)
    h = 2.0 * math.pi / n
    for i, j, k in G.nodes:
        G.nodes[(i, j, k)]["pos"] = (i * h, j * h, k * h)
    G.graph["spacing"] = h
    G.graph["resolution"] = n
    G.graph["ndim"] = 3
    return G


def taylor_green_initial_condition_3d(
    G: nx.Graph,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the classical 3D Taylor-Green velocity components sampled on ``G``.

    Initial condition on ``[0, 2 pi]^3``:

        u(x, y, z, 0) =   A * sin(x) * cos(y) * cos(z)
        v(x, y, z, 0) = - A * cos(x) * sin(y) * cos(z)
        w(x, y, z, 0) = 0

    This field is exactly divergence-free in the continuum and is the
    standard benchmark for 3D Navier-Stokes simulations of transition to
    turbulence. Unlike the 2D Taylor-Green vortex, it activates the
    vortex-stretching term ``(omega . grad) u`` immediately (the initial
    vorticity ``omega(0)`` has all three components and is not aligned
    with eigenvectors of the strain tensor), so it is the smallest test
    case in which the genuine 3D NS non-linearity is exercised.

    Returns
    -------
    (u, v, w) : tuple of numpy.ndarray
        Arrays of length ``len(G.nodes)`` indexed by the canonical
        ``list(G.nodes)`` order.
    """
    nodes = list(G.nodes)
    u = np.empty(len(nodes), dtype=float)
    v = np.empty(len(nodes), dtype=float)
    w = np.zeros(len(nodes), dtype=float)
    for idx, node in enumerate(nodes):
        x, y, z = G.nodes[node]["pos"]
        u[idx] = amplitude * math.sin(x) * math.cos(y) * math.cos(z)
        v[idx] = -amplitude * math.cos(x) * math.sin(y) * math.cos(z)
    return u, v, w


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
        if self.dimension == 2:
            # Dense Laplacian is fine for 2D at the resolutions targeted in N2
            # (n <= 64 -> 4096 nodes). 3D uses an FFT-diagonalised viscous step
            # to avoid the O(n^9) cost of np.linalg.solve on a dense (n^3)x(n^3)
            # operator (n=16 -> 4096^2 = 16.7 M entries, n=32 -> 1.07 G entries).
            self._laplacian = (
                nx.laplacian_matrix(self.graph, nodelist=self._nodes)
                .toarray()
                .astype(float)
            )
        else:
            # N6: 3D path skips the dense Laplacian. Viscous step uses FFT
            # diagonalisation on the periodic torus; dissipation_rate() likewise
            # routes through the FFT path.
            self._laplacian = np.empty((0, 0), dtype=float)
        self.phi = np.zeros((self.dimension, len(self._nodes)), dtype=float)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    def set_taylor_green(self, amplitude: float = 1.0) -> None:
        """Initialise phi from the Taylor-Green vortex.

        For ``dimension == 2`` uses the classical 2D Taylor-Green initial
        condition; for ``dimension == 3`` uses the classical *3D* Taylor-
        Green vortex (see :func:`taylor_green_initial_condition_3d`), which
        is divergence-free and activates the vortex-stretching term
        ``(omega . grad) u`` that vanishes identically in 2D.
        """
        if self.dimension == 2:
            u, v = taylor_green_initial_condition(self.graph, amplitude=amplitude)
            self.phi[0] = u
            self.phi[1] = v
        else:
            u, v, w = taylor_green_initial_condition_3d(self.graph, amplitude=amplitude)
            self.phi[0] = u
            self.phi[1] = v
            self.phi[2] = w
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
    def step(
        self,
        dt: float,
        advection: bool = False,
        incompressible: bool = False,
    ) -> None:
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

        If ``incompressible`` is ``True`` (N5, INCOMP activated) the Leray-
        Helmholtz projection :meth:`project_incompressible` is applied after
        every sub-step that may break the divergence-free constraint (each
        advection half-step). The viscous Crank-Nicolson update commutes
        with the divergence on the periodic torus, so projection there is
        unnecessary in exact arithmetic but is applied at the end of the
        step as a defensive measure against accumulated round-off.

        Pressure / INCOMP defaults to OFF; without it, divergence drifts on
        the order of ``dt`` per advection step, which is tracked explicitly
        by :meth:`divergence_residual`.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        if advection:
            adv1 = self._advection_term()
            self.phi = self.phi + 0.5 * dt * adv1
            if incompressible:
                self.project_incompressible()
            self._viscous_substep(dt)
            adv2 = self._advection_term()
            self.phi = self.phi + 0.5 * dt * adv2
            if incompressible:
                self.project_incompressible()
        else:
            self._viscous_substep(dt)
            if incompressible:
                self.project_incompressible()

        self.time += dt

    def _advection_term(self) -> np.ndarray:
        """Dispatch to the dimension-specific skew-symmetric advection term."""
        if self.dimension == 2:
            return self._advection_term_2d()
        return self._advection_term_3d()

    def _viscous_substep(self, dt: float) -> None:
        """Single Crank-Nicolson viscous update (does not advance ``time``)."""
        if self.dimension == 2:
            coeff = self.viscosity * dt / (2.0 * self._spacing**2)
            n = self._laplacian.shape[0]
            identity = np.eye(n)
            lhs = identity + coeff * self._laplacian
            rhs_matrix = identity - coeff * self._laplacian
            for a in range(self.dimension):
                self.phi[a] = np.linalg.solve(lhs, rhs_matrix @ self.phi[a])
        else:
            self._viscous_substep_fft_3d(dt)

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
            dphi_dx = (np.roll(phi_grid, -1, axis=0) - np.roll(phi_grid, 1, axis=0)) / (
                2.0 * h
            )
            dphi_dy = (np.roll(phi_grid, -1, axis=1) - np.roll(phi_grid, 1, axis=1)) / (
                2.0 * h
            )
            conv = u_grid * dphi_dx + v_grid * dphi_dy

            ux_phi = u_grid * phi_grid
            vy_phi = v_grid * phi_grid
            d_ux_phi = (np.roll(ux_phi, -1, axis=0) - np.roll(ux_phi, 1, axis=0)) / (
                2.0 * h
            )
            d_vy_phi = (np.roll(vy_phi, -1, axis=1) - np.roll(vy_phi, 1, axis=1)) / (
                2.0 * h
            )
            div_form = d_ux_phi + d_vy_phi

            adv_grid = -0.5 * (conv + div_form)
            for k, node in enumerate(self._nodes):
                i, j = node
                result[a, k] = adv_grid[i, j]
        return result

    # ------------------------------------------------------------------
    # N5: INCOMP operator (Leray-Helmholtz projection on the periodic torus)
    # ------------------------------------------------------------------
    def project_incompressible(self) -> None:
        """Dispatch Leray-Helmholtz projection to the 2D or 3D implementation.

        See :meth:`_project_incompressible_2d` (N5 baseline) and
        :meth:`_project_incompressible_3d` (N6 extension) for the algorithm
        details. Both use the *exact* central-difference symbol
        ``S_a = sin(2 pi m_a / n) / h`` so that :meth:`divergence_residual`
        drops to round-off after the call.

        Pressure (Phi_s) is cached as a physical-space grid and exposed via
        :meth:`pressure_field`.
        """
        if self.dimension == 2:
            self._project_incompressible_2d()
        else:
            self._project_incompressible_3d()

    def _project_incompressible_2d(self) -> None:
        """INCOMP (2D): Leray-Helmholtz projection onto div-free subspace.

        Removes the gradient component of ``phi`` so that the discrete
        central-difference divergence (the same operator probed by
        :meth:`divergence_residual`) drops to round-off after the call.

        The projection is performed in Fourier space using the *exact*
        symbol of the central-difference operator

            D_a f -> sin(2 pi m_a / n) / h

        which guarantees that ``D_x u_proj + D_y v_proj == 0`` to machine
        precision, not merely the *spectral* divergence ``i k . u_hat``.

        Algorithm (pseudo-spectral, O(n^2 log n)):

            1. FFT each component:    u_hat, v_hat = FFT2(u), FFT2(v)
            2. Discrete symbols:      S_a = sin(2 pi m / n) / h
            3. Discrete pressure:     p_hat = (S_x u_hat + S_y v_hat) / |S|^2
            4. Project:               u_hat -= S_x p_hat;  v_hat -= S_y p_hat
            5. Inverse FFT back to phi.

        The zero mode (m = 0) and the Nyquist modes (m = n/2 for even n)
        are in the null space of the central-difference operator and are
        left unchanged; Taylor-Green initial data (lowest mode m = 1) is
        unaffected by this null-space caveat.

        Pressure interpretation (Phi_s field). The Lagrange multiplier
        ``p_hat`` is exactly the discrete pressure that the incompressible
        Navier-Stokes equation introduces to enforce ``div u = 0``; in
        TNFR language this is the structural potential ``Phi_s`` whose
        gradient cancels the longitudinal part of the convective term.
        :meth:`pressure_field` returns the physical-space pressure.

        Canonicity (open). INCOMP is a *global, non-local* operator on the
        periodic Fourier basis; it cannot be decomposed into the
        nearest-neighbour TNFR operators {IL, UM, RA, ...}. Whether it
        should be promoted to the 14th canonical TNFR operator, or kept
        as a derived projection bound to the incompressibility constraint
        (mirroring how the Riemann program kept the catalog frozen at
        13), is an open structural question documented in the NS research
        notes alongside the analogous T-HP question.
        """
        if "resolution" not in self.graph.graph:
            raise RuntimeError(
                "graph lacks 'resolution' metadata; project_incompressible() "
                "requires a torus graph built via build_torus_graph()"
            )

        n = int(self.graph.graph["resolution"])
        h = self._spacing

        u = self._component_grid(0, n)
        v = self._component_grid(1, n)

        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        # Discrete central-difference symbols matching divergence_residual()
        # and _advection_term_2d().
        m = np.arange(n)
        S1d = np.sin(2.0 * np.pi * m / n) / h
        Sx = S1d[:, None]  # varies along axis 0 (x)
        Sy = S1d[None, :]  # varies along axis 1 (y)
        S2 = Sx**2 + Sy**2

        # Discrete pressure in Fourier space; null modes (S2 == 0)
        # contribute 0 (left unchanged).
        denom = np.where(S2 > 0.0, S2, 1.0)
        p_hat = (Sx * u_hat + Sy * v_hat) / denom
        p_hat = np.where(S2 > 0.0, p_hat, 0.0 + 0.0j)

        u_hat_proj = u_hat - Sx * p_hat
        v_hat_proj = v_hat - Sy * p_hat

        u_proj = np.fft.ifft2(u_hat_proj).real
        v_proj = np.fft.ifft2(v_hat_proj).real

        # Cache pressure (physical space) for diagnostics / Phi_s readout.
        self._pressure_grid = np.fft.ifft2(p_hat).real

        # Write back to flat phi storage following canonical node order.
        for k, node in enumerate(self._nodes):
            i, j = node
            self.phi[0, k] = u_proj[i, j]
            self.phi[1, k] = v_proj[i, j]

    def pressure_field(self) -> np.ndarray:
        """Return the most recent INCOMP pressure (Phi_s) grid, shape (n, n).

        Populated by :meth:`project_incompressible`. Returns an array of
        ``nan`` when INCOMP has not been applied yet.
        """
        if getattr(self, "_pressure_grid", None) is None:
            n = int(self.graph.graph.get("resolution", 0)) or 1
            return np.full((n, n), float("nan"))
        return self._pressure_grid.copy()

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

        3D path. The dense Laplacian is not materialised in 3D; the same
        quadratic form is evaluated in Fourier space using the analytic
        eigenvalues of the periodic torus Laplacian.
        """
        if self.dimension == 2:
            total = 0.0
            for a in range(self.dimension):
                total += float(self.phi[a] @ (self._laplacian @ self.phi[a]))
            return self.viscosity * total
        return self._dissipation_rate_fft_3d()

    def leray_budget(
        self,
        dt: float,
        steps: int,
        advection: bool = True,
        incompressible: bool = False,
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
            self.step(dt, advection=advection, incompressible=incompressible)
            time_history[k] = self.time
            energy_history[k] = self.kinetic_energy()
            dissipation_history[k] = self.dissipation_rate()
            divergence_history[k] = self.divergence_residual()

        # Trapezoidal integral of dissipation (continuous-time analogue).
        cumulative_dissipated = np.zeros(steps + 1, dtype=float)
        for k in range(1, steps + 1):
            cumulative_dissipated[k] = (
                cumulative_dissipated[k - 1]
                + 0.5 * (dissipation_history[k] + dissipation_history[k - 1]) * dt
            )
        cumulative_budget = energy_history[0] - energy_history - cumulative_dissipated

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
        h_d = self._spacing**self.dimension
        return 0.5 * float(np.sum(self.phi**2)) * h_d

    def enstrophy(self) -> float:
        """Discrete enstrophy sum_a sum_i (L phi^(a))_i^2 / h^4 * h^d.

        Uses the squared discrete Laplacian as a proxy for ``|omega|^2`` in
        the linear-viscous regime, where vorticity inherits the same decay
        rate as the velocity components for each Laplacian eigenmode.
        """
        h_d = self._spacing**self.dimension
        h4 = self._spacing**4
        total = 0.0
        for a in range(self.dimension):
            laplaced = self._laplacian @ self.phi[a]
            total += float(np.sum(laplaced**2))
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
        if self.dimension == 2:
            u = self._component_grid(0, n)
            v = self._component_grid(1, n)
            du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * h)
            dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * h)
            div = du_dx + dv_dy
        else:
            u = self._component_grid_3d(0, n)
            v = self._component_grid_3d(1, n)
            w = self._component_grid_3d(2, n)
            du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * h)
            dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * h)
            dw_dz = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2.0 * h)
            div = du_dx + dv_dy + dw_dz
        return float(np.sqrt(np.mean(div**2)))

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
            raise NotImplementedError("vorticity_2d() requires self.dimension == 2")
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
        """Discrete ``|| omega ||_{L^inf}`` (2D scalar or 3D vector).

        In 2D ``omega`` is the scalar curl; in 3D it is the magnitude of
        the vector curl ``|omega|``. The Beale-Kato-Majda (BKM) criterion
        controls finite-time continuation of smooth solutions by the time
        integral ``int_0^T || omega(tau) ||_{L^inf} dtau``.
        """
        if self.dimension == 2:
            return float(np.max(np.abs(self.vorticity_2d())))
        omega = self.vorticity_3d()
        mag = np.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
        return float(np.max(mag))

    def enstrophy_curl(self) -> float:
        """Discrete enstrophy ``Omega = (1/2) * integral |omega|^2 dV``.

        Built from the true curl (:meth:`vorticity_2d` in 2D,
        :meth:`vorticity_3d` in 3D) rather than the Laplacian-squared proxy
        used by :meth:`enstrophy`. The integral is approximated by Riemann
        sum with volume element ``h^d``.
        """
        if self.dimension == 2:
            omega = self.vorticity_2d()
            h2 = self._spacing**2
            return 0.5 * float(np.sum(omega**2)) * h2
        omega = self.vorticity_3d()
        mag2 = omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2
        h3 = self._spacing**3
        return 0.5 * float(np.sum(mag2)) * h3

    def bkm_budget(
        self,
        dt: float,
        steps: int,
        advection: bool = True,
        incompressible: bool = False,
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
            self.step(dt, advection=advection, incompressible=incompressible)
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
        prefactor = (math.pi**2) * amplitude**2
        return prefactor * np.exp(-4.0 * self.viscosity * np.asarray(times))

    # ==================================================================
    # N6: 3D extension (vortex stretching, Constantin-Fefferman regime)
    # ==================================================================
    # The methods below activate the genuine Clay regime: the velocity
    # field lives on a periodic 3-torus, the advection term carries the
    # vortex-stretching coupling ``(omega . grad) u`` that vanishes
    # identically in 2D, and the Leray-Helmholtz projection enforces
    # incompressibility in 3D. Honest scope: none of this *closes* the
    # global-regularity question NS-G5 (Clay Millennium Problem). What
    # it does is provide the discrete infrastructure on which the
    # Constantin-Fefferman geometric-depletion mechanism (alignment of
    # the vorticity direction with strain eigenvectors) can be measured
    # empirically.
    # ------------------------------------------------------------------

    def _component_grid_3d(self, a: int, n: int) -> np.ndarray:
        """Return ``phi^(a)`` reshaped onto the periodic ``(n, n, n)`` grid.

        Indexed by the canonical node order ``(i, j, k)`` produced by
        :func:`build_torus_graph_3d`.
        """
        grid = np.empty((n, n, n), dtype=float)
        for idx, node in enumerate(self._nodes):
            i, j, k = node
            grid[i, j, k] = self.phi[a, idx]
        return grid

    def _flatten_grid_3d(self, grid: np.ndarray, a: int) -> None:
        """Write a ``(n, n, n)`` grid back into ``self.phi[a]`` following node order."""
        for idx, node in enumerate(self._nodes):
            i, j, k = node
            self.phi[a, idx] = grid[i, j, k]

    def _fft_symbols_3d(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the discrete central-difference symbols ``S_x, S_y, S_z`` and ``|S|^2``.

        Matches the symbol used by :meth:`divergence_residual` and the
        skew-symmetric advection term, so that projection drives the
        physical-space discrete divergence to round-off.
        """
        h = self._spacing
        m = np.arange(n)
        S1d = np.sin(2.0 * np.pi * m / n) / h
        Sx = S1d[:, None, None]
        Sy = S1d[None, :, None]
        Sz = S1d[None, None, :]
        S2 = Sx**2 + Sy**2 + Sz**2
        return Sx, Sy, Sz, S2

    def _laplace_eigenvalues_3d(self, n: int) -> np.ndarray:
        """Eigenvalues of the graph Laplacian on the periodic 3-torus.

        For an unnormalised combinatorial Laplacian ``L = D - A`` with
        degree 6 on a periodic 3D grid the eigenvalues at Fourier mode
        ``(m1, m2, m3)`` are

            lambda(m) = 2 * (3 - cos(2 pi m1/n) - cos(2 pi m2/n) - cos(2 pi m3/n)).

        Returned as an ``(n, n, n)`` array indexed as ``[m1, m2, m3]``.
        """
        m = np.arange(n)
        c = np.cos(2.0 * np.pi * m / n)
        cx = c[:, None, None]
        cy = c[None, :, None]
        cz = c[None, None, :]
        return 2.0 * (3.0 - cx - cy - cz)

    def _viscous_substep_fft_3d(self, dt: float) -> None:
        """Crank-Nicolson viscous half-step in 3D, diagonalised by FFT.

        Replaces ``np.linalg.solve`` on a dense ``(n^3) x (n^3)`` matrix
        with a per-mode scalar update, reducing cost from O(n^9) to
        O(n^3 log n). Bit-equivalence with the dense 2D path is not
        required (each path uses its own canonical implementation); both
        paths use the *combinatorial* graph Laplacian with an explicit
        ``1/h^2`` scale factor.
        """
        n = int(self.graph.graph["resolution"])
        coeff = self.viscosity * dt / (2.0 * self._spacing**2)
        lam = self._laplace_eigenvalues_3d(n)
        amp = (1.0 - coeff * lam) / (1.0 + coeff * lam)
        for a in range(self.dimension):
            grid = self._component_grid_3d(a, n)
            hat = np.fft.fftn(grid, axes=(0, 1, 2))
            hat *= amp
            grid = np.fft.ifftn(hat, axes=(0, 1, 2)).real
            self._flatten_grid_3d(grid, a)

    def _dissipation_rate_fft_3d(self) -> float:
        """Instantaneous dissipation rate ``nu * sum_a <phi_a, L phi_a>`` in 3D.

        Uses the analytic graph-Laplacian eigenvalues so that no dense
        matrix is materialised. The convention matches the 2D path:
        unnormalised combinatorial Laplacian (eigenvalues O(1)), not the
        physical Laplacian ``L/h^2``.
        """
        n = int(self.graph.graph["resolution"])
        lam = self._laplace_eigenvalues_3d(n)
        total = 0.0
        for a in range(self.dimension):
            grid = self._component_grid_3d(a, n)
            hat = np.fft.fftn(grid, axes=(0, 1, 2))
            # Parseval: sum |hat|^2 / n^3 = sum |grid|^2 ; weighted by lambda.
            total += float(np.sum(lam * np.abs(hat) ** 2)) / (n**3)
        return self.viscosity * total

    def _advection_term_3d(self) -> np.ndarray:
        """Skew-symmetric 3D advection ``-(1/2)[(u.grad) phi + grad.(u phi)]``.

        Returns an array shaped ``(dim, n_nodes)`` matching the layout of
        ``self.phi``. The skew-symmetric (rotational) form is chosen so
        that quadratic invariants (energy, helicity) are preserved
        discretely by the advection step up to round-off (the dissipation
        comes exclusively from the viscous CN step).

        Vortex-stretching content. In 3D, after projection onto the
        divergence-free subspace the convective term is equivalent to
        ``(omega x u) + grad(|u|^2 / 2)``; the gradient part is removed
        by INCOMP, leaving the Lamb form whose curl contains the genuine
        ``(omega . grad) u`` stretching term that drives the Clay
        question NS-G5. This term is exposed independently by
        :meth:`vortex_stretching_field` for diagnostics.
        """
        n = int(self.graph.graph["resolution"])
        h = self._spacing
        u = self._component_grid_3d(0, n)
        v = self._component_grid_3d(1, n)
        w = self._component_grid_3d(2, n)
        comps = (u, v, w)

        def d_axis(arr: np.ndarray, axis: int) -> np.ndarray:
            return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (
                2.0 * h
            )

        result = np.empty_like(self.phi)
        for a in range(3):
            phi_a = comps[a]
            # Convective form: u_b * d_b phi_a
            conv = u * d_axis(phi_a, 0) + v * d_axis(phi_a, 1) + w * d_axis(phi_a, 2)
            # Divergence form: d_b (u_b * phi_a)
            divf = d_axis(u * phi_a, 0) + d_axis(v * phi_a, 1) + d_axis(w * phi_a, 2)
            adv_grid = -0.5 * (conv + divf)
            for idx, node in enumerate(self._nodes):
                i, j, k = node
                result[a, idx] = adv_grid[i, j, k]
        return result

    def _project_incompressible_3d(self) -> None:
        """INCOMP (3D): Leray-Helmholtz projection onto the divergence-free subspace.

        Pseudo-spectral algorithm using the *exact* discrete central-difference
        symbol ``S_a = sin(2 pi m_a / n) / h``, so that
        :meth:`divergence_residual` drops to round-off after the call (it
        agrees with the central-difference divergence operator probed by
        :meth:`divergence_residual`, not merely with the spectral symbol
        ``i k . u_hat``).

        Cost: O(n^3 log n) via :func:`numpy.fft.fftn`.

        Pressure (Phi_s). The Lagrange multiplier ``p_hat`` is the discrete
        pressure that the incompressible NS equation introduces to enforce
        ``div u = 0``; in TNFR language this is the structural potential
        ``Phi_s`` whose gradient cancels the longitudinal part of the
        convective term. The physical-space pressure is cached as an
        ``(n, n, n)`` array and exposed via :meth:`pressure_field`.

        Null space: zero mode and Nyquist modes (``S2 == 0``) are left
        unchanged; classical 3D Taylor-Green (lowest non-trivial mode) is
        unaffected.
        """
        n = int(self.graph.graph["resolution"])
        Sx, Sy, Sz, S2 = self._fft_symbols_3d(n)

        u = self._component_grid_3d(0, n)
        v = self._component_grid_3d(1, n)
        w = self._component_grid_3d(2, n)

        u_hat = np.fft.fftn(u, axes=(0, 1, 2))
        v_hat = np.fft.fftn(v, axes=(0, 1, 2))
        w_hat = np.fft.fftn(w, axes=(0, 1, 2))

        denom = np.where(S2 > 0.0, S2, 1.0)
        p_hat = (Sx * u_hat + Sy * v_hat + Sz * w_hat) / denom
        p_hat = np.where(S2 > 0.0, p_hat, 0.0 + 0.0j)

        u_hat_proj = u_hat - Sx * p_hat
        v_hat_proj = v_hat - Sy * p_hat
        w_hat_proj = w_hat - Sz * p_hat

        u_proj = np.fft.ifftn(u_hat_proj, axes=(0, 1, 2)).real
        v_proj = np.fft.ifftn(v_hat_proj, axes=(0, 1, 2)).real
        w_proj = np.fft.ifftn(w_hat_proj, axes=(0, 1, 2)).real

        self._pressure_grid = np.fft.ifftn(p_hat, axes=(0, 1, 2)).real

        self._flatten_grid_3d(u_proj, 0)
        self._flatten_grid_3d(v_proj, 1)
        self._flatten_grid_3d(w_proj, 2)

    def vorticity_3d(self) -> np.ndarray:
        """Discrete vector vorticity ``omega = curl u`` on the 3D torus.

        Returns an ``(3, n, n, n)`` array with components

            omega_x = d_y w - d_z v
            omega_y = d_z u - d_x w
            omega_z = d_x v - d_y u

        Central differences on the canonical periodic grid.
        """
        if self.dimension != 3:
            raise NotImplementedError("vorticity_3d() requires self.dimension == 3")
        n = int(self.graph.graph["resolution"])
        h = self._spacing
        u = self._component_grid_3d(0, n)
        v = self._component_grid_3d(1, n)
        w = self._component_grid_3d(2, n)

        def d_axis(arr: np.ndarray, axis: int) -> np.ndarray:
            return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (
                2.0 * h
            )

        omega = np.empty((3, n, n, n), dtype=float)
        omega[0] = d_axis(w, 1) - d_axis(v, 2)
        omega[1] = d_axis(u, 2) - d_axis(w, 0)
        omega[2] = d_axis(v, 0) - d_axis(u, 1)
        return omega

    def vortex_stretching_field(self) -> np.ndarray:
        """Discrete vortex-stretching term ``S_a = (omega . grad) u_a`` (3D).

        Returns an ``(3, n, n, n)`` array whose component ``a`` is

            S_a = omega_x * d_x u_a + omega_y * d_y u_a + omega_z * d_z u_a.

        This term is the *defining* obstacle of the 3D Navier-Stokes
        global-regularity problem: in 2D it is identically zero (vorticity
        is a scalar perpendicular to the velocity plane and has no
        spatial gradient along the velocity direction), so 2D NS is
        globally regular; in 3D it can in principle amplify enstrophy
        without bound, and whether the viscous term ``nu * Delta omega``
        always tames this amplification is the Clay Millennium Problem
        NS-G5.

        Constantin-Fefferman (1993). The geometric-depletion result
        states that when the vorticity *direction* ``xi = omega / |omega|``
        is uniformly Holder-(1/2) continuous in regions where ``|omega|``
        is large, the projection of the stretching term onto ``omega``
        decays as ``|omega|^2 * sin(theta)`` where ``theta`` is the angle
        between ``omega`` and the eigenvector of the strain tensor; the
        present routine returns the full vector field so that downstream
        diagnostics can compute alignment angles and effective stretching
        rates.
        """
        if self.dimension != 3:
            raise NotImplementedError(
                "vortex_stretching_field() requires self.dimension == 3"
            )
        n = int(self.graph.graph["resolution"])
        h = self._spacing
        omega = self.vorticity_3d()
        comps = (
            self._component_grid_3d(0, n),
            self._component_grid_3d(1, n),
            self._component_grid_3d(2, n),
        )

        def d_axis(arr: np.ndarray, axis: int) -> np.ndarray:
            return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (
                2.0 * h
            )

        stretch = np.empty((3, n, n, n), dtype=float)
        for a in range(3):
            ua = comps[a]
            stretch[a] = (
                omega[0] * d_axis(ua, 0)
                + omega[1] * d_axis(ua, 1)
                + omega[2] * d_axis(ua, 2)
            )
        return stretch

    def stretching_production(self) -> float:
        """Volume integral of ``omega . [(omega . grad) u]`` (3D).

        This is the production term in the enstrophy budget

            d/dt [ (1/2) integral |omega|^2 dV ]
                = integral omega . (omega . grad) u  dV
                  - nu * integral |grad omega|^2 dV.

        A positive value means stretching is *injecting* enstrophy
        (vorticity amplification); a negative value means folding /
        depletion. Global regularity in 3D would follow if this
        production were uniformly dominated by the viscous dissipation
        in the enstrophy budget for all time, which is precisely the
        Clay open question NS-G5.
        """
        if self.dimension != 3:
            raise NotImplementedError(
                "stretching_production() requires self.dimension == 3"
            )
        omega = self.vorticity_3d()
        stretch = self.vortex_stretching_field()
        h3 = self._spacing**3
        prod = omega[0] * stretch[0] + omega[1] * stretch[1] + omega[2] * stretch[2]
        return float(np.sum(prod)) * h3


__all__ = [
    "TNFRNavierStokesOperator",
    "build_torus_graph",
    "build_torus_graph_3d",
    "taylor_green_initial_condition",
    "taylor_green_initial_condition_3d",
]
