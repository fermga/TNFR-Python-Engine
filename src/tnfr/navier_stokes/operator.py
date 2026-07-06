"""TNFR-Navier-Stokes — faithful spectral core (re-founded 2026-07).

The previous program (diffusive-face enstrophy budget, N1-N14 milestone
accretion) was retired.  This is the re-founding on the current paradigm.

Honest two-face reading (physics-first, no forced analogy)
----------------------------------------------------------
Incompressible NS vorticity obeys ``d_t omega = (omega.grad)u + nu*Lap omega``.
This is **first order in time**, so its *linear* part lives on the **diffusive
(over-damped) face** of the nodal dynamics -- exactly the nodal equation
``d_t EPI = nu_f * dNFR`` with ``nu_f <-> nu`` and ``dNFR = -L_rw * K_phi``
(the vorticity is the phase-curvature field ``K_phi``; the viscous term IS the
canonical graph diffusion / IL coherence stabiliser).  The **conservative /
inertial** character of NS -- the energy-conserving Euler cascade where any
blow-up would live -- is entirely in the **nonlinear** stretching source
``(omega.grad)u`` (the VAL destabiliser).  So, unlike oscillatory data (EEG,
whose *linear* neural dynamics are under-damped), NS is linearly diffusive and
its blow-up threat is a **nonlinear K_phi cascade** as ``nu -> 0`` (Re -> inf).

Field dictionary (canonical)
----------------------------
    velocity u_a            <->  per-component phase field phi^(a)
    vorticity omega=curl u  <->  K_phi per component
    pressure p              <->  Phi_s (Leray/incompressibility multiplier)
    viscosity nu            <->  nu_f (diffusive-face structural frequency)
    enstrophy ||omega||^2   <->  sum K_phi^2   (the conserved-pressure energy)

This module provides a faithful, lean pseudo-spectral 3D NS integrator
(rotational form, exact Leray projection, integrating-factor RK2) plus the
periodic torus graph and Taylor-Green initial condition used by the emergent
geometry read-outs.

Honest scope: this closes NOTHING.  Global regularity of 3D incompressible
Navier-Stokes (the Clay problem) stays OPEN.  The re-founding gives the honest
canonical *language* (the nonlinear K_phi cascade on the diffusive face) and a
measurement of where the wall sits (``conservative_face.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # networkx is a hard engine dependency; guarded for isolated imports
    import networkx as nx
except Exception:  # pragma: no cover - networkx always present in the engine
    nx = None  # type: ignore

__all__ = [
    "TNFRNavierStokes",
    "build_torus_graph_3d",
    "taylor_green_initial_condition_3d",
]


# ---------------------------------------------------------------------------
# Emergent geometry helpers (periodic 3-torus): the graph whose L_rw modes are
# the Fourier basis -- the natural NS structural geometry.
# ---------------------------------------------------------------------------
def build_torus_graph_3d(n: int) -> Any:
    r"""Periodic 3D torus grid graph on ``n**3`` nodes (h = 2*pi/n).

    Node ``i = ix*n^2 + iy*n + iz`` is linked to its six periodic neighbours.
    The graph is vertex-transitive (circulant), so its canonical structural
    eigenmodes (``structural_eigenmodes`` / L_rw) are the discrete Fourier
    modes -- the emergent geometry coincides with the natural NS Fourier basis.
    """
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx is required for build_torus_graph_3d")
    if n < 2:
        raise ValueError("n must be >= 2")
    g = nx.Graph()
    g.add_nodes_from(range(n**3))

    def idx(ix: int, iy: int, iz: int) -> int:
        return (ix % n) * n * n + (iy % n) * n + (iz % n)

    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                a = idx(ix, iy, iz)
                g.add_edge(a, idx(ix + 1, iy, iz))
                g.add_edge(a, idx(ix, iy + 1, iz))
                g.add_edge(a, idx(ix, iy, iz + 1))
    g.graph["n"] = int(n)
    g.graph["ndim"] = 3
    return g


def taylor_green_initial_condition_3d(
    graph: Any, amplitude: float = 1.0
) -> tuple[Any, Any, Any]:
    r"""Classic 3D Taylor-Green velocity in ``build_torus_graph_3d`` node order.

    ``u = A sin x cos y cos z``, ``v = -A cos x sin y cos z``, ``w = 0`` with
    ``x = 2*pi*ix/n``.  Returns three flat arrays aligned with ``list(G.nodes)``.
    """
    n = int(graph.graph["n"])
    i = np.arange(n**3)
    ix = i // (n * n)
    iy = (i // n) % n
    iz = i % n
    x = 2.0 * np.pi * ix / n
    y = 2.0 * np.pi * iy / n
    z = 2.0 * np.pi * iz / n
    a = float(amplitude)
    u = a * np.sin(x) * np.cos(y) * np.cos(z)
    v = -a * np.cos(x) * np.sin(y) * np.cos(z)
    w = np.zeros_like(u)
    return u, v, w


# ---------------------------------------------------------------------------
# Faithful pseudo-spectral 3D incompressible Navier-Stokes integrator.
# ---------------------------------------------------------------------------
class TNFRNavierStokes:
    r"""Lean pseudo-spectral solver for 3D incompressible Navier-Stokes.

    Rotational form ``d_t u = u x omega - grad(P) + nu Lap u`` with exact
    spectral Leray projection (``P(k) = I - k k^T/|k|^2``), 2/3-rule
    dealiasing and an integrating-factor RK2 step (exact viscous propagator
    ``exp(-nu |k|^2 dt)`` times an explicit midpoint for the nonlinear term).

    All fields live on the periodic box ``[0, 2*pi)^3`` at resolution ``n``.
    ``viscosity`` is the canonical diffusive-face ``nu_f`` (see module docstring).
    """

    def __init__(self, n: int, viscosity: float, amplitude: float = 1.0) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = int(n)
        self.nu = float(viscosity)
        k1 = np.fft.fftfreq(self.n, d=1.0 / self.n)  # integer wavenumbers
        self.kx, self.ky, self.kz = np.meshgrid(k1, k1, k1, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self._k2nz = np.where(self.k2 == 0.0, 1.0, self.k2)
        kmax = self.n // 3  # 2/3 dealiasing
        self._mask = (
            (np.abs(self.kx) <= kmax)
            & (np.abs(self.ky) <= kmax)
            & (np.abs(self.kz) <= kmax)
        )
        self.set_taylor_green(amplitude)

    # -- initial conditions --------------------------------------------------
    def set_taylor_green(self, amplitude: float = 1.0) -> None:
        """Reset the state to the 3D Taylor-Green vortex (divergence-free)."""
        n = self.n
        c = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x, y, z = np.meshgrid(c, c, c, indexing="ij")
        a = float(amplitude)
        u = a * np.sin(x) * np.cos(y) * np.cos(z)
        v = -a * np.cos(x) * np.sin(y) * np.cos(z)
        w = np.zeros_like(u)
        self.u_hat = np.stack(
            [np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)]
        )
        self._project()

    # -- incompressibility ---------------------------------------------------
    def _project(self) -> None:
        """Exact spectral Leray-Helmholtz projection onto div-free fields."""
        div = (
            self.u_hat[0] * self.kx
            + self.u_hat[1] * self.ky
            + self.u_hat[2] * self.kz
        )
        self.u_hat[0] -= self.kx * div / self._k2nz
        self.u_hat[1] -= self.ky * div / self._k2nz
        self.u_hat[2] -= self.kz * div / self._k2nz

    def _vorticity_hat(self, u_hat: Any) -> Any:
        """Spectral vorticity omega_hat = i k x u_hat."""
        wx = 1j * (self.ky * u_hat[2] - self.kz * u_hat[1])
        wy = 1j * (self.kz * u_hat[0] - self.kx * u_hat[2])
        wz = 1j * (self.kx * u_hat[1] - self.ky * u_hat[0])
        return np.stack([wx, wy, wz])

    def _nonlinear(self, u_hat: Any) -> Any:
        """Projected rotational nonlinear term P[FFT(u x omega)]."""
        u = [np.fft.ifftn(u_hat[a]).real for a in range(3)]
        w_hat = self._vorticity_hat(u_hat)
        w = [np.fft.ifftn(w_hat[a]).real for a in range(3)]
        nx_ = u[1] * w[2] - u[2] * w[1]
        ny_ = u[2] * w[0] - u[0] * w[2]
        nz_ = u[0] * w[1] - u[1] * w[0]
        n_hat = np.stack(
            [np.fft.fftn(nx_), np.fft.fftn(ny_), np.fft.fftn(nz_)]
        )
        n_hat *= self._mask
        div = (
            n_hat[0] * self.kx + n_hat[1] * self.ky + n_hat[2] * self.kz
        )
        n_hat[0] -= self.kx * div / self._k2nz
        n_hat[1] -= self.ky * div / self._k2nz
        n_hat[2] -= self.kz * div / self._k2nz
        return n_hat

    def step(self, dt: float) -> None:
        """Advance one integrating-factor RK2 (midpoint) step."""
        e_full = np.exp(-self.nu * self.k2 * dt)
        e_half = np.exp(-self.nu * self.k2 * dt * 0.5)
        n1 = self._nonlinear(self.u_hat)
        u_mid = e_half * (self.u_hat + 0.5 * dt * n1)
        n2 = self._nonlinear(u_mid)
        self.u_hat = e_full * self.u_hat + dt * e_half * n2
        self._project()

    # -- read-outs (physical space) -----------------------------------------
    def velocity(self) -> tuple[Any, Any, Any]:
        """Physical-space velocity components (real arrays of shape n^3)."""
        return tuple(np.fft.ifftn(self.u_hat[a]).real for a in range(3))

    def vorticity(self) -> tuple[Any, Any, Any]:
        """Physical-space vorticity components (K_phi per component)."""
        w_hat = self._vorticity_hat(self.u_hat)
        return tuple(np.fft.ifftn(w_hat[a]).real for a in range(3))

    def energy(self) -> float:
        """Kinetic energy density (1/2)<|u|^2> (box mean)."""
        u = self.velocity()
        return 0.5 * float(np.mean(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))

    def enstrophy(self) -> float:
        """Enstrophy density (1/2)<|omega|^2> = (1/2)<sum K_phi^2>."""
        w = self.vorticity()
        return 0.5 * float(np.mean(w[0] ** 2 + w[1] ** 2 + w[2] ** 2))

    def divergence_sup(self) -> float:
        """Max |div u| (should stay at round-off after projection)."""
        div_hat = (
            self.u_hat[0] * self.kx
            + self.u_hat[1] * self.ky
            + self.u_hat[2] * self.kz
        ) * 1j
        return float(np.max(np.abs(np.fft.ifftn(div_hat).real)))

    def stretching_production(self) -> float:
        """Vortex-stretching production integral <omega . (omega.grad) u>.

        The nonlinear (VAL) source of enstrophy; identically zero for
        two-dimensional (z-independent) data and nonzero in genuine 3D.
        """
        u = self.velocity()
        w = self.vorticity()
        prod = 0.0
        for a in range(3):
            # (omega . grad) u_a  via spectral derivatives
            grad_a = [
                np.fft.ifftn(1j * kk * self.u_hat[a]).real
                for kk in (self.kx, self.ky, self.kz)
            ]
            wgrad_ua = w[0] * grad_a[0] + w[1] * grad_a[1] + w[2] * grad_a[2]
            prod += float(np.mean(w[a] * wgrad_ua))
        return prod

    def vorticity_magnitude_nodes(self) -> Any:
        """|omega| sampled in ``build_torus_graph_3d`` node order.

        Bridges the flow to the emergent-geometry read-outs: the returned
        flat array (length n^3) is aligned with ``list(G.nodes)`` so it can be
        written onto the torus graph as the ``K_phi`` magnitude field.
        """
        w = self.vorticity()
        mag = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
        return mag.reshape(-1)
