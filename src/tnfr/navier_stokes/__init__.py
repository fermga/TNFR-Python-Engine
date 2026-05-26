"""TNFR-Navier-Stokes program.

Attacks the Clay Millennium Problem on existence and smoothness of solutions
to the 3D incompressible Navier-Stokes equations through the TNFR structural
translation documented in ``theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md``.

Field dictionary (canonical):

    velocity component u_a       <->  per-component phase field phi^(a)
    vorticity omega = curl u     <->  K_phi per component
    pressure p                   <->  Phi_s (Lagrange multiplier)
    kinematic viscosity nu       <->  U2 stabiliser strength
    incompressibility div u = 0  <->  sum_a grad_a phi^(a) = 0 (INCOMP, reserved)
    kinetic energy ||u||^2 / 2   <->  tetrad energy density E
    enstrophy ||omega||^2        <->  sum K_phi^2
    helicity int u . omega       <->  topological charge Q

Sub-modules
-----------
operator
    N2: discrete TNFR-NS operator on periodic grid graphs (linear viscous
    baseline, no advection). Validates exponential decay of the Taylor-Green
    vortex against the analytical rate E(t) = E(0) * exp(-4 nu t) for the
    fundamental mode.

    N3: skew-symmetric advection ``-(u . grad) u`` added via Strang splitting
    + dissipation rate / Leray budget diagnostics. Verifies the discrete
    analogue of the Leray energy inequality
    E(t) + nu * int_0^t ||grad u||^2 dtau <= E(0) for E = (1/2) ||u||_L2^2.
    Pressure (INCOMP) remains held in reserve; divergence drift is tracked
    explicitly and documented as the cost of deferring NS-G2.

    N4: discrete Beale-Kato-Majda criterion via ``vorticity_2d()``,
    ``vorticity_sup_norm()``, ``enstrophy_curl()`` and ``bkm_budget()``.
    Tracks the BKM integral ``int_0^T ||omega||_{L^inf} dtau`` along the
    discrete flow. In 2D this integral stays bounded (no vortex
    stretching), consistent with the classical 2D global-regularity
    result. The same infrastructure will lift to 3D where the open Clay
    question (NS-G5) is whether the integral can diverge in finite time.

    N5: INCOMP operator (Leray-Helmholtz projection) via
    ``project_incompressible()`` and ``pressure_field()``, plus an
    ``incompressible=True`` keyword on ``step()``, ``leray_budget()`` and
    ``bkm_budget()``. Uses pseudo-spectral projection with the *exact*
    central-difference symbol ``S_a = sin(2*pi*m_a/n)/h`` (not the
    spectral ``i k``), so the divergence operator probed by
    ``divergence_residual()`` drops to round-off after every projection.
    Activation on the 2D Taylor-Green benchmark (see ``examples/80_*``)
    closes the N3/N4 INFO caveats: max ||div(t)||_2 ~ 1e-16, energy
    matches ``E(0) exp(-4 nu t)`` within 0.13%, BKM integral matches its
    analytical envelope within 0.61%, vorticity log-slope matches
    ``-2 nu`` within 0.32%. Canonicity status: INCOMP is a *global,
    non-local* projection and cannot be decomposed into nearest-neighbour
    TNFR operators; whether it is the 14th canonical operator or a
    derived projection bound to the incompressibility constraint is held
    as an OPEN structural question, mirroring how the Riemann program
    kept its catalog frozen at 13.

Honest scope
------------
This module does NOT claim a proof or disproof of the Clay statement. Both
directions (global smoothness vs finite-time blow-up) remain OPEN. The program
follows the gap-audit methodology established by the now-paused Riemann
program; see NS-G1 (continuum limit) and NS-G3 (discrete <-> continuum BKM
transfer) in the research notes for the documented obstructions.
"""

from .operator import (
    TNFRNavierStokesOperator,
    build_torus_graph,
    taylor_green_initial_condition,
)

__all__ = [
    "TNFRNavierStokesOperator",
    "build_torus_graph",
    "taylor_green_initial_condition",
]
