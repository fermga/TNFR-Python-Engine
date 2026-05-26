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
