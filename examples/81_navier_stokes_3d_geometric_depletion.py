"""
N6: 3D Navier-Stokes — Constantin-Fefferman vortex-stretching diagnostics.

This demo activates the genuine Clay regime for the 3D incompressible
Navier-Stokes equations on a periodic 3-torus. It exercises the full TNFR
infrastructure:

  - 3D graph torus (``build_torus_graph_3d``),
  - 3D Taylor-Green initial condition,
  - skew-symmetric advection ``-(1/2)[(u.grad) phi + grad.(u phi)]``,
  - FFT-diagonalised Crank-Nicolson viscous half-step,
  - pseudo-spectral Leray-Helmholtz projection with the discrete symbol
    ``S_a = sin(2*pi*m_a/n)/h`` (so that the central-difference divergence
    measured by ``divergence_residual`` drops to round-off),
  - BKM enstrophy / vorticity-sup-norm budget,
  - Constantin-Fefferman vortex-stretching field ``(omega . grad) u``
    and its production integral.

HONEST SCOPE
============

This demo does NOT prove global regularity of 3D Navier-Stokes. The Clay
Millennium Problem NS-G5 (existence and smoothness of strong solutions
for all time, starting from smooth divergence-free data of finite energy)
remains OPEN.

What this demo *does*:

  - Validates the 3D incompressible time-stepper on a short horizon,
  - Demonstrates that the divergence is preserved at machine precision,
  - Shows energy/enstrophy monotone non-increasing under viscosity,
  - Confirms that the vortex-stretching field, which is identically zero
    in 2D, is non-trivial in 3D and produces an enstrophy-source term in
    the budget,
  - Provides the discrete infrastructure on which the Constantin-Fefferman
    geometric-depletion mechanism (alignment of ``omega`` with strain
    eigenvectors) can be measured empirically.

What it does NOT do:

  - Prove a uniform bound on the BKM integral
    ``integral_0^T ||omega(.,t)||_infinity dt`` for arbitrary T,
  - Close NS-G5,
  - Establish that the geometric depletion mechanism beats vortex
    stretching globally in time.
"""

from __future__ import annotations

import math

import numpy as np

from tnfr.navier_stokes.operator import (
    TNFRNavierStokesOperator,
    build_torus_graph_3d,
)

# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

N = 12                  # grid resolution per axis (12**3 = 1728 nodes)
VISCOSITY = 0.05
DT = 0.01
STEPS = 100             # T = 1.0
AMPLITUDE = 1.0

# Acceptance thresholds (honest, conservative, machine-relative)
TOL_DIV = 1e-8
TOL_ENERGY_GROWTH = 1e-10
TOL_ENSTROPHY_GROWTH = 1e-10


def main() -> int:
    print("=" * 72)
    print("N6: 3D Navier-Stokes (Taylor-Green) + Constantin-Fefferman stretching")
    print("=" * 72)
    print(f"  Grid: {N}**3 = {N**3} nodes,  h = 2*pi/{N} = {2*math.pi/N:.4f}")
    print(f"  Viscosity nu = {VISCOSITY},  dt = {DT},  steps = {STEPS},  T = {DT*STEPS}")
    print()

    # ---- build operator -----------------------------------------------------
    G = build_torus_graph_3d(N)
    op = TNFRNavierStokesOperator(graph=G, viscosity=VISCOSITY, dimension=3)
    op.set_taylor_green(AMPLITUDE)

    E0 = op.kinetic_energy()
    Z0 = op.enstrophy_curl()
    div0 = op.divergence_residual()
    omega_inf0 = op.vorticity_sup_norm()
    print(f"  Initial energy   E(0)         = {E0:.6f}")
    print(f"  Initial enstrophy Z(0)        = {Z0:.6f}")
    print(f"  Initial ||omega||_inf         = {omega_inf0:.6f}")
    print(f"  Initial ||div u||             = {div0:.3e}")
    print(f"  Initial stretching production = {op.stretching_production():.3e}")
    print()

    # ---- run BKM budget -----------------------------------------------------
    report = op.bkm_budget(
        dt=DT,
        steps=STEPS,
        advection=True,
        incompressible=True,
    )
    times = np.asarray(report["time"])
    omega_inf = np.asarray(report["vorticity_sup"])
    enstrophy = np.asarray(report["enstrophy"])
    divs = np.asarray(report["divergence"])
    bkm = np.asarray(report["bkm_integral"])

    # Sample energy and stretching at a few checkpoints by running shorter pieces.
    # bkm_budget reports its own trajectory; we sample current state quantities.
    EN = op.kinetic_energy()
    ZN = op.enstrophy_curl()
    stretch_final = op.stretching_production()

    print("Trajectory diagnostics")
    print("-" * 72)
    print(f"  ||omega||_inf  first / last  = {omega_inf[0]:.6f} / {omega_inf[-1]:.6f}")
    print(f"  enstrophy      first / last  = {enstrophy[0]:.6f} / {enstrophy[-1]:.6f}")
    print(f"  max ||div u|| over trajectory = {divs.max():.3e}")
    print(f"  BKM integral  int_0^T ||omega||_inf dt = {bkm[-1]:.6f}")
    print()
    print(f"  Final energy   E(T)             = {EN:.6f}")
    print(f"  Final enstrophy Z(T)            = {ZN:.6f}")
    print(f"  Final stretching production     = {stretch_final:.6f}")
    print()

    # ---- pass / fail --------------------------------------------------------
    c1 = bool(divs.max() <= TOL_DIV)
    c2_E = bool(EN <= E0 + TOL_ENERGY_GROWTH)
    c2_Z = bool(np.all(np.diff(enstrophy) <= TOL_ENSTROPHY_GROWTH))
    c2 = c2_E and c2_Z
    c3 = bool(np.isfinite(bkm[-1]))
    c4 = bool(abs(stretch_final) > 0.0)

    print("Acceptance criteria (honest scope)")
    print("-" * 72)
    print(f"  C1 INCOMP        max ||div|| <= {TOL_DIV:.0e}      : "
          f"{'PASS' if c1 else 'FAIL'}  (got {divs.max():.3e})")
    print(f"  C2 Monotone      E(T) <= E(0) and dZ/dt <= 0       : "
          f"{'PASS' if c2 else 'FAIL'}  (E: {E0:.4f} -> {EN:.4f}; "
          f"Z: {Z0:.4f} -> {ZN:.4f})")
    print(f"  C3 BKM finite    int_0^T ||w||_inf dt finite        : "
          f"{'PASS' if c3 else 'FAIL'}  (got {bkm[-1]:.4f})")
    print(f"  C4 3D non-triv   |stretching production| > 0        : "
          f"{'PASS' if c4 else 'FAIL'}  (got {stretch_final:.3e})")

    n_pass = sum(int(b) for b in (c1, c2, c3, c4))
    print()
    print(f"  Result: {n_pass}/4 PASS")
    print()
    print("Honest-scope coda")
    print("-" * 72)
    print("  N6 validates the 3D incompressible TNFR-Navier-Stokes time-stepper")
    print("  plus the Constantin-Fefferman vortex-stretching diagnostics on a")
    print("  short horizon of classical Taylor-Green data. The vortex-stretching")
    print("  term (omega . grad) u, which vanishes identically in 2D, is the")
    print("  defining obstacle of the 3D global-regularity problem (Clay NS-G5).")
    print("  Observing it numerically and confirming that, on this window, the")
    print("  viscous term controls it (energy/enstrophy monotone, BKM integral")
    print("  finite) does NOT prove global regularity. NS-G5 remains OPEN.")

    return 0 if n_pass == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
