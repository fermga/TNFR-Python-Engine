"""N4: discrete Beale-Kato-Majda criterion on the 2D periodic torus.

Tracks the BKM integral

    I_BKM(T) = integral_0^T || omega(tau) ||_{L^inf} dtau

along the discrete TNFR-Navier-Stokes flow with skew-symmetric advection
switched on (Strang splitting from N3). The BKM theorem (Beale, Kato &
Majda, 1984) asserts that a smooth solution of the incompressible 3D
Navier-Stokes equations on ``[0, T*)`` extends past ``T*`` if and only
if ``I_BKM(T*) < infinity``. In 2D the vorticity equation has no
stretching term ``(omega . grad) u``, so the BKM integral is bounded
for all time and the flow is globally regular -- this is the
well-known 2D global-regularity result.

The point of this demo is therefore NOT to ``prove`` 2D regularity
(it is classical) but to certify that the discrete TNFR operator is
consistent with it: ``I_BKM`` must stay bounded, enstrophy must decay
monotonically (no spurious enstrophy production), and the time series
must follow the analytical envelope

    || omega(t) ||_{L^inf} ~ || omega(0) ||_{L^inf} * exp(- 2 nu t)

for the Taylor-Green initial condition, whose vorticity is the single
Laplacian eigenmode ``omega(x, y, 0) = 2 sin(x) sin(y)``.

Success criteria (N4)
---------------------
On a 32 x 32 grid, nu = 0.05, dt = 0.01, 200 steps (T = 2):

    1. BKM integral stays FINITE: ``I_BKM(T) <= I_trivial = || omega(0) ||_inf * T``
       (the trivial upper bound). The exact analytical envelope
       ``I_an(T) = (omega0 / (2 nu)) * (1 - exp(-2 nu T))`` is reported
       as INFO only -- it assumes ``div u = 0`` (INCOMP), which is
       still held in reserve.
    2. Enstrophy is monotone non-increasing:
       ``Omega(t_{n+1}) <= Omega(t_n) + eps_round``.
    3. Vorticity sup-norm does NOT exceed its initial value:
       ``max_n || omega(t_n) ||_inf <= || omega(0) ||_inf * (1 + eps_tol)``.
       In 2D, vorticity is transported + diffused with NO stretching, so
       its sup-norm cannot grow (modulo finite-difference / splitting
       error and the compressibility error from missing INCOMP).

The exact analytical decay rate ``slope = -2 nu`` requires ``div u = 0``
and is therefore not enforced as a PASS/FAIL criterion at this milestone;
the discrepancy between the fitted slope and ``-2 nu`` is reported as
INFO and quantifies the cost of deferring INCOMP.

Honest scope
------------
N4 certifies the BKM infrastructure on 2D, where the answer is known.
It does NOT:

    * address 3D vortex stretching (NS-G5),
    * settle the Clay statement (3D global regularity vs finite-time
      blow-up remains OPEN in both directions),
    * enforce incompressibility (INCOMP still reserved; divergence
      drift is tracked as INFO, as in N3).

It is the prerequisite for N5 (geometric depletion / Constantin-Fefferman
vortex direction estimates) and, eventually, for any 3D extension.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from tnfr.navier_stokes import TNFRNavierStokesOperator, build_torus_graph


def main() -> int:
    n = 32
    viscosity = 0.05
    dt = 0.01
    steps = 200
    amplitude = 1.0

    print("=" * 72)
    print("TNFR-Navier-Stokes N4 demo: discrete BKM criterion (2D)")
    print("=" * 72)
    print(f"Grid resolution    : {n} x {n} (periodic)")
    print(f"Viscosity nu       : {viscosity}")
    print(f"Time step dt       : {dt}")
    print(f"Number of steps    : {steps}")
    print(f"Amplitude U_0      : {amplitude}")
    print(f"Advection          : ON (skew-symmetric, Strang split)")
    print()

    graph = build_torus_graph(n)
    operator = TNFRNavierStokesOperator(graph, viscosity=viscosity, dimension=2)
    operator.set_taylor_green(amplitude=amplitude)

    # Analytical reference for Taylor-Green vorticity
    # omega(x,y,t) = 2 * amplitude * sin(x) * sin(y) * exp(-2 nu t)
    omega0_analytical = 2.0 * amplitude
    omega0_numeric = operator.vorticity_sup_norm()
    enstrophy0_numeric = operator.enstrophy_curl()
    # Analytical enstrophy: (1/2) integral [2 sin x sin y]^2 dA = (1/2) * 4 * (pi^2)
    # = 2 pi^2 ~ 19.7392 at t = 0
    enstrophy0_analytical = 2.0 * (math.pi ** 2) * amplitude ** 2

    print("Initial state")
    print("-" * 72)
    print(f"|| omega(0) ||_inf  (analytical) : {omega0_analytical:.6f}")
    print(f"|| omega(0) ||_inf  (numerical)  : {omega0_numeric:.6f}")
    print(f"Omega(0)            (analytical) : {enstrophy0_analytical:.6f}")
    print(f"Omega(0)            (numerical)  : {enstrophy0_numeric:.6f}")
    print()

    history = operator.bkm_budget(dt, steps, advection=True)
    time = history["time"]
    vort_sup = history["vorticity_sup"]
    enstrophy_hist = history["enstrophy"]
    bkm_int = history["bkm_integral"]
    div_hist = history["divergence"]
    T = float(time[-1])

    # Analytical BKM integral for the linear 2D heat equation envelope:
    #   I_BKM(T) = (omega0 / (2 nu)) * (1 - exp(-2 nu T))
    bkm_analytical = (omega0_analytical / (2.0 * viscosity)) * (
        1.0 - math.exp(-2.0 * viscosity * T)
    )

    print("BKM budget over time (selected steps)")
    print("-" * 72)
    print(
        f"{'t':>8} {'||omega||_inf':>14} {'Omega(t)':>12} "
        f"{'I_BKM(t)':>12} {'div':>10}"
    )
    sample_steps = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    for k in sample_steps:
        print(
            f"{time[k]:8.4f} {vort_sup[k]:14.6f} {enstrophy_hist[k]:12.6f} "
            f"{bkm_int[k]:12.6f} {div_hist[k]:10.2e}"
        )
    print()

    # ------------------------------------------------------------------
    # Criterion 1: BKM integral stays finite (bounded by trivial upper bound)
    # ------------------------------------------------------------------
    bkm_final = float(bkm_int[-1])
    bkm_trivial_bound = omega0_numeric * T  # constant vorticity envelope
    bkm_rel_to_analytical = abs(bkm_final - bkm_analytical) / bkm_analytical

    # ------------------------------------------------------------------
    # Criterion 2: enstrophy monotone non-increasing
    # ------------------------------------------------------------------
    eps_round = 1e-10
    enstrophy_diffs = np.diff(enstrophy_hist)
    max_enstrophy_jump = float(np.max(enstrophy_diffs))

    # ------------------------------------------------------------------
    # Criterion 3: vorticity sup-norm bounded by its initial value
    # ------------------------------------------------------------------
    eps_tol = 5e-2  # allow ~5% slack for splitting + compressibility error
    max_vort_sup = float(np.max(vort_sup))
    vort_sup_bound = omega0_numeric * (1.0 + eps_tol)

    # INFO: fitted decay rate (would equal -2 nu under INCOMP)
    log_vort = np.log(vort_sup[1:])
    t_fit = time[1:]
    slope, _ = np.polyfit(t_fit, log_vort, 1)
    rate_target = -2.0 * viscosity

    print("Diagnostics summary")
    print("-" * 72)
    print(f"I_BKM(T) numerical          : {bkm_final:.6f}")
    print(f"I_BKM trivial upper bound   : {bkm_trivial_bound:.6f}  (omega0 * T)")
    print(f"I_BKM analytical (INCOMP)   : {bkm_analytical:.6f}  (INFO)")
    print(f"|I_num - I_an|/I_an         : {bkm_rel_to_analytical:.3e}  (INFO; INCOMP reserved)")
    print(f"max enstrophy jump dOmega   : {max_enstrophy_jump:.3e}  (target <= {eps_round:.1e})")
    print(f"max ||omega||_inf in [0,T]  : {max_vort_sup:.6f}")
    print(f"||omega(0)||_inf * (1+eps)  : {vort_sup_bound:.6f}  (target upper bound)")
    print(f"fitted decay slope          : {slope:.6f}  (INFO)")
    print(f"analytical slope (-2 nu)    : {rate_target:.6f}  (INFO; INCOMP reserved)")
    print(f"max |divergence| (INFO)     : {float(np.max(np.abs(div_hist))):.3e}  (no PASS/FAIL - INCOMP reserved)")
    print()

    pass1 = bkm_final <= bkm_trivial_bound
    pass2 = max_enstrophy_jump <= eps_round
    pass3 = max_vort_sup <= vort_sup_bound

    print("Success criteria (N4)")
    print("-" * 72)
    print(f"  1. BKM integral finite (<= omega0 * T)      : {'PASS' if pass1 else 'FAIL'}")
    print(f"  2. Enstrophy monotone non-increasing        : {'PASS' if pass2 else 'FAIL'}")
    print(f"  3. ||omega||_inf <= ||omega(0)||_inf (+5%)  : {'PASS' if pass3 else 'FAIL'}")
    print()

    all_pass = pass1 and pass2 and pass3
    if all_pass:
        print("[N4 PASS] Discrete BKM infrastructure certified for the 2D")
        print("          Taylor-Green flow: ||omega||_inf stays bounded by")
        print("          its initial value, enstrophy decays monotonically,")
        print("          and the BKM integral stays well below its trivial")
        print("          upper bound. No vortex stretching occurs in 2D.")
        print()
        print("Note: the gap between the numerical BKM integral and the")
        print("analytical INCOMP envelope (omega0 / 2nu) * (1 - exp(-2 nu T))")
        print("quantifies the cost of deferring INCOMP (pressure projection).")
        print("This does NOT bear on the 3D Clay problem, where vortex")
        print("stretching (NS-G5) is the open obstruction.")
        print()
        print("Next: N5 (geometric depletion / Constantin-Fefferman vortex")
        print("      direction estimates -- likely INCOMP activation point).")
        return 0

    print("[N4 FAIL] One or more BKM criteria did not pass; see diagnostics.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
