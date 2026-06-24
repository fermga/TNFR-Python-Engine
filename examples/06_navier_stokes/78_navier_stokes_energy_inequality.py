"""N3: discrete Leray energy inequality with advection on.

Verifies the discrete analogue of the Leray weak-solution estimate

    E(t) + nu * integral_0^t ||grad u(tau)||^2 dtau  <=  E(0)

for E = (1/2) ||u||_{L^2}^2 and the 2D Taylor-Green vortex on the periodic
torus, with the non-linear
advection ``-(u . grad) u`` switched on via the skew-symmetric formulation
and Strang splitting. The relevant TNFR canonical translation is documented
in ``theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md`` (NS-N3).

Why this is non-trivial
-----------------------
Adding advection can in principle inject spurious energy through two
mechanisms:

    1. Non-skew discretisations of ``u . grad u`` whose discrete inner
       product against ``u`` does not vanish (numerical energy production).
    2. Compressibility errors: in the absence of an explicit pressure
       projection, the discrete divergence drifts away from zero, and the
       energy identity ``<u . grad u, u> = 0`` only holds for divergence-free
       fields.

The skew-symmetric form
    A_a = - (1/2) [ u_b * d_b phi_a + d_b (u_b * phi_a) ]
annihilates ``<A, phi>`` for ANY field (no divergence-free assumption), so
the discrete kinetic energy is dissipated solely by viscosity, up to the
O(dt^2) splitting error and round-off.

Success criteria (N3)
---------------------
On a 32 x 32 grid, nu = 0.05, dt = 0.01, 200 steps:

    1. Discrete Leray budget
           B(t_n) = E(0) - E(t_n) - integral_0^{t_n} nu ||grad u||^2 dtau
       satisfies ``B(t_n) >= - eps_split`` at every step
       (eps_split ~ a few times O(dt^2) per step is acceptable).
    2. Energy is monotone non-increasing: ``E(t_{n+1}) <= E(t_n) + eps_round``.
    3. With advection ON the measured decay rate differs from the linear
       4 nu prediction (sanity check that the non-linear term is active).

Divergence drift is tracked and reported but NOT a PASS/FAIL criterion at
this milestone: enforcing ``div u = 0`` requires the INCOMP / pressure
projection step, which is held in reserve and will be activated in a later
milestone (likely N5 or N6). Documenting the drift here makes the trade-off
explicit and prevents premature claims of incompressibility.

Honest scope
------------
N3 certifies the discrete energy identity on the periodic torus with
skew-symmetric advection. It does NOT:

    * enforce incompressibility (pressure / INCOMP still reserved),
    * address 3D effects (NS-G5),
    * bear on the Clay statement (finite-time singularity formation in 3D).

It is the prerequisite for N4 (discrete BKM criterion).
"""

from __future__ import annotations

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
    print("TNFR-Navier-Stokes N3 demo: discrete Leray energy inequality")
    print("=" * 72)
    print(f"Grid resolution    : {n} x {n} (periodic)")
    print(f"Viscosity nu       : {viscosity}")
    print(f"Time step dt       : {dt}")
    print(f"Number of steps    : {steps}")
    print(f"Amplitude U_0      : {amplitude}")
    print(f"Advection          : ON (skew-symmetric, Strang split)")
    print()

    G = build_torus_graph(n)
    op = TNFRNavierStokesOperator(graph=G, viscosity=viscosity, dimension=2)
    op.set_taylor_green(amplitude=amplitude)

    initial_div = op.divergence_residual()
    initial_energy = op.kinetic_energy()
    print("Initial state")
    print("-" * 72)
    print(f"divergence (L2)            : {initial_div:.3e}")
    print(f"kinetic energy E(0)        : {initial_energy:.6f}")
    print(f"dissipation rate D(0)      : {op.dissipation_rate():.6f}")
    print()

    budget = op.leray_budget(dt=dt, steps=steps, advection=True)
    times = budget["time"]
    energy = budget["energy"]
    dissipation = budget["dissipation"]
    cum_dissipated = budget["cumulative_dissipated"]
    cum_budget = budget["cumulative_budget"]
    divergence = budget["divergence"]

    # Selected snapshots for the report
    print("Leray budget over time (selected steps)")
    print("-" * 72)
    print(
        f"{'t':>8} {'E(t)':>12} {'2nu*|gradU|^2':>14} "
        f"{'cum.diss.':>12} {'budget B(t)':>14} {'div':>10}"
    )
    idx = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    for k in idx:
        print(
            f"{times[k]:8.4f} {energy[k]:12.6f} {dissipation[k]:14.6f} "
            f"{cum_dissipated[k]:12.6f} {cum_budget[k]:14.3e} "
            f"{divergence[k]:10.2e}"
        )
    print()

    # Diagnostics
    min_budget = float(np.min(cum_budget))
    max_div = float(np.max(np.abs(divergence)))
    energy_increments = np.diff(energy)
    max_energy_jump = float(np.max(energy_increments))

    # Tolerances. Strang splitting is O(dt^2) per step => O(dt) accumulated.
    # Round-off ~ machine eps * E(0) per step => O(steps * eps).
    eps_split = 5.0 * dt**2 * steps  # generous bound on splitting drift
    eps_round = 1e-10

    # Sanity: with advection ON the effective decay differs from 4 nu.
    # Fit ln E vs t on the second half (advection-modified transient settled).
    half = steps // 2
    log_e = np.log(energy[half:])
    slope, _ = np.polyfit(times[half:], log_e, 1)
    measured_rate = -slope
    linear_rate = 4.0 * viscosity
    rate_diff = abs(measured_rate - linear_rate)

    print("Diagnostics summary")
    print("-" * 72)
    print(
        f"min cumulative budget B    : {min_budget:.3e}  (target >= {-eps_split:.2e})"
    )
    print(
        f"max energy jump dE         : {max_energy_jump:.3e}  (target <= {eps_round:.1e})"
    )
    print(f"linear 4 nu rate           : {linear_rate:.6f}")
    print(f"measured rate (advection)  : {measured_rate:.6f}")
    print(f"|measured - linear|        : {rate_diff:.3e}  (sanity: should be > 1e-4)")
    print(
        f"max |divergence| (INFO)    : {max_div:.3e}  (no PASS/FAIL - INCOMP reserved)"
    )
    print()

    pass1 = min_budget >= -eps_split
    pass2 = max_energy_jump <= eps_round
    pass3 = rate_diff > 1e-4

    print("Success criteria (N3)")
    print("-" * 72)
    print(f"  1. Discrete Leray budget B(t) >= -eps  : {'PASS' if pass1 else 'FAIL'}")
    print(f"  2. Energy monotone non-increasing      : {'PASS' if pass2 else 'FAIL'}")
    print(f"  3. Advection actively modifies flow    : {'PASS' if pass3 else 'FAIL'}")
    print()

    all_pass = pass1 and pass2 and pass3
    if all_pass:
        print("[N3 PASS] Discrete Leray energy inequality certified for the")
        print("          2D Taylor-Green flow with skew-symmetric advection.")
        print("          Divergence drift is the expected cost of deferring INCOMP.")
        print()
        print("Next: N4 (discrete BKM criterion + enstrophy growth tracking).")
        return 0

    print("[N3 FAIL] One or more discrete energy criteria violated.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
