"""Example 80 (N5): INCOMP activation on the 2D Taylor-Green vortex.

This example demonstrates the milestone N5 of the TNFR-Navier-Stokes
program: activation of the **INCOMP** operator (Leray-Helmholtz projection
on the periodic torus) and re-running of the Taylor-Green decay benchmark
to tighten the agreement with the closed-form analytical solution.

Honest scope
------------

Pre-INCOMP baseline (N3 / N4):

* divergence drifts to ``max|div| ~ 1`` after a few hundred advection
  steps (no constraint enforced)
* energy decay rate fitted to ``0.70 * (...)`` of the analytical
  ``4 nu`` envelope (the rest is energy bled into the longitudinal /
  gradient mode that should not exist for the incompressible flow)
* BKM integral falls about 26% short of the analytical envelope
  ``omega_0 / (2 nu) * (1 - exp(-2 nu T))``

This demo activates INCOMP after every advection sub-step and verifies
that all three INFO gaps close simultaneously:

1. ``max|div(t)| <= 1e-8`` throughout the run (machine epsilon at the
   resolution n=32; INCOMP actually enforces ``div u = 0`` to round-off,
   not just bounds it)
2. energy ``E(t)`` follows the analytical ``E(0) * exp(-4 nu t)`` decay
   within ~5%
3. BKM integral matches the analytical envelope within ~5%
4. vorticity sup-norm decays at the analytical rate ``-2 nu`` within ~5%

Failure of any of these would indicate that the INCOMP projection has
been mis-derived, that the discrete symbols do not match the central-
difference operator used elsewhere in the operator, or that some other
sub-step (viscous CN) is breaking incompressibility unexpectedly.

This milestone closes the "INCOMP reserved" caveat for 2D Taylor-Green
and validates the N3/N4 diagnostic interpretation: the documented INFO
gaps were genuinely INCOMP-induced (i.e. attributable to the unenforced
incompressibility constraint), not to advection or viscous discretisation
artefacts.

Scope of what is NOT closed
---------------------------

N5 does **not** address the open Clay obstruction (NS-G5, 3D vortex
stretching). The 2D vorticity equation has no stretching term, so global
regularity is trivial; the milestone certifies discrete consistency of
the prerequisite infrastructure, not closure of the 3D blow-up question.
A genuine 3D Constantin-Fefferman geometric-depletion analysis is
deferred to N6 (requires extending advection and vorticity to
``dimension == 3``).
"""

from __future__ import annotations

import math

import numpy as np

from tnfr.navier_stokes.operator import TNFRNavierStokesOperator, build_torus_graph


def _format_pass(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def main() -> None:
    # Reproducible setup (matches demos 78/79 baseline parameters)
    n = 32
    viscosity = 0.05
    dt = 0.01
    steps = 200
    amplitude = 1.0

    graph = build_torus_graph(n)
    op = TNFRNavierStokesOperator(graph, viscosity=viscosity, dimension=2)
    op.set_taylor_green(amplitude=amplitude)

    # Sanity: initial Taylor-Green is analytically divergence-free
    initial_div = op.divergence_residual()
    initial_energy = op.kinetic_energy()
    initial_vort_sup = op.vorticity_sup_norm()
    initial_enstrophy = op.enstrophy_curl()

    print("=" * 70)
    print("Example 80 (N5): INCOMP activation - 2D Taylor-Green vortex")
    print("=" * 70)
    print(f"Resolution    n = {n}    (spacing h = 2*pi/n = {2*math.pi/n:.6f})")
    print(f"Viscosity   nu = {viscosity}")
    print(f"Time step   dt = {dt}    steps = {steps}    T = {dt*steps}")
    print(f"Amplitude   A  = {amplitude}")
    print()
    print("Initial state:")
    print(f"  E(0)            = {initial_energy:.6f}")
    print(f"  ||omega(0)||inf = {initial_vort_sup:.6f}  (analytical = 2.000)")
    print(
        f"  Omega(0)        = {initial_enstrophy:.6f}  (analytical = 2*pi^2 = {2*math.pi**2:.6f})"
    )
    print(f"  ||div(0)||_2    = {initial_div:.3e}    (must be ~ round-off)")
    print()

    # --- Run with INCOMP active ---
    bkm = op.bkm_budget(dt, steps, advection=True, incompressible=True)
    # Also need energy history; re-run leray_budget to capture E(t) under
    # the same INCOMP-on configuration. Two separate runs avoid coupling
    # the BKM and Leray accumulators, at the cost of doubling work for a
    # tiny example.
    op2 = TNFRNavierStokesOperator(graph, viscosity=viscosity, dimension=2)
    op2.set_taylor_green(amplitude=amplitude)
    leray = op2.leray_budget(dt, steps, advection=True, incompressible=True)

    time = bkm["time"]
    vort_sup = bkm["vorticity_sup"]
    enstrophy = bkm["enstrophy"]
    bkm_int = bkm["bkm_integral"]
    div_hist = bkm["divergence"]
    energy = leray["energy"]

    # Analytical envelopes (continuum Taylor-Green at unit wavenumber)
    # u(x,y,t) =  A sin(x) cos(y) exp(-2 nu t)
    # v(x,y,t) = -A cos(x) sin(y) exp(-2 nu t)
    # omega(x,y,t) = -2 A sin(x) sin(y) exp(-2 nu t) ... NO, the curl of
    #   (sin x cos y, -cos x sin y) is:  d/dx(-cos x sin y) - d/dy(sin x cos y)
    #   = sin x sin y - (-sin x sin y) = 2 sin x sin y
    # so |omega|_inf = 2 A exp(-2 nu t)  and  Omega(t) = 2 pi^2 A^2 exp(-4 nu t)
    # E(t) = E(0) exp(-4 nu t)
    energy_analytical = initial_energy * np.exp(-4.0 * viscosity * time)
    vort_analytical = 2.0 * amplitude * np.exp(-2.0 * viscosity * time)
    enstrophy_analytical = (
        2.0 * math.pi**2 * amplitude**2 * np.exp(-4.0 * viscosity * time)
    )
    T = time[-1]
    bkm_envelope = (
        (2.0 * amplitude) / (2.0 * viscosity) * (1.0 - math.exp(-2.0 * viscosity * T))
    )

    # --- PASS criteria ---
    max_div = float(np.max(div_hist))
    div_pass = max_div <= 1.0e-8

    energy_rel_err = float(
        np.max(np.abs(energy - energy_analytical) / energy_analytical)
    )
    energy_pass = energy_rel_err <= 0.05

    bkm_final = float(bkm_int[-1])
    bkm_rel_err = abs(bkm_final - bkm_envelope) / bkm_envelope
    bkm_pass = bkm_rel_err <= 0.05

    # Fit log-vorticity slope (linear regression, time > 0)
    mask = time > 0.0
    log_vort = np.log(np.maximum(vort_sup[mask], 1.0e-30))
    t_pos = time[mask]
    slope, _ = np.polyfit(t_pos, log_vort, 1)
    slope_analytical = -2.0 * viscosity
    slope_rel_err = abs(slope - slope_analytical) / abs(slope_analytical)
    vort_pass = slope_rel_err <= 0.05

    print("Run summary (INCOMP ON):")
    print(f"  max ||div(t)||_2          = {max_div:.3e}")
    print(f"  E(T) measured             = {energy[-1]:.6f}")
    print(f"  E(T) analytical           = {energy_analytical[-1]:.6f}")
    print(f"  max |E - E_an| / E_an     = {energy_rel_err:.3%}")
    print(f"  I_BKM(T) measured         = {bkm_final:.6f}")
    print(f"  I_BKM(T) analytical env.  = {bkm_envelope:.6f}")
    print(f"  |I_BKM - env| / env       = {bkm_rel_err:.3%}")
    print(f"  ||omega(T)||inf measured  = {vort_sup[-1]:.6f}")
    print(f"  ||omega(T)||inf analyt.   = {vort_analytical[-1]:.6f}")
    print(f"  fitted log-slope          = {slope:.6f}")
    print(f"  analytical log-slope      = {slope_analytical:.6f}")
    print(f"  slope relative error      = {slope_rel_err:.3%}")
    print(f"  Omega(T) measured         = {enstrophy[-1]:.6f}")
    print(f"  Omega(T) analytical       = {enstrophy_analytical[-1]:.6f}")
    print()

    print("PASS criteria:")
    print(f"  [{_format_pass(div_pass)}]  C1: max ||div(t)||_2 <= 1e-8")
    print(f"  [{_format_pass(energy_pass)}]  C2: energy matches analytical within 5%")
    print(f"  [{_format_pass(bkm_pass)}]  C3: BKM integral matches envelope within 5%")
    print(f"  [{_format_pass(vort_pass)}]  C4: vorticity slope matches -2*nu within 5%")

    all_pass = div_pass and energy_pass and bkm_pass and vort_pass
    print()
    print(
        "Result: "
        + (
            "ALL PASS - INCOMP activated, N3/N4 INFO gaps closed for 2D Taylor-Green"
            if all_pass
            else "FAILURE - INCOMP projection or analytics need review"
        )
    )
    print()
    print("Honest scope: this milestone (N5) validates the INCOMP operator on")
    print("2D Taylor-Green; the 3D Clay obstruction NS-G5 (vortex stretching)")
    print("remains OPEN. The genuine geometric-depletion analysis of")
    print("Constantin-Fefferman is fundamentally 3D and is deferred to N6.")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
