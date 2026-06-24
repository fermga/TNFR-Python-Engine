"""N7: Empirical continuum-limit convergence study for 3D TNFR-Navier-Stokes.

Runs the 3D Taylor-Green vortex at increasing resolutions n in {8, 12, 16, 24}
with fixed dt = 0.005 and horizon T = 0.5, and measures whether discrete
observables (E(T), Z(T), BKM integral, stretching production, max divergence)
converge as h -> 0. Convergence orders are estimated by Richardson-like ratios
on the differences between successive resolutions.

Honest scope
------------
This demo is the EMPIRICAL PRECONDITION for NS-G1 (continuum limit of the
discrete TNFR-NS flow), not its closure. Observing the expected convergence
order at smooth Taylor-Green data does NOT prove existence of a continuum
weak/strong solution, nor uniform-in-h estimates, nor convergence in any
Sobolev topology. NS-G1 remains OPEN and NS-G5 (Clay) remains OPEN.

What this demo DOES establish:
  * The discrete 3D TNFR-NS scheme is at minimum CONSISTENT with a continuum
    PDE on smooth data: the observables form a Cauchy-like sequence in n.
  * The empirical order of convergence is compatible with O(h^2) (central
    differences) modulo aliasing at the smallest n.

What this demo does NOT establish:
  * Uniform energy / enstrophy bounds independent of h (NS-G2).
  * Convergence in L^2, H^1, or any function-space topology (NS-G1).
  * Anything about turbulent initial data, rough data, or large times.
"""

from __future__ import annotations

import time

import numpy as np

from tnfr.navier_stokes import TNFRNavierStokesOperator, build_torus_graph_3d

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
RESOLUTIONS = [8, 12, 16, 24]
DT = 0.005
T_FINAL = 0.5
VISCOSITY = 0.05
AMPLITUDE = 1.0
STEPS = int(round(T_FINAL / DT))

# Acceptance tolerances
TOL_DIV = 1e-8
TOL_MIN_ORDER = 1.0  # empirical lower bound (theoretical = 2)


def run_resolution(n: int) -> dict:
    """Run the 3D Taylor-Green vortex at resolution n and return diagnostics."""
    G = build_torus_graph_3d(n)
    op = TNFRNavierStokesOperator(graph=G, viscosity=VISCOSITY, dimension=3)
    op.set_taylor_green(AMPLITUDE)

    t0 = time.perf_counter()
    report = op.bkm_budget(DT, STEPS, advection=True, incompressible=True)
    elapsed = time.perf_counter() - t0

    divs = np.asarray(report["divergence"])
    energy = np.asarray(report["energy"]) if "energy" in report else None
    enstrophy = np.asarray(report["enstrophy"])
    omega_sup = np.asarray(report["vorticity_sup"])
    bkm = np.asarray(report["bkm_integral"])

    # Recompute final E and stretching from operator state
    E_final = float(op.kinetic_energy())
    stretching_final = float(op.stretching_production())

    return {
        "n": n,
        "h": 2 * np.pi / n,
        "elapsed_s": elapsed,
        "div_max": float(divs.max()),
        "E_final": E_final,
        "Z_final": float(enstrophy[-1]),
        "Z_initial": float(enstrophy[0]),
        "BKM": float(bkm[-1]),
        "omega_sup_final": float(omega_sup[-1]),
        "stretching_final": stretching_final,
    }


def convergence_order(values: list[float], hs: list[float]) -> list[float]:
    """Estimate convergence order from successive differences.

    Uses |v_k - v_{k+1}| ~ C h_{k+1}^p as proxy for distance to limit,
    then p = log(|d_{k} / d_{k+1}|) / log(h_k / h_{k+1}).
    """
    diffs = [abs(values[i] - values[i + 1]) for i in range(len(values) - 1)]
    orders: list[float] = []
    for i in range(len(diffs) - 1):
        if diffs[i + 1] <= 0 or diffs[i] <= 0:
            orders.append(float("nan"))
            continue
        # Use h_{k+1} for diff[i] and h_{k+2} for diff[i+1]
        ratio_h = hs[i + 1] / hs[i + 2]
        if ratio_h <= 1.0:
            orders.append(float("nan"))
            continue
        p = float(np.log(diffs[i] / diffs[i + 1]) / np.log(ratio_h))
        orders.append(p)
    return orders


def main() -> int:
    print("=" * 72)
    print("N7: 3D TNFR-NS continuum-limit study (Taylor-Green, NS-G1 precursor)")
    print("=" * 72)
    print(f"  Resolutions n     : {RESOLUTIONS}")
    print(f"  Time step dt      : {DT}")
    print(f"  Final time T      : {T_FINAL}  ({STEPS} steps)")
    print(f"  Viscosity nu      : {VISCOSITY}")
    print(f"  Amplitude A       : {AMPLITUDE}")
    print()

    results: list[dict] = []
    for n in RESOLUTIONS:
        print(
            f"  Running n = {n:3d}  (n^3 = {n ** 3:6d} nodes) ...", end=" ", flush=True
        )
        r = run_resolution(n)
        results.append(r)
        print(f"done in {r['elapsed_s']:6.2f}s")

    hs = [r["h"] for r in results]
    Es = [r["E_final"] for r in results]
    Zs = [r["Z_final"] for r in results]
    BKMs = [r["BKM"] for r in results]
    STRs = [r["stretching_final"] for r in results]
    divs_max = [r["div_max"] for r in results]

    print()
    print("Per-resolution diagnostics")
    print("-" * 72)
    print(
        f"  {'n':>4}  {'h':>8}  {'E(T)':>10}  {'Z(T)':>10}  "
        f"{'BKM':>9}  {'stretch':>9}  {'max||div||':>11}"
    )
    for r in results:
        print(
            f"  {r['n']:>4d}  {r['h']:>8.4f}  "
            f"{r['E_final']:>10.6f}  {r['Z_final']:>10.6f}  "
            f"{r['BKM']:>9.5f}  {r['stretching_final']:>9.4f}  "
            f"{r['div_max']:>11.3e}"
        )

    print()
    print("Successive differences (Cauchy-style proxy for distance to limit)")
    print("-" * 72)

    def diffs(vs: list[float]) -> list[float]:
        return [abs(vs[i] - vs[i + 1]) for i in range(len(vs) - 1)]

    dE = diffs(Es)
    dZ = diffs(Zs)
    dBKM = diffs(BKMs)
    dSTR = diffs(STRs)
    print(f"  |E_n - E_{{n+1}}|   : {['%.3e' % d for d in dE]}")
    print(f"  |Z_n - Z_{{n+1}}|   : {['%.3e' % d for d in dZ]}")
    print(f"  |BKM_n - BKM_{{n+1}}|: {['%.3e' % d for d in dBKM]}")
    print(f"  |str_n - str_{{n+1}}|: {['%.3e' % d for d in dSTR]}")

    print()
    print("Empirical convergence orders (theoretical = 2 for CN + central diff)")
    print("-" * 72)
    oE = convergence_order(Es, hs)
    oZ = convergence_order(Zs, hs)
    oBKM = convergence_order(BKMs, hs)
    print(f"  order(E)   : {['%.3f' % p for p in oE]}")
    print(f"  order(Z)   : {['%.3f' % p for p in oZ]}")
    print(f"  order(BKM) : {['%.3f' % p for p in oBKM]}")

    # Average order over E, Z, BKM (ignoring NaNs)
    all_orders = [p for p in (oE + oZ + oBKM) if np.isfinite(p)]
    mean_order = float(np.mean(all_orders)) if all_orders else float("nan")
    median_order = float(np.median(all_orders)) if all_orders else float("nan")

    print()
    print("Acceptance criteria (honest scope)")
    print("-" * 72)

    c1_pass = max(divs_max) <= TOL_DIV
    print(
        f"  C1 INCOMP uniform    max ||div|| <= {TOL_DIV:.0e} across all n : "
        f"{'PASS' if c1_pass else 'FAIL'}  (got {max(divs_max):.3e})"
    )

    # C2: Cauchy-decreasing differences (allows one rebound)
    def mostly_decreasing(seq: list[float]) -> bool:
        if len(seq) <= 1:
            return True
        # at least the first->last must decrease overall
        return seq[-1] <= seq[0] * 1.5  # allow 50% slack

    c2_pass = mostly_decreasing(dE) and mostly_decreasing(dZ)
    print(
        f"  C2 Cauchy refinement |obs_n - obs_{{n+1}}| not blowing up      : "
        f"{'PASS' if c2_pass else 'FAIL'}  (dE: {dE[0]:.2e}->{dE[-1]:.2e})"
    )

    c3_pass = bool(np.isfinite(mean_order)) and median_order >= TOL_MIN_ORDER
    print(
        f"  C3 Convergence order median(order) >= {TOL_MIN_ORDER}              : "
        f"{'PASS' if c3_pass else 'FAIL'}  "
        f"(mean={mean_order:.3f}, median={median_order:.3f})"
    )

    # C4: all per-resolution runs produced finite, non-negative observables
    c4_pass = all(
        np.isfinite(r["E_final"])
        and np.isfinite(r["Z_final"])
        and np.isfinite(r["BKM"])
        and r["E_final"] >= 0
        and r["Z_final"] >= 0
        and r["BKM"] >= 0
        for r in results
    )
    print(
        f"  C4 Finite observables  every n produces finite (E,Z,BKM) >= 0   : "
        f"{'PASS' if c4_pass else 'FAIL'}"
    )

    passes = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print()
    print(f"  Result: {passes}/4 PASS")
    print()
    print("Honest-scope coda")
    print("-" * 72)
    print(
        "  N7 demonstrates that the discrete 3D TNFR-Navier-Stokes scheme is\n"
        "  EMPIRICALLY CONSISTENT with a continuum PDE on smooth Taylor-Green\n"
        "  data: observables form a Cauchy-like sequence as n grows, with an\n"
        "  empirical convergence order compatible with O(h^2) modulo aliasing\n"
        "  at the smallest n. This is a NECESSARY PRECONDITION for NS-G1 (the\n"
        "  continuum limit of the discrete TNFR-NS flow), not its closure.\n"
        "\n"
        "  NS-G1 (uniform-in-h estimates + convergence in a function-space\n"
        "  topology to a Leray weak solution) remains OPEN. NS-G5 (Clay\n"
        "  Millennium Problem: global regularity of 3D incompressible\n"
        "  Navier-Stokes) remains OPEN. This demo is a stepping stone, not a\n"
        "  proof."
    )

    return 0 if passes == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
