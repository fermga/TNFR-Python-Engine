"""N8 — Uniform-in-h discrete Leray energy inequality (NS-G2 precursor).

Goal
----
Empirically check that the discrete Leray energy inequality

    E_n(t) + nu * integral_0^t ||grad_h u_n||^2 ds  <=  E_n(0)

holds at *every* resolution n in a refinement sequence, **and** that the
relevant scalars (initial energy, cumulative dissipation, sup dissipation,
and the cumulative-budget defect) remain bounded uniformly in h = 2*pi/n
as n grows. This is the empirical content of NS-G2 in the research notes
(`theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` Sec. 5).

Setup
-----
3D Taylor-Green vortex on a periodic torus graph; same parameters as N7
so the two studies are directly comparable. INCOMP active throughout so
the energy budget reflects viscous dissipation only (no pressure work
artefact).

Finding (dimensional scaling)
-----------------------------
The raw `dissipation_rate()` exposed by the operator evaluates the
combinatorial-Laplacian quadratic form `<phi, L_comb phi>`. On a uniform
grid, the continuum `||grad u||^2_{L^2}` is approximated by
`h^{d-2} * <phi, L_comb phi>` (the `h^d` Riemann volume element cancels
the implicit `1/h^2` of the physical Laplacian). For d=2 the factor is
`h^0 = 1` and the raw value already matches; for d=3 the factor is
`h^1 = h`, so the raw dissipation rate overestimates the physical one by
`1/h` and DIVERGES as the mesh is refined. This demo therefore reports
both the raw quantity (as returned by `leray_budget`) and a
dimensionally-corrected `D_phys = h^{d-2} * D_raw` for the uniform-in-h
check. The corrected dissipation is the one that bounds the continuum
Leray balance.

Honest scope
------------
This demo is the **empirical precondition** for NS-G2 (uniformity of the
discrete energy inequality in mesh refinement). It does NOT prove
uniform-in-h estimates — that would require analytical bounds on the
discrete Laplacian quadratic form and the advection term independent of
n. A passing demo only certifies that the dimensionally-corrected
constants do not blow up over the tested resolution window
{8, 12, 16, 24} on this particular smooth initial datum. NS-G2 closure
remains OPEN.

NS-G1, NS-G3, NS-G4, NS-G5 are not addressed by this demo. The Clay
Millennium Problem (global regularity of 3D incompressible Navier-Stokes)
remains OPEN.
"""

from __future__ import annotations

import time

import numpy as np

from tnfr.navier_stokes import TNFRNavierStokesOperator, build_torus_graph_3d

# ---------------------------------------------------------------------------
# Configuration (kept aligned with N7 so the two studies share the same
# refinement axis and can be cross-referenced).
# ---------------------------------------------------------------------------
RESOLUTIONS = [8, 12, 16, 24]
DT = 0.005
T_FINAL = 0.5
STEPS = int(round(T_FINAL / DT))
VISCOSITY = 0.05
AMPLITUDE = 1.0

# Tolerances for the acceptance criteria below. All are honest engineering
# slacks; none of them is a tight theoretical bound.
#
# TOL_BUDGET_DEFECT: at every step, ``cumulative_budget[k] = E(0) - E(k) -
# integral_0^{t_k} D(s) ds`` should be non-negative up to numerical noise.
# The slack absorbs Strang-splitting O(dt^2)*steps drift in the energy
# balance (continuum identity holds; splitting introduces a controlled error
# that scales with dt^2 and the number of steps).
TOL_BUDGET_DEFECT = 5.0e-2

# TOL_DIV: incompressibility residual; identical to N7.
TOL_DIV = 1.0e-8


def run_resolution(n: int) -> dict:
    """Run the 3D Taylor-Green vortex at resolution n and return the
    full Leray budget plus uniform-in-h diagnostics."""
    G = build_torus_graph_3d(n)
    op = TNFRNavierStokesOperator(graph=G, viscosity=VISCOSITY, dimension=3)
    op.set_taylor_green(AMPLITUDE)

    h = 2 * np.pi / n
    # In 3D the continuum-equivalent dissipation rate is h^{d-2} * D_raw.
    scale = h ** (3 - 2)  # = h

    t0 = time.perf_counter()
    budget = op.leray_budget(dt=DT, steps=STEPS, advection=True, incompressible=True)
    elapsed = time.perf_counter() - t0

    E0 = float(budget["energy"][0])
    E_T = float(budget["energy"][-1])
    sup_D_raw = float(np.max(budget["dissipation"]))
    int_D_raw = float(budget["cumulative_dissipated"][-1])
    sup_D_phys = sup_D_raw * scale
    int_D_phys = int_D_raw * scale
    # Physical cumulative-budget defect uses the corrected integral.
    times = budget["time"]
    diss_phys = budget["dissipation"] * scale
    cum_diss_phys = np.zeros_like(diss_phys)
    for k in range(1, len(diss_phys)):
        cum_diss_phys[k] = (
            cum_diss_phys[k - 1] + 0.5 * (diss_phys[k] + diss_phys[k - 1]) * DT
        )
    cum_budget_phys = E0 - budget["energy"] - cum_diss_phys
    min_budget_phys = float(np.min(cum_budget_phys))
    max_div = float(np.max(budget["divergence"]))

    return {
        "n": n,
        "h": h,
        "elapsed_s": elapsed,
        "E0": E0,
        "E_T": E_T,
        "sup_D_raw": sup_D_raw,
        "int_D_raw": int_D_raw,
        "sup_D_phys": sup_D_phys,
        "int_D_phys": int_D_phys,
        "min_budget_phys": min_budget_phys,
        "max_div": max_div,
        "dE": E0 - E_T,
    }


def fmt(x: float, w: int = 11, p: int = 4) -> str:
    return f"{x:{w}.{p}f}"


def main() -> None:
    print("=" * 72)
    print("N8: uniform-in-h discrete Leray energy inequality (NS-G2 precursor)")
    print("=" * 72)
    print(f"  Resolutions n     : {RESOLUTIONS}")
    print(f"  Time step dt      : {DT}")
    print(f"  Final time T      : {T_FINAL}  ({STEPS} steps)")
    print(f"  Viscosity nu      : {VISCOSITY}")
    print(f"  Amplitude A       : {AMPLITUDE}")
    print("  Note: D_phys = h^{d-2} * D_raw  (= h * D_raw in 3D)")
    print()

    rows = []
    for n in RESOLUTIONS:
        print(f"  Running n = {n:3d}  (n^3 = {n**3:6d} nodes) ... ", end="", flush=True)
        r = run_resolution(n)
        rows.append(r)
        print(f"done in {r['elapsed_s']:6.2f}s")

    print()
    print("Per-resolution Leray-budget diagnostics")
    print("-" * 78)
    print(
        f"  {'n':>4}  {'h':>6}  {'E(0)':>9}  {'E(T)':>9}  "
        f"{'sup D_raw':>10}  {'sup D_phys':>10}  {'int D_phys':>10}  "
        f"{'min bdg_phys':>13}  {'max ||div||':>11}"
    )
    for r in rows:
        print(
            f"  {r['n']:>4d}  {r['h']:>6.4f}  {r['E0']:>9.4f}  {r['E_T']:>9.4f}  "
            f"{r['sup_D_raw']:>10.4f}  {r['sup_D_phys']:>10.4f}  "
            f"{r['int_D_phys']:>10.4f}  {r['min_budget_phys']:>13.4e}  "
            f"{r['max_div']:>11.3e}"
        )
    print()

    # ------------------------------------------------------------------
    # Uniform-in-h diagnostics on the DIMENSIONALLY-CORRECTED quantities.
    # ------------------------------------------------------------------
    def diffs(key: str) -> list[float]:
        vals = [r[key] for r in rows]
        return [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]

    diff_E0 = diffs("E0")
    diff_intD = diffs("int_D_phys")
    diff_supD = diffs("sup_D_phys")
    diff_dE = diffs("dE")

    print("Successive differences across refinement (uniform-in-h check)")
    print("-" * 78)
    print(f"  |E0_n - E0_{{n+1}}|         : {[f'{x:.3e}' for x in diff_E0]}")
    print(f"  |sup D_phys diffs|      : {[f'{x:.3e}' for x in diff_supD]}")
    print(f"  |int D_phys diffs|      : {[f'{x:.3e}' for x in diff_intD]}")
    print(f"  |dE diffs|              : {[f'{x:.3e}' for x in diff_dE]}")
    print()

    # ------------------------------------------------------------------
    # Acceptance criteria (honest, NOT theoretical bounds).
    # ------------------------------------------------------------------
    print("Acceptance criteria (honest scope)")
    print("-" * 78)

    # C1: per-resolution discrete Leray inequality on the DIMENSIONALLY-
    # CORRECTED budget. cumulative_budget_phys[k] >= -TOL_BUDGET_DEFECT at
    # every step, every n. Continuum identity is dE/dt = -D_phys, so the
    # corrected cumulative budget should be ~ 0 up to Strang O(dt^2) error.
    c1_worst = min(r["min_budget_phys"] for r in rows)
    c1_pass = c1_worst >= -TOL_BUDGET_DEFECT
    print(
        f"  C1 Per-n Leray ineq  min cumul. budget (phys) >= -{TOL_BUDGET_DEFECT:.0e}   : "
        f"{'PASS' if c1_pass else 'FAIL'}  (got {c1_worst:.3e})"
    )

    # C2: uniform incompressibility, identical to N7.
    c2_worst = max(r["max_div"] for r in rows)
    c2_pass = c2_worst <= TOL_DIV
    print(
        f"  C2 INCOMP uniform    max ||div|| <= {TOL_DIV:.0e} across all n : "
        f"{'PASS' if c2_pass else 'FAIL'}  (got {c2_worst:.3e})"
    )

    # C3: uniform-in-h boundedness — E(0), sup D_phys, int D_phys should
    # NOT diverge as n grows. We enforce: the LAST diff must not exceed
    # the FIRST diff by a factor larger than 2.
    def mostly_decreasing(xs: list[float], slack: float = 2.0) -> bool:
        if len(xs) < 2:
            return True
        return xs[-1] <= slack * xs[0]

    c3_E0 = mostly_decreasing(diff_E0)
    c3_supD = mostly_decreasing(diff_supD)
    c3_intD = mostly_decreasing(diff_intD)
    c3_pass = c3_E0 and c3_supD and c3_intD
    print(
        "  C3 Uniform-in-h bnd  diffs(E0, sup D_phys, int D_phys) bounded  : "
        f"{'PASS' if c3_pass else 'FAIL'}  "
        f"(E0:{'OK' if c3_E0 else 'X'} supD:{'OK' if c3_supD else 'X'} intD:{'OK' if c3_intD else 'X'})"
    )

    # C4: every observable finite and non-negative where physical.
    c4_pass = all(
        np.isfinite(r["E0"])
        and r["E0"] >= 0
        and np.isfinite(r["E_T"])
        and r["E_T"] >= 0
        and np.isfinite(r["sup_D_phys"])
        and r["sup_D_phys"] >= 0
        and np.isfinite(r["int_D_phys"])
        and r["int_D_phys"] >= 0
        for r in rows
    )
    print(
        "  C4 Finite observables  E(0), E(T), sup D_phys, int D_phys finite>=0 : "
        f"{'PASS' if c4_pass else 'FAIL'}"
    )

    passes = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print()
    print(f"  Result: {passes}/4 PASS")
    print()

    # ------------------------------------------------------------------
    # Honest scope coda
    # ------------------------------------------------------------------
    print("Honest-scope coda")
    print("-" * 72)
    print(
        "  N8 checks empirically that the discrete Leray energy inequality\n"
        "  holds at every n in {8, 12, 16, 24} and that its scalar inputs\n"
        "  (E(0), sup D_phys(t), integral D_phys dt) remain bounded\n"
        "  uniformly in h = 2*pi/n once the d=3 volume-element factor\n"
        "  h^{d-2} = h is restored. This is the EMPIRICAL CONTENT of NS-G2\n"
        "  (uniformity of the discrete energy inequality in mesh refinement)\n"
        "  on a smooth Taylor-Green datum.\n"
        "\n"
        "  Side finding: the operator's `dissipation_rate()` in 3D returns\n"
        "  the bare combinatorial quadratic form `<phi, L_comb phi>` and\n"
        "  therefore overestimates the continuum `nu * ||grad u||_{L^2}^2`\n"
        "  by a factor `1/h` in 3D (in 2D this factor is 1 and the\n"
        "  agreement is exact). Any consumer of `leray_budget` that needs\n"
        "  the physical dissipation rate in 3D must multiply by `h`. This\n"
        "  is a unit-of-measure issue, not a sign of a bug in the flow.\n"
        "\n"
        "  This is NOT a proof of NS-G2. A proof requires analytical\n"
        "  bounds on the discrete Laplacian quadratic form, the\n"
        "  Strang-splitting truncation error, and the advection-projection\n"
        "  composition, ALL uniform in h. None of those are produced here.\n"
        "  Even if NS-G2 were closed, NS-G1 (continuum limit), NS-G3\n"
        "  (discrete-to-continuum BKM), and NS-G5 (Clay) would remain OPEN.\n"
        "  This demo is a stepping stone, not a proof."
    )


if __name__ == "__main__":
    main()
