"""U2-compliance of 3D Navier-Stokes in TNFR structural time.

Memo: theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md §21.
Post-hoc reproducible harness (measure-first study, NOT a locked-before-execution
N-milestone). Deterministic (Taylor-Green IC, no RNG).

QUESTION (user, 2026-06-28): does the 3D NS vorticity dynamics map to a U1-U6
operator sequence with a BOUNDED destabilizer debt? If yes the Lyapunov closes;
if no, the wall is now grammatical.

FAITHFUL OPERATOR MAPPING (vorticity NS):
    d_t omega + (u.grad)omega = (omega.grad)u + nu*Lap omega
  nu*Lap omega   -> IL  (Coherence)  stabilizer = nodal-eq graph diffusion (EXACT)
  (omega.grad)u  -> VAL (Expansion)  DESTABILIZER: raises enstrophy K_phi^2 = P
  (u.grad)omega  -> NAV (Transition) skew, enstrophy-NEUTRAL, NOT a U2 destabilizer
  -grad p, div=0 -> UM  (Coupling)   absent in the vorticity eqn, enstrophy-NEUTRAL
The neutrality is ANALYTIC: the pressure gradient VANISHES under the curl, and the
advective transport is skew, int omega.(u.grad)omega = -1/2 int (div u)|omega|^2 = 0
(incompressible). Discrete zero-production for the 2D-embedded IC is independently
confirmed in N10 / §19. Hence ONLY VAL (P) and IL (D_ens) drive the enstrophy
budget dOmega/dt = P - D_ens, so the U2 DEBT
    int_0^t (P - D_ens) dt' = Omega(t) - Omega(0)
is exactly the enstrophy change, and U2-compliance <=> bounded enstrophy <=>
classical regularity. The grammar FAITHFULLY re-expresses Clay (not a surrogate).

TNFR TIME IS STRUCTURAL: tau_str = nu_f*t (the eigenmode-decay clock
e^{-nu_f*lambda_k*t}; the U2 capacity floor(1/(nu_f*dt*rho)) is in nu_f*dt units).
In NS nu_f<->nu so tau_str = nu*t (the viscous time). TWO operator clocks:
IL (dissipation) on the structural clock tau_str; VAL (stretching) on the
advective eddy clock tau_eddy ~ 1/|omega| (nu-INDEPENDENT). Re = their ratio =
VAL firings per IL relaxation = the U2 debt rate.

F-CRITERIA (measured from the operator's own enstrophy_curl / stretching_production):
  F1 fixed-Re saturation: every run's enstrophy peaks and DECAYS in tau_str
     (debt bounded -> U2-compliant -> Lyapunov closes at fixed Re).
  F2 peak-debt trend (RECORDED, not pass/fail): peak debt Omega/Omega0 at matched
     tau_str vs Re, with the Kolmogorov resolution flag k_max*eta.

HONEST SCOPE: closes NOTHING. The faithful U2 reformulation RELOCATES Clay to
"is the peak structural-time U2 debt uniformly bounded in Re?"; it does NOT lower
the wall. Fixed-Re U2-compliance = known finite-Re regularity re-expressed. The
peak debt is MEASURED to GROW with Re over the accessible (resolution-limited)
range, and resolving the grid (n=32->48 at Re~1257: 2.42->2.89) RAISES it (coarse
grids under-state the production); whether it stays finite as Re->inf is the open
Clay question. No new analytical bound is produced.

Override N=48 (and re-run) to reproduce the resolved Re~1257 point (peak ~2.89).

Output: results/u2_compliance/u2_compliance_navier_stokes_structural_time.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from tnfr.navier_stokes.operator import (  # noqa: E402
    TNFRNavierStokesOperator,
    build_torus_graph_3d,
)

# ---------------------------------------------------------------------------
# Locked configuration (deterministic; override N to reproduce the n=48 point)
# ---------------------------------------------------------------------------
N = 32
DT = 0.0125
AMPLITUDE = 1.0
TAU_TARGET = 0.20  # structural time tau_str = nu*t reached by every run
VISCOSITY_SWEEP: list[float] = [0.02, 0.01, 0.005]  # Re ~ 314, 628, 1257
RESOLVED_KETA = 1.3  # k_max*eta above this = resolved at the enstrophy peak
BOX_VOLUME = (2.0 * np.pi) ** 3


def run_one(nu: float) -> dict[str, Any]:
    """Evolve TG at viscosity nu to tau_str=TAU_TARGET; read the operator's own
    enstrophy (enstrophy_curl) and production (stretching_production)."""
    steps = int(round(TAU_TARGET / nu / DT))
    sample = max(1, steps // 30)
    graph = build_torus_graph_3d(N)
    op = TNFRNavierStokesOperator(graph=graph, viscosity=nu, dimension=3)
    op.set_taylor_green(amplitude=AMPLITUDE)

    tau, omega_e, prod = [], [], []
    for s in range(steps + 1):
        if s % sample == 0:
            tau.append(nu * s * DT)
            omega_e.append(op.enstrophy_curl())
            prod.append(op.stretching_production())
        if s < steps:
            op.step(DT, advection=True, incompressible=True)

    omega_e = np.asarray(omega_e)
    Om0 = float(omega_e[0])
    peak = float(np.max(omega_e))
    ipk = int(np.argmax(omega_e))
    eps_peak = 2.0 * nu * peak / BOX_VOLUME
    eta_peak = (nu**3 / eps_peak) ** 0.25
    keta = (N / 2.0) * eta_peak
    return {
        "nu": nu,
        "Re": 2.0 * np.pi / nu,
        "Omega0": Om0,
        "peak_debt": peak / Om0,
        "tau_at_peak": float(tau[ipk]),
        "final_over_peak": float(omega_e[-1] / peak),
        "saturates": bool(omega_e[-1] < peak * 0.999),
        "max_production": float(np.max(prod)),
        "kmax_eta_at_peak": keta,
        "resolved": bool(keta > RESOLVED_KETA),
        "tau": [float(x) for x in tau],
        "debt": [float(x / Om0) for x in omega_e],
    }


def main() -> None:
    runs = [run_one(nu) for nu in VISCOSITY_SWEEP]

    f1 = all(r["saturates"] for r in runs)
    peak_trend = [(r["Re"], r["peak_debt"], r["resolved"]) for r in runs]
    rising = all(
        peak_trend[i + 1][1] > peak_trend[i][1] for i in range(len(peak_trend) - 1)
    )
    verdict = (
        "FAITHFUL_U2_MAPPING"
        + ("__COMPLIANT_AT_FIXED_RE" if f1 else "__FIXED_RE_COMPLIANCE_NOT_SHOWN")
        + "__ASYMPTOTIC_UNDECIDED"
    )

    print("=" * 70)
    print("U2-COMPLIANCE OF NS IN STRUCTURAL TIME tau_str = nu*t")
    print("=" * 70)
    print(f"  n={N}, dt={DT}, tau_target={TAU_TARGET}, TG amplitude={AMPLITUDE}")
    print("  mapping: VAL=(omega.grad)u=P (destabilizer), IL=nu*Lap omega=D_ens")
    print("  (stabilizer); NAV transport + UM pressure enstrophy-neutral (analytic).")
    print("\n  F1 fixed-Re saturation (enstrophy peaks & decays in tau_str):")
    for r in runs:
        print(f"     Re~{r['Re']:.0f}: max production P={r['max_production']:.2f} "
              f"(VAL active); peak_debt={r['peak_debt']:.3f} at "
              f"tau_str={r['tau_at_peak']:.3f}, final/peak={r['final_over_peak']:.3f} "
              f"-> {'SATURATES' if r['saturates'] else 'CLIMBING'}")
    print(f"   -> F1 {'PASS' if f1 else 'FAIL'} "
          "(debt bounded -> U2-compliant -> Lyapunov closes at fixed Re)")
    print(f"\n  F2 peak U2 debt vs Re (matched tau_str={TAU_TARGET}) [RECORDED]:")
    print("       Re     peak Om/Om0   kmax*eta   resolved")
    for r in runs:
        print(f"     {r['Re']:6.0f}     {r['peak_debt']:6.3f}      "
              f"{r['kmax_eta_at_peak']:5.2f}      {r['resolved']}")
    print(f"   -> peak debt {'RISES' if rising else 'does NOT rise'} with Re; "
          "asymptotic Re->inf bound = Clay, NOT decided here.")
    print(f"\n  VERDICT: {verdict}")
    print("  HONEST SCOPE: faithful reformulation RELOCATES Clay (does not lower")
    print("  the wall); no uniform-in-Re bound; closes nothing. Clay open.")

    out_dir = _REPO_ROOT / "results" / "u2_compliance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "u2_compliance_navier_stokes_structural_time.json"
    out_path.write_text(json.dumps(
        {"config": {"N": N, "DT": DT, "TAU_TARGET": TAU_TARGET,
                    "VISCOSITY_SWEEP": VISCOSITY_SWEEP},
         "F1_fixed_Re_saturation": f1,
         "F2_peak_debt_trend": peak_trend, "peak_debt_rises_with_Re": rising,
         "verdict": verdict, "runs": runs}, indent=2))
    print(f"\n  JSON -> {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
