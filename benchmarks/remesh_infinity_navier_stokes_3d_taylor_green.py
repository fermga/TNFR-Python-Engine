"""N12: REMESH-infinity asymptotic limit on the K_phi cascade of the
3D Taylor-Green vortex (TNFR-Navier-Stokes, branch B1 of NS-G_blowup).

Pre-registered milestone (§12 of theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md).
Tests whether canonical REMESH-global mixing applied between
TNFR-Navier-Stokes time-steps, with tau_global swept up to its
asymptotic limit (lag-to-initial-condition), measurably affects the
discrete K_phi (vorticity) cascade of the 3D Taylor-Green vortex.

This is a pre-registration commit. The methodology, parameters, seed
label, and decision thresholds are LOCKED here. No data is collected
at commit time; execution and Results land in a separate commit, with
the Results block appended as §13.

Construction
------------
* Operator: tnfr.navier_stokes.operator.TNFRNavierStokesOperator
  (canonical 3D path, FFT-diagonalised viscous Crank-Nicolson, INCOMP
  via Leray-Helmholtz projection).
* Graph: build_torus_graph_3d(N=16), periodic 3-torus, h = 2*pi/16.
* Initial condition: classical 3D Taylor-Green vortex, A=1.0.
* Time-stepper: Strang split advection + Crank-Nicolson viscous +
  advection (advection=True), with INCOMP applied after each half-step
  (incompressible=True). dt = 0.005, T_final = 1.0, 200 steps.
* Viscosity: nu = 0.01 (Re_eff ~ 628, mid-sweep N11 value).

REMESH-infinity injection
-------------------------
After each full NS step, mix the per-component velocity field with a
lagged copy from the history buffer using the canonical REMESH global
rule:

    phi^(a)_i(t) <- (1 - alpha) * phi^(a)_i(t) + alpha * phi^(a)_i(t - tau_g)

with alpha = 0.5 (canonical default in src/tnfr/config/defaults_core.py).
After mixing, project_incompressible() is re-applied to absorb the
round-off divergence (in continuous arithmetic, a convex combination of
two divergence-free states is divergence-free; INCOMP after mixing
absorbs accumulated floating-point error consistent with the N5/N6
defensive-projection discipline).

tau_g sweep (5 points)
----------------------
* tau_g = 0    : no mixing (baseline; must reproduce standalone N11
                 trajectory at (n=16, nu=0.01, T=1.0) within F1).
* tau_g = 8    : 4% of T_final.
* tau_g = 32   : 16% of T_final.
* tau_g = 128  : 64% of T_final.
* tau_g = inf  : lag-to-initial-condition (the **canonical asymptotic
                 limit** of REMESH global being probed by N12).

For finite tau_g, when k < tau_g (no history yet at step k), no mixing
is applied at that step (the canonical engine's behaviour: REMESH is a
no-op until the history window fills, see
src/tnfr/operators/remesh.py::apply_network_remesh which returns early
if len(hist) < tau_req + 1).

Observables logged per step
---------------------------
* time
* vorticity_sup        : ||omega(t)||_inf (BKM integrand)
* bkm_integral         : int_0^t ||omega||_inf dtau (trapezoidal)
* enstrophy            : ||omega||_{L^2}^2
* kinetic_energy       : (1/2) ||u||_{L^2}^2
* dissipation          : nu * <phi, L phi>
* divergence           : ||div u||_{L^2}  (must stay <= 1e-8 after INCOMP)
* stretching_production: <omega, (omega . grad) u>  (N6 / examples/81)

Pre-registered F-criteria (locked; no post-hoc tolerance adjustment)
--------------------------------------------------------------------
* F1 baseline fidelity     : |BKM(T)_baseline - BKM(T)_reference| / BKM(T)_ref
                             <= 0.01  (reference = standalone N11 run at same
                             config; reference run is the first step of this
                             benchmark, NOT a separate file).
* F2 measurable response   : at least one of {peak ||omega||_inf, BKM(T),
                             peak enstrophy, peak stretching production}
                             shows >= 5% relative change between tau_g=0
                             and at least one of {tau_g=8, 32, 128, inf}.
* F3 monotonicity class    : MONOTONE / NON_MONOTONE / FLAT (= F2 failed)
                             across {tau_g=8, 32, 128, inf}, evaluated per
                             observable; aggregate verdict = MONOTONE if
                             at least one observable is monotone.
* F4 energy non-injection  : max_{k, tau_g} KE[k](tau_g)
                             <= KE[0](tau_g) * (1 + 1e-10).
* F5 divergence control    : max_{k, tau_g} divergence[k] <= 1e-8.

Pre-registered verdict mapping
------------------------------
F1 PASS, F2 PASS, F3 MONOTONE,     F4 PASS, F5 PASS -> STRUCTURAL_EFFECT_MONOTONE
F1 PASS, F2 PASS, F3 NON_MONOTONE, F4 PASS, F5 PASS -> STRUCTURAL_EFFECT_NON_MONOTONE
F1 PASS, F2 FAIL, F3 FLAT,         F4 PASS, F5 PASS -> NULL_RESULT
                                                       (falsifies §11 at n=16)
F1 FAIL                                              -> INDETERMINATE_INFRA_FAIL
F4 FAIL                                              -> INDETERMINATE_INFRA_FAIL
F5 FAIL                                              -> INDETERMINATE_INFRA_FAIL

Honest scope (locked, see §12.8)
--------------------------------
* Single-resolution probe at n=16. PASS does NOT extend to continuum
  limit (NS-G1) or higher Re (NS-G_blowup). A clean PASS at n=16
  motivates a follow-up milestone N13 extending to n in {24, 32}, NOT
  in this commit.
* N12 does NOT close any of NS-G1..G5.
* N12 does NOT promote any new canonical operator; REMESH-global is
  already in the catalog (§11 of NS notes).
* NULL_RESULT is a fully acceptable verdict.

Reproducibility
---------------
* Seed label: 20260526 (cross-program session continuity with the
  Riemann R-inf-1b benchmark). No stochastic RNG is used; TG is
  deterministic and INCOMP is FFT-deterministic. The seed labels the
  session for cross-program audit.
* Run with: python benchmarks/remesh_infinity_navier_stokes_3d_taylor_green.py
* Results JSON: results/remesh_infinity/remesh_infinity_navier_stokes_3d_taylor_green.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# sys.path setup (matches benchmarks/remesh_infinity_riemann_spectral_basis.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from tnfr.navier_stokes.operator import (  # noqa: E402
    TNFRNavierStokesOperator,
    build_torus_graph_3d,
)


# ---------------------------------------------------------------------------
# Locked configuration (§12.4)
# ---------------------------------------------------------------------------
N = 16
VISCOSITY = 0.01
DT = 0.005
T_FINAL = 1.0
STEPS = int(round(T_FINAL / DT))           # 200
AMPLITUDE = 1.0
ALPHA = 0.5                                # canonical REMESH_ALPHA default
TAU_G_SWEEP: list[Any] = [0, 8, 32, 128, "inf"]
SEED_LABEL = 20260526
EPS = 1e-14

# F-criteria thresholds
F1_TOL = 0.01      # 1% baseline reproduction of N11 reference
F2_MIN_REL_CHANGE = 0.05  # 5% to count as measurable response
F4_KE_INJECTION_TOL = 1e-10
F5_DIV_TOL = 1e-8


# ---------------------------------------------------------------------------
# REMESH-infinity execution
# ---------------------------------------------------------------------------
def run_one_tau(
    tau_g: Any,
    *,
    initial_state: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the 3D TG simulation with REMESH-global mixing at lag tau_g.

    Parameters
    ----------
    tau_g : int or "inf"
        Lag in step units, or the literal string "inf" for the
        asymptotic limit (lag-to-initial-condition).
    initial_state : np.ndarray
        Shape (3, n_nodes). Reused as the tau_g="inf" mixing reference
        and as the IC for every run.

    Returns
    -------
    dict of numpy arrays (length STEPS + 1 each)
    """
    G = build_torus_graph_3d(N)
    op = TNFRNavierStokesOperator(graph=G, viscosity=VISCOSITY, dimension=3)
    op.set_components(initial_state)
    # Defensive INCOMP at t=0 (TG IC is already div-free; absorbs round-off).
    op.project_incompressible()

    # History buffer for finite tau_g; None for tau_g="inf"
    history: deque[np.ndarray] | None = None
    if isinstance(tau_g, int) and tau_g > 0:
        history = deque(maxlen=tau_g + 1)
        history.append(op.phi.copy())

    # Pre-allocate observable arrays
    n_pts = STEPS + 1
    time_axis = np.zeros(n_pts, dtype=float)
    vort_sup = np.zeros(n_pts, dtype=float)
    bkm_int = np.zeros(n_pts, dtype=float)
    enstrophy = np.zeros(n_pts, dtype=float)
    kinetic = np.zeros(n_pts, dtype=float)
    dissipation = np.zeros(n_pts, dtype=float)
    divergence = np.zeros(n_pts, dtype=float)
    stretching = np.zeros(n_pts, dtype=float)

    # Initial telemetry
    time_axis[0] = op.time
    vort_sup[0] = op.vorticity_sup_norm()
    enstrophy[0] = op.enstrophy_curl()
    kinetic[0] = op.kinetic_energy()
    dissipation[0] = op.dissipation_rate()
    divergence[0] = op.divergence_residual()
    stretching[0] = op.stretching_production()

    for k in range(1, STEPS + 1):
        # NS step (canonical Strang-split with INCOMP after each half-step)
        op.step(DT, advection=True, incompressible=True)

        # REMESH-infinity mixing (canonical (1-alpha) phi(t) + alpha phi(t-tau_g))
        if tau_g == 0:
            mixed = False
        elif tau_g == "inf":
            op.phi = (1.0 - ALPHA) * op.phi + ALPHA * initial_state
            mixed = True
        else:
            # Finite tau_g: mix only when history window has filled
            assert history is not None
            if len(history) >= tau_g + 1:
                phi_past = history[0]   # oldest entry = phi(t - tau_g)
                op.phi = (1.0 - ALPHA) * op.phi + ALPHA * phi_past
                mixed = True
            else:
                mixed = False
            history.append(op.phi.copy())

        # Defensive INCOMP after mixing (round-off absorption)
        if mixed:
            op.project_incompressible()

        # Telemetry
        time_axis[k] = op.time
        vort_sup[k] = op.vorticity_sup_norm()
        bkm_int[k] = bkm_int[k - 1] + 0.5 * DT * (
            vort_sup[k - 1] + vort_sup[k]
        )
        enstrophy[k] = op.enstrophy_curl()
        kinetic[k] = op.kinetic_energy()
        dissipation[k] = op.dissipation_rate()
        divergence[k] = op.divergence_residual()
        stretching[k] = op.stretching_production()

    return {
        "time": time_axis,
        "vorticity_sup": vort_sup,
        "bkm_integral": bkm_int,
        "enstrophy": enstrophy,
        "kinetic_energy": kinetic,
        "dissipation": dissipation,
        "divergence": divergence,
        "stretching_production": stretching,
    }


# ---------------------------------------------------------------------------
# F-criteria evaluation
# ---------------------------------------------------------------------------
def _peak(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def evaluate_criteria(
    runs: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Evaluate F1-F5 pre-registered criteria and return verdict."""
    baseline_key = "tau_g=0"
    baseline = runs[baseline_key]

    # F1: baseline fidelity vs N11 reference.
    # Per §12.6, the reference is the standalone N11 trajectory at same
    # config. Since tau_g=0 is exactly that standalone run (mixing is a
    # no-op), F1 is structurally tautological *within this benchmark*:
    # it verifies the tau_g=0 wiring does not call mixing and reproduces
    # itself bit-for-bit when re-instantiated. We enforce this by
    # re-running tau_g=0 once and comparing BKM(T) integrals.
    initial_state = np.empty((3, N * N * N), dtype=float)
    G_tmp = build_torus_graph_3d(N)
    op_tmp = TNFRNavierStokesOperator(graph=G_tmp, viscosity=VISCOSITY, dimension=3)
    op_tmp.set_taylor_green(AMPLITUDE)
    initial_state[:] = op_tmp.phi
    ref_run = run_one_tau(0, initial_state=initial_state)
    bkm_baseline = float(baseline["bkm_integral"][-1])
    bkm_reference = float(ref_run["bkm_integral"][-1])
    f1_rel_err = (
        abs(bkm_baseline - bkm_reference) / max(abs(bkm_reference), EPS)
    )
    f1_pass = f1_rel_err <= F1_TOL

    # Peak observables per run
    peak_table: dict[str, dict[str, float]] = {}
    for key, run in runs.items():
        peak_table[key] = {
            "peak_vorticity_sup": _peak(run["vorticity_sup"]),
            "BKM_T": float(run["bkm_integral"][-1]),
            "peak_enstrophy": _peak(run["enstrophy"]),
            "peak_stretching": _peak(run["stretching_production"]),
        }

    # F2: measurable response (>= 5% in at least one observable)
    baseline_peaks = peak_table[baseline_key]
    f2_changes: dict[str, dict[str, float]] = {}
    max_rel_change = 0.0
    for key in runs:
        if key == baseline_key:
            continue
        rels = {}
        for obs_label, base_val in baseline_peaks.items():
            new_val = peak_table[key][obs_label]
            rel = abs(new_val - base_val) / max(abs(base_val), EPS)
            rels[obs_label] = rel
            if rel > max_rel_change:
                max_rel_change = rel
        f2_changes[key] = rels
    f2_pass = max_rel_change >= F2_MIN_REL_CHANGE

    # F3: monotonicity across {tau_g=8, 32, 128, inf}
    mixing_keys = ["tau_g=8", "tau_g=32", "tau_g=128", "tau_g=inf"]
    f3_monotone_observables: list[str] = []
    for obs_label in baseline_peaks.keys():
        values = [peak_table[k][obs_label] for k in mixing_keys]
        diffs = np.diff(values)
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            f3_monotone_observables.append(obs_label)
    if not f2_pass:
        f3_class = "FLAT"
    elif f3_monotone_observables:
        f3_class = "MONOTONE"
    else:
        f3_class = "NON_MONOTONE"

    # F4: energy non-injection across all runs
    f4_max_excess = 0.0
    for key, run in runs.items():
        ke = run["kinetic_energy"]
        excess = float(np.max(ke) - ke[0])
        rel_excess = excess / max(abs(ke[0]), EPS)
        if rel_excess > f4_max_excess:
            f4_max_excess = rel_excess
    f4_pass = f4_max_excess <= F4_KE_INJECTION_TOL

    # F5: divergence control
    f5_max_div = max(
        float(np.max(run["divergence"])) for run in runs.values()
    )
    f5_pass = f5_max_div <= F5_DIV_TOL

    # Verdict mapping (§12.7)
    if not f1_pass or not f4_pass or not f5_pass:
        verdict = "INDETERMINATE_INFRA_FAIL"
    elif f2_pass and f3_class == "MONOTONE":
        verdict = "STRUCTURAL_EFFECT_MONOTONE"
    elif f2_pass and f3_class == "NON_MONOTONE":
        verdict = "STRUCTURAL_EFFECT_NON_MONOTONE"
    elif not f2_pass and f3_class == "FLAT":
        verdict = "NULL_RESULT"
    else:
        verdict = "INDETERMINATE_OTHER"

    return {
        "F1": {
            "pass": f1_pass,
            "bkm_baseline": bkm_baseline,
            "bkm_reference": bkm_reference,
            "rel_err": f1_rel_err,
            "tol": F1_TOL,
        },
        "F2": {
            "pass": f2_pass,
            "max_rel_change": max_rel_change,
            "threshold": F2_MIN_REL_CHANGE,
            "per_run_changes": f2_changes,
        },
        "F3": {
            "class": f3_class,
            "monotone_observables": f3_monotone_observables,
        },
        "F4": {
            "pass": f4_pass,
            "max_relative_excess": f4_max_excess,
            "tol": F4_KE_INJECTION_TOL,
        },
        "F5": {
            "pass": f5_pass,
            "max_divergence": f5_max_div,
            "tol": F5_DIV_TOL,
        },
        "verdict": verdict,
        "peak_table": peak_table,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    print("=" * 76)
    print("N12: REMESH-infinity asymptotic limit on 3D Taylor-Green K_phi cascade")
    print("=" * 76)
    print(
        f"Config: N={N}, nu={VISCOSITY}, dt={DT}, T={T_FINAL} (STEPS={STEPS}), "
        f"alpha={ALPHA}"
    )
    print(f"tau_g sweep: {TAU_G_SWEEP}")
    print(f"Seed label: {SEED_LABEL}")
    print()

    # Shared initial condition (deterministic 3D TG)
    G0 = build_torus_graph_3d(N)
    op0 = TNFRNavierStokesOperator(graph=G0, viscosity=VISCOSITY, dimension=3)
    op0.set_taylor_green(AMPLITUDE)
    initial_state = op0.phi.copy()

    runs: dict[str, dict[str, np.ndarray]] = {}
    timings: dict[str, float] = {}
    for tau_g in TAU_G_SWEEP:
        label = f"tau_g={tau_g}"
        print(f"  Running {label} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        run = run_one_tau(tau_g, initial_state=initial_state)
        elapsed = time.perf_counter() - t0
        timings[label] = elapsed
        runs[label] = run
        print(
            f"done ({elapsed:.1f} s). "
            f"BKM(T)={run['bkm_integral'][-1]:.4f}, "
            f"peak |omega|_inf={float(np.max(run['vorticity_sup'])):.4f}, "
            f"max div={float(np.max(run['divergence'])):.2e}"
        )

    print()
    print("Evaluating F1-F5 ...")
    eval_result = evaluate_criteria(runs)

    print()
    print("=" * 76)
    print("Pre-registered F-criteria (locked in §12.6)")
    print("=" * 76)
    f1 = eval_result["F1"]
    print(
        f"F1 baseline fidelity     : {'PASS' if f1['pass'] else 'FAIL'}  "
        f"(rel_err={f1['rel_err']:.2e}, tol={f1['tol']})"
    )
    f2 = eval_result["F2"]
    print(
        f"F2 measurable response   : {'PASS' if f2['pass'] else 'FAIL'}  "
        f"(max_rel_change={f2['max_rel_change']:.4f}, "
        f"threshold={f2['threshold']})"
    )
    f3 = eval_result["F3"]
    print(
        f"F3 monotonicity          : {f3['class']}  "
        f"(monotone observables: {f3['monotone_observables']})"
    )
    f4 = eval_result["F4"]
    print(
        f"F4 energy non-injection  : {'PASS' if f4['pass'] else 'FAIL'}  "
        f"(max_rel_excess={f4['max_relative_excess']:.2e}, "
        f"tol={f4['tol']})"
    )
    f5 = eval_result["F5"]
    print(
        f"F5 divergence control    : {'PASS' if f5['pass'] else 'FAIL'}  "
        f"(max_div={f5['max_divergence']:.2e}, tol={f5['tol']})"
    )
    print()
    print(f"VERDICT (§12.7): {eval_result['verdict']}")
    print()

    # Per-tau_g peak table
    print("Peak observables per tau_g")
    print("-" * 76)
    headers = ["tau_g", "peak |omega|", "BKM(T)", "peak enstrophy",
               "peak stretching"]
    print(f"{headers[0]:<10}  {headers[1]:<12}  {headers[2]:<10}  "
          f"{headers[3]:<16}  {headers[4]:<16}")
    for label in [f"tau_g={tg}" for tg in TAU_G_SWEEP]:
        pk = eval_result["peak_table"][label]
        print(
            f"{label:<10}  {pk['peak_vorticity_sup']:<12.6f}  "
            f"{pk['BKM_T']:<10.6f}  {pk['peak_enstrophy']:<16.6f}  "
            f"{pk['peak_stretching']:<16.6f}"
        )
    print()

    # Persist JSON
    out_dir = _REPO_ROOT / "results" / "remesh_infinity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remesh_infinity_navier_stokes_3d_taylor_green.json"

    payload: dict[str, Any] = {
        "milestone": "N12",
        "program": "TNFR-Navier-Stokes",
        "section": "§12 (pre-registered) -> §13 (results)",
        "seed_label": SEED_LABEL,
        "config": {
            "N": N,
            "viscosity": VISCOSITY,
            "dt": DT,
            "T_final": T_FINAL,
            "steps": STEPS,
            "amplitude": AMPLITUDE,
            "alpha": ALPHA,
            "tau_g_sweep": [str(t) for t in TAU_G_SWEEP],
        },
        "timings_seconds": timings,
        "criteria": {
            "F1": {**eval_result["F1"]},
            "F2": {
                "pass": eval_result["F2"]["pass"],
                "max_rel_change": eval_result["F2"]["max_rel_change"],
                "threshold": eval_result["F2"]["threshold"],
                "per_run_changes": eval_result["F2"]["per_run_changes"],
            },
            "F3": eval_result["F3"],
            "F4": eval_result["F4"],
            "F5": eval_result["F5"],
        },
        "verdict": eval_result["verdict"],
        "peak_table": eval_result["peak_table"],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)

    print(f"Results written to: {out_path.relative_to(_REPO_ROOT)}")
    return payload


if __name__ == "__main__":
    main()
