"""N13: REMESH-infinity resolution extension on 3D Taylor-Green with refined F3.

Pre-registered in `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §14
(committed atomically with this file, before any data is observed).

Purpose
-------
Replicate the N12 benchmark
(`benchmarks/remesh_infinity_navier_stokes_3d_taylor_green.py`,
results commit `1a5a38e4`) at two new spatial resolutions n in {24, 32}
with a *refined* F3 monotonicity test that excludes the two
IC-dominated observables identified in N12 §13.4
(`peak_vorticity_sup`, `peak_enstrophy`).

What this answers
-----------------
1. Does the dynamic non-monotonicity of N12 (U-shape in BKM(T),
   collapse of peak stretching at tau_g=inf) persist when F3 only
   counts observables that *can* respond to mixing?
2. Does the N12 signature survive one resolution doubling
   (n=16 -> n=32) and one intermediate step (n=24)?

A read-only `CROSS_RES_CONSISTENT` flag is pre-registered in §14.7:
it tags whether n=24 and n=32 yield the same per-resolution verdict
AND whether the sign of the tau_g=inf change in peak_stretching_post_t1
matches the n=16 N12 reading (collapse, i.e. drop > 50 %).

This file is the executable pre-registration of N13. Edits to its
locked constants (§14.5) or to the refined F3 definitions (§14.6) are
not permitted after the Results commit lands.

Locked configuration (§14.3)
----------------------------
* Resolutions: n in {24, 32} (outer loop)
* Viscosity nu = 0.01 (identical to N12)
* Time-step dt = 0.005 (identical to N12; satisfies CFL at both n)
* Final time T = 1.0 (200 steps)
* Amplitude A = 1.0
* Mixing weight alpha = 0.5
* tau_g sweep: [0, 8, 32, 128, "inf"]
* Seed label: 20260526 (cross-program session continuity)

Refined F-criteria (§14.6, §14.7)
---------------------------------
* F1 baseline fidelity: rel. err. on BKM(T) vs re-run of tau_g=0
  <= 1e-2 (identical to N12; per-resolution).
* F2 measurable response: max rel. change on F3 observable set
  (see below) >= 5 % (identical to N12 threshold; new observable set).
* F3 monotonicity (REFINED): test restricted to
  {BKM_T, peak_stretching, peak_stretching_post_t1, peak_enstrophy_post_t1}.
  EXCLUDED from F3: peak_vorticity_sup, peak_enstrophy (IC-dominated).
* F4 energy non-injection: max kinetic-energy excess vs IC
  <= 1e-10 (identical to N12).
* F5 divergence control: max |div(u)| <= 1e-8 (identical to N12).

Verdict mapping (locked, per-resolution, §14.7)
-----------------------------------------------
* STRUCTURAL_EFFECT_MONOTONE: F1+F2+F4+F5 PASS, F3 = MONOTONE
* STRUCTURAL_EFFECT_NON_MONOTONE: F1+F2+F4+F5 PASS, F3 = NON_MONOTONE
* NULL_RESULT: F1+F4+F5 PASS, F2 fails, F3 = FLAT
* INDETERMINATE_INFRA_FAIL: any of F1/F4/F5 fails
* INDETERMINATE_OTHER: any other combination

Cross-resolution flag (read-only, §14.7)
----------------------------------------
* CROSS_RES_CONSISTENT iff:
    (a) per-resolution verdicts at n=24 and n=32 agree, AND
    (b) sign of (peak_stretching_post_t1[tau=inf] - baseline) at both
        resolutions matches the n=16 N12 sign (collapse, i.e. drop
        relative to baseline > 50 %).
* CROSS_RES_INCONSISTENT otherwise.

This flag is NOT a verdict gate and does NOT strengthen or weaken §11
by itself.

Scope (§14.9)
-------------
N13 is a single-axis (resolution) extension of N12 with a corrected F3.
It does NOT:
* Close any of NS-G1..G5,
* Prove or disprove Clay Millennium 3D NS regularity,
* Establish a continuum limit, an asymptotic-in-n trend, or any
  analytical bound,
* Promote any new canonical operator.
* NULL_RESULT is a fully acceptable verdict.

Reproducibility
---------------
* Seed label: 20260526 (cross-program continuity with N12 and Riemann
  R-inf-1b benchmarks). No stochastic RNG is used.
* Run with: python benchmarks/remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.py
* Results JSON: results/remesh_infinity/remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
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
# Locked configuration (§14.3, §14.5)
# ---------------------------------------------------------------------------
N_SWEEP: list[int] = [24, 32]
VISCOSITY = 0.01
DT = 0.005
T_FINAL = 1.0
STEPS = int(round(T_FINAL / DT))  # 200
AMPLITUDE = 1.0
ALPHA = 0.5  # canonical REMESH_ALPHA default
TAU_G_SWEEP: list[Any] = [0, 8, 32, 128, "inf"]
SEED_LABEL = 20260526
EPS = 1e-14

# F-criteria thresholds (identical to N12 §12.6, except F3 observable set)
F1_TOL = 0.01
F2_MIN_REL_CHANGE = 0.05
F4_KE_INJECTION_TOL = 1e-10
F5_DIV_TOL = 1e-8

# §14.6 — observables excluded from F3 (IC-dominated, identified in §13.4)
F3_REFINED_EXCLUDED = ["peak_vorticity_sup", "peak_enstrophy"]
# §14.6 — observables included in F3 (post-IC, dynamically active)
F3_REFINED_INCLUDED = [
    "BKM_T",
    "peak_stretching",
    "peak_stretching_post_t1",
    "peak_enstrophy_post_t1",
]

# N12 reference for cross-resolution flag (§14.7)
# Source: §13.3 numerical table.
N12_PEAK_STRETCHING_BASELINE = 14.483747
N12_PEAK_STRETCHING_TAU_INF = 0.077555
N12_COLLAPSE_DROP_REL = (
    N12_PEAK_STRETCHING_BASELINE - N12_PEAK_STRETCHING_TAU_INF
) / max(
    N12_PEAK_STRETCHING_BASELINE, EPS
)  # 0.9946; sign = COLLAPSE


# ---------------------------------------------------------------------------
# REMESH-infinity execution (functionally identical to N12)
# ---------------------------------------------------------------------------
def run_one_tau(
    n: int,
    tau_g: Any,
    *,
    initial_state: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the 3D TG simulation at resolution n with REMESH-global at lag tau_g.

    Parameters
    ----------
    n : int
        Grid resolution per dimension (number of nodes = n**3).
    tau_g : int or "inf"
        Lag in step units, or the literal string "inf" for the
        asymptotic limit (lag-to-initial-condition).
    initial_state : np.ndarray
        Shape (3, n**3). Reused as the tau_g="inf" mixing reference
        and as the IC for every run at this resolution.

    Returns
    -------
    dict of numpy arrays (length STEPS + 1 each)
    """
    G = build_torus_graph_3d(n)
    op = TNFRNavierStokesOperator(graph=G, viscosity=VISCOSITY, dimension=3)
    op.set_components(initial_state)
    op.project_incompressible()

    history: deque[np.ndarray] | None = None
    if isinstance(tau_g, int) and tau_g > 0:
        history = deque(maxlen=tau_g + 1)
        history.append(op.phi.copy())

    n_pts = STEPS + 1
    time_axis = np.zeros(n_pts, dtype=float)
    vort_sup = np.zeros(n_pts, dtype=float)
    bkm_int = np.zeros(n_pts, dtype=float)
    enstrophy = np.zeros(n_pts, dtype=float)
    kinetic = np.zeros(n_pts, dtype=float)
    dissipation = np.zeros(n_pts, dtype=float)
    divergence = np.zeros(n_pts, dtype=float)
    stretching = np.zeros(n_pts, dtype=float)

    time_axis[0] = op.time
    vort_sup[0] = op.vorticity_sup_norm()
    enstrophy[0] = op.enstrophy_curl()
    kinetic[0] = op.kinetic_energy()
    dissipation[0] = op.dissipation_rate()
    divergence[0] = op.divergence_residual()
    stretching[0] = op.stretching_production()

    for k in range(1, STEPS + 1):
        op.step(DT, advection=True, incompressible=True)

        if tau_g == 0:
            mixed = False
        elif tau_g == "inf":
            op.phi = (1.0 - ALPHA) * op.phi + ALPHA * initial_state
            mixed = True
        else:
            assert history is not None
            if len(history) >= tau_g + 1:
                phi_past = history[0]
                op.phi = (1.0 - ALPHA) * op.phi + ALPHA * phi_past
                mixed = True
            else:
                mixed = False
            history.append(op.phi.copy())

        if mixed:
            op.project_incompressible()

        time_axis[k] = op.time
        vort_sup[k] = op.vorticity_sup_norm()
        bkm_int[k] = bkm_int[k - 1] + 0.5 * DT * (vort_sup[k - 1] + vort_sup[k])
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
# Refined F-criteria evaluation (§14.6, §14.7)
# ---------------------------------------------------------------------------
def _peak(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def _peak_post_t1(arr: np.ndarray) -> float:
    """Peak of |arr| restricted to k >= 1 (excludes IC at k=0)."""
    if arr.size <= 1:
        return 0.0
    return float(np.max(np.abs(arr[1:])))


def evaluate_criteria_at_n(
    n: int,
    runs: dict[str, dict[str, np.ndarray]],
    initial_state: np.ndarray,
) -> dict[str, Any]:
    """Evaluate refined F1-F5 pre-registered criteria at one resolution."""
    baseline_key = "tau_g=0"
    baseline = runs[baseline_key]

    # F1: re-run tau_g=0 and compare BKM(T)
    ref_run = run_one_tau(n, 0, initial_state=initial_state)
    bkm_baseline = float(baseline["bkm_integral"][-1])
    bkm_reference = float(ref_run["bkm_integral"][-1])
    f1_rel_err = abs(bkm_baseline - bkm_reference) / max(abs(bkm_reference), EPS)
    f1_pass = f1_rel_err <= F1_TOL

    # All observables (full table for telemetry; F3 uses only included ones)
    peak_table: dict[str, dict[str, float]] = {}
    for key, run in runs.items():
        peak_table[key] = {
            "peak_vorticity_sup": _peak(run["vorticity_sup"]),
            "BKM_T": float(run["bkm_integral"][-1]),
            "peak_enstrophy": _peak(run["enstrophy"]),
            "peak_stretching": _peak(run["stretching_production"]),
            "peak_stretching_post_t1": _peak_post_t1(run["stretching_production"]),
            "peak_enstrophy_post_t1": _peak_post_t1(run["enstrophy"]),
        }

    # F2: measurable response on the F3 INCLUDED set (refined)
    baseline_peaks = peak_table[baseline_key]
    f2_changes: dict[str, dict[str, float]] = {}
    max_rel_change = 0.0
    for key in runs:
        if key == baseline_key:
            continue
        rels = {}
        for obs_label in F3_REFINED_INCLUDED:
            base_val = baseline_peaks[obs_label]
            new_val = peak_table[key][obs_label]
            rel = abs(new_val - base_val) / max(abs(base_val), EPS)
            rels[obs_label] = rel
            if rel > max_rel_change:
                max_rel_change = rel
        f2_changes[key] = rels
    f2_pass = max_rel_change >= F2_MIN_REL_CHANGE

    # F3: monotonicity across {tau_g=8, 32, 128, inf} on INCLUDED set only
    mixing_keys = ["tau_g=8", "tau_g=32", "tau_g=128", "tau_g=inf"]
    f3_monotone_observables: list[str] = []
    for obs_label in F3_REFINED_INCLUDED:
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

    # F4: energy non-injection
    f4_max_excess = 0.0
    for key, run in runs.items():
        ke = run["kinetic_energy"]
        excess = float(np.max(ke) - ke[0])
        rel_excess = excess / max(abs(ke[0]), EPS)
        if rel_excess > f4_max_excess:
            f4_max_excess = rel_excess
    f4_pass = f4_max_excess <= F4_KE_INJECTION_TOL

    # F5: divergence control
    f5_max_div = max(float(np.max(run["divergence"])) for run in runs.values())
    f5_pass = f5_max_div <= F5_DIV_TOL

    # Verdict mapping (§14.7)
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
        "n": n,
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
            "observable_set": list(F3_REFINED_INCLUDED),
        },
        "F3": {
            "class": f3_class,
            "monotone_observables": f3_monotone_observables,
            "observable_set": list(F3_REFINED_INCLUDED),
            "excluded_observables": list(F3_REFINED_EXCLUDED),
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


def cross_resolution_flag(
    eval_n24: dict[str, Any],
    eval_n32: dict[str, Any],
) -> dict[str, Any]:
    """Pre-registered read-only flag (§14.7)."""
    verdict_agree = eval_n24["verdict"] == eval_n32["verdict"]

    def _collapse_sign(eval_block: dict[str, Any]) -> str:
        base = eval_block["peak_table"]["tau_g=0"]["peak_stretching_post_t1"]
        tinf = eval_block["peak_table"]["tau_g=inf"]["peak_stretching_post_t1"]
        if base <= EPS:
            return "TRIVIAL"
        drop_rel = (base - tinf) / base
        if drop_rel > 0.5:
            return "COLLAPSE"
        if drop_rel < -0.5:
            return "GROWTH"
        return "FLAT"

    sign_n24 = _collapse_sign(eval_n24)
    sign_n32 = _collapse_sign(eval_n32)
    n12_sign = "COLLAPSE"  # §13.4 reading at n=16

    sign_consistent = sign_n24 == n12_sign and sign_n32 == n12_sign
    flag = (
        "CROSS_RES_CONSISTENT"
        if (verdict_agree and sign_consistent)
        else "CROSS_RES_INCONSISTENT"
    )

    return {
        "flag": flag,
        "verdict_agree": verdict_agree,
        "verdict_n24": eval_n24["verdict"],
        "verdict_n32": eval_n32["verdict"],
        "sign_n24": sign_n24,
        "sign_n32": sign_n32,
        "sign_n16_n12_reference": n12_sign,
        "sign_consistent": sign_consistent,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    print("=" * 76)
    print(
        "N13: REMESH-infinity resolution extension on 3D Taylor-Green " "(refined F3)"
    )
    print("=" * 76)
    print(
        f"Config: N_SWEEP={N_SWEEP}, nu={VISCOSITY}, dt={DT}, T={T_FINAL} "
        f"(STEPS={STEPS}), alpha={ALPHA}"
    )
    print(f"tau_g sweep: {TAU_G_SWEEP}")
    print(f"Seed label: {SEED_LABEL}")
    print(f"F3 INCLUDED: {F3_REFINED_INCLUDED}")
    print(f"F3 EXCLUDED: {F3_REFINED_EXCLUDED}")
    print()

    results_by_n: dict[int, dict[str, Any]] = {}
    timings_by_n: dict[int, dict[str, float]] = {}
    evals_by_n: dict[int, dict[str, Any]] = {}

    for n in N_SWEEP:
        print(f"--- Resolution n = {n} (degrees of freedom: {3 * n**3}) ---")
        G0 = build_torus_graph_3d(n)
        op0 = TNFRNavierStokesOperator(graph=G0, viscosity=VISCOSITY, dimension=3)
        op0.set_taylor_green(AMPLITUDE)
        initial_state = op0.phi.copy()

        runs: dict[str, dict[str, np.ndarray]] = {}
        timings: dict[str, float] = {}
        for tau_g in TAU_G_SWEEP:
            label = f"tau_g={tau_g}"
            print(f"  Running n={n}, {label} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            run = run_one_tau(n, tau_g, initial_state=initial_state)
            elapsed = time.perf_counter() - t0
            timings[label] = elapsed
            runs[label] = run
            print(
                f"done ({elapsed:.1f} s). "
                f"BKM(T)={run['bkm_integral'][-1]:.4f}, "
                f"peak stretching post-t1="
                f"{_peak_post_t1(run['stretching_production']):.4f}, "
                f"max div={float(np.max(run['divergence'])):.2e}"
            )

        print(f"  Evaluating refined F1-F5 at n={n} ...")
        eval_result = evaluate_criteria_at_n(n, runs, initial_state)

        # Console summary
        f1 = eval_result["F1"]
        print(
            f"    F1 baseline fidelity     : "
            f"{'PASS' if f1['pass'] else 'FAIL'}  "
            f"(rel_err={f1['rel_err']:.2e}, tol={f1['tol']})"
        )
        f2 = eval_result["F2"]
        print(
            f"    F2 measurable response   : "
            f"{'PASS' if f2['pass'] else 'FAIL'}  "
            f"(max_rel_change={f2['max_rel_change']:.4f}, "
            f"threshold={f2['threshold']})"
        )
        f3 = eval_result["F3"]
        print(
            f"    F3 monotonicity (refined): {f3['class']}  "
            f"(monotone observables: {f3['monotone_observables']})"
        )
        f4 = eval_result["F4"]
        print(
            f"    F4 energy non-injection  : "
            f"{'PASS' if f4['pass'] else 'FAIL'}  "
            f"(max_rel_excess={f4['max_relative_excess']:.2e}, "
            f"tol={f4['tol']})"
        )
        f5 = eval_result["F5"]
        print(
            f"    F5 divergence control    : "
            f"{'PASS' if f5['pass'] else 'FAIL'}  "
            f"(max_div={f5['max_divergence']:.2e}, tol={f5['tol']})"
        )
        print(f"    -> verdict (n={n}): {eval_result['verdict']}")
        print()

        results_by_n[n] = {
            "peak_table": eval_result["peak_table"],
            "verdict": eval_result["verdict"],
            "timings_seconds": timings,
        }
        timings_by_n[n] = timings
        evals_by_n[n] = eval_result

    # Cross-resolution flag (§14.7)
    print("=" * 76)
    cross_flag = cross_resolution_flag(evals_by_n[24], evals_by_n[32])
    print(f"Cross-resolution flag (read-only): {cross_flag['flag']}")
    print(
        f"  verdicts: n=24 -> {cross_flag['verdict_n24']}, "
        f"n=32 -> {cross_flag['verdict_n32']} "
        f"(agree: {cross_flag['verdict_agree']})"
    )
    print(
        f"  stretching sign: n=24 -> {cross_flag['sign_n24']}, "
        f"n=32 -> {cross_flag['sign_n32']}, "
        f"n=16(N12) -> {cross_flag['sign_n16_n12_reference']} "
        f"(consistent: {cross_flag['sign_consistent']})"
    )
    print("=" * 76)

    # Convert numpy arrays to lists for JSON
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    output = {
        "milestone": "N13",
        "preregistration_section": ("theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md §14"),
        "config": {
            "N_SWEEP": N_SWEEP,
            "viscosity": VISCOSITY,
            "dt": DT,
            "t_final": T_FINAL,
            "steps": STEPS,
            "amplitude": AMPLITUDE,
            "alpha": ALPHA,
            "tau_g_sweep": TAU_G_SWEEP,
            "seed_label": SEED_LABEL,
            "f3_refined_included": F3_REFINED_INCLUDED,
            "f3_refined_excluded": F3_REFINED_EXCLUDED,
        },
        "per_resolution": {str(n): _convert(evals_by_n[n]) for n in N_SWEEP},
        "cross_resolution_flag": _convert(cross_flag),
        "timings_seconds": {str(n): timings_by_n[n] for n in N_SWEEP},
    }

    out_dir = _REPO_ROOT / "results" / "remesh_infinity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results JSON written to: {out_path}")

    return output


if __name__ == "__main__":
    main()
