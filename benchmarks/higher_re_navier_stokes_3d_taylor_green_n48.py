"""N14: Higher-Re CF eigenframe sweep on 3D Taylor-Green at n=48.

Pre-registration: theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md §16
Extends N11 (examples/86, n=24) by one resolution doubling
(n=24 -> n=48) and one new viscosity step (nu=0.002, Re_eff~3142),
unreachable at n=24 due to the Kolmogorov scale.

Helpers (velocity_grids, strain_tensor, alignment_diagnostics) are
ported verbatim from N11 to preserve cross-resolution comparability.

No REMESH. Clean Reynolds-only probe of:
  - F1 INCOMP: max ||div||_L2 <= 1e-8 in every run
  - F2 Finite: BKM(T), max Z, max |omega|_inf, mean P all finite
  - F3 Re-monotone <P> at n=48 ascending across the 5-nu sweep
  - F4 CF e_lambda2 dominance probe at nu=0.002
  - F5 cross-N11 sign consistency on shared nu values

Verdict mapping locked in §16.7.

Honest scope (§16.9): single-resolution one-axis extension of N11 with
one new ν step. NOT a continuum-limit study, NOT a NS-G4 closure,
NOT a fully-turbulent DNS study. Does NOT close NS-G1..G5.

Output: results/reynolds_sweep/higher_re_navier_stokes_3d_taylor_green_n48.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Repo-root path resolution (mirror N13 pattern)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from tnfr.navier_stokes.operator import TNFRNavierStokesOperator, build_torus_graph_3d

# ---------------------------------------------------------------------------
# Locked configuration (§16.3, §16.5)
# ---------------------------------------------------------------------------
N = 48
DT = 0.005
T_FINAL = 1.0
STEPS = int(round(T_FINAL / DT))  # 200
AMPLITUDE = 1.0
VISCOSITY_SWEEP: list[float] = [0.05, 0.02, 0.01, 0.005, 0.002]
RECORD_EVERY = 40  # 5 snapshots over the 200-step run
DIVERGENCE_PROBE_EVERY = 50
HIGH_VORT_QUANTILE = 0.75  # top 25% of |omega| nodes
ISOTROPIC_BASELINE = 1.0 / 3.0
EPS = 1e-14
INCOMP_TOL = 1e-8
SEED_LABEL = 20260526

# N11 reference series (examples/86, commit afc65b49) for F5 + cross-res flag
N11_NU_SHARED: list[float] = [0.05, 0.02, 0.01, 0.005]
N11_MEAN_P_AT_N24: list[float] = [5.74, 7.65, 8.44, 8.88]

# F-criteria thresholds (§16.6)
F1_DIV_TOL = INCOMP_TOL


# ---------------------------------------------------------------------------
# Helpers (ported VERBATIM from N11 / examples/86 to preserve comparability)
# ---------------------------------------------------------------------------
def velocity_grids(
    op: TNFRNavierStokesOperator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        op._component_grid_3d(0, n),
        op._component_grid_3d(1, n),
        op._component_grid_3d(2, n),
    )


def strain_tensor(u: np.ndarray, v: np.ndarray, w: np.ndarray, h: float) -> np.ndarray:
    def d(arr: np.ndarray, axis: int) -> np.ndarray:
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * h)

    du = (d(u, 0), d(u, 1), d(u, 2))
    dv = (d(v, 0), d(v, 1), d(v, 2))
    dw = (d(w, 0), d(w, 1), d(w, 2))
    grad = [du, dv, dw]  # grad[a][i] = d_i u_a

    n0, n1, n2 = u.shape
    S = np.empty((n0, n1, n2, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            S[..., i, j] = 0.5 * (grad[j][i] + grad[i][j])
    return S


def alignment_diagnostics(
    omega: np.ndarray,
    S: np.ndarray,
    high_vort_quantile: float,
) -> dict[str, float]:
    """CF alignment diagnostics in the high-vorticity region.

    omega: raw (3, n, n, n) layout (output of op.vorticity_3d()).
    S    : (n, n, n, 3, 3) symmetric strain.
    """
    om = np.moveaxis(omega, 0, -1)
    mag = np.linalg.norm(om, axis=-1)

    eigvals, eigvecs = np.linalg.eigh(S)

    order = np.argsort(eigvals, axis=-1)[..., ::-1]
    lam = np.take_along_axis(eigvals, order, axis=-1)
    idx = order[..., np.newaxis, :]
    vecs = np.take_along_axis(eigvecs, np.broadcast_to(idx, eigvecs.shape), axis=-1)

    safe_mag = np.where(mag > EPS, mag, 1.0)
    om_hat = om / safe_mag[..., np.newaxis]

    cos = np.einsum("...i,...ij->...j", om_hat, vecs)
    cos2 = cos**2

    sigma_eff = np.einsum("...k,...k->...", lam, cos2)
    lam_max_abs = np.maximum(np.abs(lam[..., 0]), EPS)
    depletion = 1.0 - sigma_eff / lam_max_abs

    threshold = np.quantile(mag, high_vort_quantile)
    high = mag >= threshold

    return {
        "max_mag_omega": float(mag.max()),
        "mean_mag_omega": float(mag.mean()),
        "mean_cos2_lambda1": float(cos2[..., 0][high].mean()),
        "mean_cos2_lambda2": float(cos2[..., 1][high].mean()),
        "mean_cos2_lambda3": float(cos2[..., 2][high].mean()),
        "mean_sigma_eff_high": float(sigma_eff[high].mean()),
        "mean_lambda1_high": float(lam[..., 0][high].mean()),
        "mean_depletion_high": float(depletion[high].mean()),
    }


# ---------------------------------------------------------------------------
# Per-viscosity run
# ---------------------------------------------------------------------------
def run_at_viscosity(nu: float) -> dict[str, Any]:
    G = build_torus_graph_3d(N)
    op = TNFRNavierStokesOperator(G, viscosity=nu, dimension=3)
    op.set_taylor_green(AMPLITUDE)
    op.project_incompressible()

    h = 2.0 * math.pi / N

    times: list[float] = []
    snapshots: list[dict[str, Any]] = []
    P_series: list[float] = []
    Z_series: list[float] = []
    omega_inf_series: list[float] = []
    max_div_observed = float(op.divergence_residual())

    def record(t: float) -> None:
        omega = op.vorticity_3d()
        u, v, w = velocity_grids(op, N)
        S = strain_tensor(u, v, w, h)
        d = alignment_diagnostics(omega, S, HIGH_VORT_QUANTILE)
        d["t"] = float(t)
        d["stretching_production"] = float(op.stretching_production())
        snapshots.append(d)
        times.append(t)
        mag = np.linalg.norm(np.moveaxis(omega, 0, -1), axis=-1)
        omega_inf_series.append(float(mag.max()))
        Z_series.append(float(0.5 * (mag**2).sum() * (h**3)))
        P_series.append(float(op.stretching_production()))

    record(0.0)

    # BKM integral via trapezoidal accumulation on per-step vorticity sup norm
    bkm = 0.0
    vort_prev = float(op.vorticity_sup_norm())
    max_omega_inf_all = vort_prev
    max_Z = float(op.enstrophy_curl())

    for k in range(1, STEPS + 1):
        op.step(DT, advection=True, incompressible=True)

        vort_cur = float(op.vorticity_sup_norm())
        bkm += 0.5 * DT * (vort_prev + vort_cur)
        vort_prev = vort_cur

        max_omega_inf_all = max(max_omega_inf_all, vort_cur)
        max_Z = max(max_Z, float(op.enstrophy_curl()))

        if k % DIVERGENCE_PROBE_EVERY == 0:
            div = float(op.divergence_residual())
            if div > max_div_observed:
                max_div_observed = div

        if k % RECORD_EVERY == 0:
            record(k * DT)

    max_div_observed = max(max_div_observed, float(op.divergence_residual()))

    mean_P = float(np.mean(P_series))
    mean_cos2_l1 = float(np.mean([s["mean_cos2_lambda1"] for s in snapshots]))
    mean_cos2_l2 = float(np.mean([s["mean_cos2_lambda2"] for s in snapshots]))
    mean_cos2_l3 = float(np.mean([s["mean_cos2_lambda3"] for s in snapshots]))
    mean_depletion = float(np.mean([s["mean_depletion_high"] for s in snapshots]))

    re_eff = float(AMPLITUDE / nu)

    return {
        "nu": nu,
        "re_eff": re_eff,
        "max_div_L2": max_div_observed,
        "BKM_T": float(bkm),
        "max_Z": max_Z,
        "max_omega_inf": max_omega_inf_all,
        "mean_P": mean_P,
        "mean_cos2_lambda1": mean_cos2_l1,
        "mean_cos2_lambda2": mean_cos2_l2,
        "mean_cos2_lambda3": mean_cos2_l3,
        "mean_depletion_high": mean_depletion,
        "P_series": P_series,
        "Z_series": Z_series,
        "omega_inf_series": omega_inf_series,
        "times": times,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# F-criteria + verdict (§16.6, §16.7)
# ---------------------------------------------------------------------------
def evaluate_criteria(runs: list[dict[str, Any]]) -> dict[str, Any]:
    nus = [r["nu"] for r in runs]
    assert (
        nus == VISCOSITY_SWEEP
    ), f"Run order mismatch: {nus} vs locked {VISCOSITY_SWEEP}"

    # F1: INCOMP control
    max_divs = [r["max_div_L2"] for r in runs]
    f1_pass = all(d <= F1_DIV_TOL for d in max_divs)

    # F2: finiteness
    def _finite(x: float) -> bool:
        return bool(np.isfinite(x)) and not bool(np.isnan(x))

    f2_pass = all(
        _finite(r["BKM_T"])
        and _finite(r["max_Z"])
        and _finite(r["max_omega_inf"])
        and _finite(r["mean_P"])
        for r in runs
    )

    # F3: Re-monotone <P> ascending across 5-nu sweep (nu descending)
    mean_Ps = [r["mean_P"] for r in runs]
    p_diffs = list(np.diff(mean_Ps))
    f3_pass = all(d >= 0.0 for d in p_diffs)

    # F4: CF e_lambda2 dominance probe at Re_eff_max (nu=0.002, last run)
    run_max_re = runs[-1]
    cos2_l1 = run_max_re["mean_cos2_lambda1"]
    cos2_l2 = run_max_re["mean_cos2_lambda2"]
    cos2_l3 = run_max_re["mean_cos2_lambda3"]
    f4_e2_over_e1 = cos2_l2 > cos2_l1
    f4_e2_over_e3 = cos2_l2 > cos2_l3
    f4_satisfied = bool(f4_e2_over_e1 and f4_e2_over_e3)

    # F5: cross-N11 endpoint sign consistency (nu=0.005 vs nu=0.05)
    n11_endpoint_diff = N11_MEAN_P_AT_N24[-1] - N11_MEAN_P_AT_N24[0]
    n14_endpoint_diff = mean_Ps[3] - mean_Ps[0]
    f5_pass = bool(np.sign(n14_endpoint_diff) == np.sign(n11_endpoint_diff))

    # Cross-resolution flag (read-only)
    n11_diffs_shared = list(np.diff(N11_MEAN_P_AT_N24))
    n14_diffs_shared = list(np.diff(mean_Ps[:4]))
    sign_match_all = all(
        np.sign(d14) == np.sign(d11) and d14 != 0 and d11 != 0
        for d14, d11 in zip(n14_diffs_shared, n11_diffs_shared)
    )
    cross_res_flag = (
        "RE_TREND_CONSISTENT_WITH_N11"
        if sign_match_all
        else "RE_TREND_INCONSISTENT_WITH_N11"
    )

    # Verdict mapping (§16.7)
    if not f1_pass or not f2_pass:
        verdict = "INDETERMINATE_INFRA_FAIL"
    elif not f3_pass:
        verdict = "N11_RE_MONOTONICITY_REFUTED"
    elif not f5_pass:
        verdict = "N11_RESOLUTION_INCONSISTENT"
    elif f4_satisfied:
        verdict = "CF_EIGENFRAME_TRANSITION_OBSERVED"
    elif f1_pass and f2_pass and f3_pass and f5_pass and not f4_satisfied:
        verdict = "CF_EIGENFRAME_TRANSITION_NOT_OBSERVED_AT_REEFF_3142"
    else:
        verdict = "INDETERMINATE_OTHER"

    return {
        "verdict": verdict,
        "F1": {
            "pass": bool(f1_pass),
            "tol": F1_DIV_TOL,
            "max_div_per_run": max_divs,
        },
        "F2": {
            "pass": bool(f2_pass),
            "BKM_T_per_run": [r["BKM_T"] for r in runs],
            "max_Z_per_run": [r["max_Z"] for r in runs],
            "max_omega_inf_per_run": [r["max_omega_inf"] for r in runs],
            "mean_P_per_run": mean_Ps,
        },
        "F3": {
            "pass": bool(f3_pass),
            "mean_P_sequence": mean_Ps,
            "p_diffs": [float(d) for d in p_diffs],
        },
        "F4": {
            "satisfied": f4_satisfied,
            "nu_probed": run_max_re["nu"],
            "re_eff_probed": run_max_re["re_eff"],
            "cos2_lambda1": cos2_l1,
            "cos2_lambda2": cos2_l2,
            "cos2_lambda3": cos2_l3,
            "e2_over_e1": bool(f4_e2_over_e1),
            "e2_over_e3": bool(f4_e2_over_e3),
        },
        "F5": {
            "pass": f5_pass,
            "n11_endpoint_diff": float(n11_endpoint_diff),
            "n14_endpoint_diff": float(n14_endpoint_diff),
            "n11_mean_P_at_n24": N11_MEAN_P_AT_N24,
            "shared_nu": N11_NU_SHARED,
        },
        "cross_resolution_flag": {
            "flag": cross_res_flag,
            "n11_diffs_shared": [float(d) for d in n11_diffs_shared],
            "n14_diffs_shared": [float(d) for d in n14_diffs_shared],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    print("=" * 76)
    print("N14: Higher-Re CF eigenframe sweep on 3D Taylor-Green at n=48")
    print("=" * 76)
    print(
        f"Config: N={N}, nu_sweep={VISCOSITY_SWEEP}, dt={DT}, T={T_FINAL} "
        f"(STEPS={STEPS}), A={AMPLITUDE}"
    )
    print("  REMESH=OFF, INCOMP=ON, advection=ON")
    print(f"  Seed label: {SEED_LABEL}")
    print("  Pre-registration: theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md §16")
    print()

    runs: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    for nu in VISCOSITY_SWEEP:
        label = f"nu={nu}"
        print(
            f"  Running {label} (Re_eff={AMPLITUDE / nu:.0f}) ...",
            end=" ",
            flush=True,
        )
        t0 = time.perf_counter()
        run = run_at_viscosity(nu)
        elapsed = time.perf_counter() - t0
        timings[label] = elapsed
        runs.append(run)
        print(
            f"done ({elapsed:.1f} s). "
            f"<P>={run['mean_P']:.3f}, "
            f"cos2(l1,l2,l3)=({run['mean_cos2_lambda1']:.3f},"
            f"{run['mean_cos2_lambda2']:.3f},"
            f"{run['mean_cos2_lambda3']:.3f}), "
            f"BKM(T)={run['BKM_T']:.3f}, "
            f"max_div={run['max_div_L2']:.2e}"
        )

    print()
    print("  Evaluating locked F1..F5 (§16.6, §16.7) ...")
    eval_result = evaluate_criteria(runs)

    f1 = eval_result["F1"]
    f2 = eval_result["F2"]
    f3 = eval_result["F3"]
    f4 = eval_result["F4"]
    f5 = eval_result["F5"]
    cflag = eval_result["cross_resolution_flag"]

    print(
        f"    F1 INCOMP control          : "
        f"{'PASS' if f1['pass'] else 'FAIL'}  "
        f"(max max_div = {max(f1['max_div_per_run']):.2e}, tol={f1['tol']})"
    )
    print(f"    F2 finiteness              : " f"{'PASS' if f2['pass'] else 'FAIL'}")
    print(
        f"    F3 Re-monotone <P>         : "
        f"{'PASS' if f3['pass'] else 'FAIL'}  "
        f"(<P> seq = {[round(x, 3) for x in f3['mean_P_sequence']]}, "
        f"diffs = {[round(x, 3) for x in f3['p_diffs']]})"
    )
    print(
        f"    F4 CF e_lambda2 dominance  : "
        f"{'SATISFIED' if f4['satisfied'] else 'NOT SATISFIED'}  "
        f"(at nu={f4['nu_probed']}, Re_eff={f4['re_eff_probed']:.0f}; "
        f"cos2 l2 vs l1: {f4['e2_over_e1']}; l2 vs l3: {f4['e2_over_e3']})"
    )
    print(
        f"    F5 cross-N11 sign          : "
        f"{'PASS' if f5['pass'] else 'FAIL'}  "
        f"(N14 endpoint diff = {f5['n14_endpoint_diff']:+.3f}; "
        f"N11 endpoint diff = {f5['n11_endpoint_diff']:+.3f})"
    )
    print(
        f"    Cross-res flag (read-only) : {cflag['flag']}  "
        f"(N14 shared diffs = {[round(x, 3) for x in cflag['n14_diffs_shared']]}; "
        f"N11 = {[round(x, 3) for x in cflag['n11_diffs_shared']]})"
    )
    print()
    print(f"  ===> VERDICT: {eval_result['verdict']}")
    print()

    output = {
        "milestone": "N14",
        "preregistration_section": ("theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md §16"),
        "config": {
            "N": N,
            "dt": DT,
            "t_final": T_FINAL,
            "steps": STEPS,
            "amplitude": AMPLITUDE,
            "viscosity_sweep": VISCOSITY_SWEEP,
            "record_every": RECORD_EVERY,
            "divergence_probe_every": DIVERGENCE_PROBE_EVERY,
            "high_vorticity_quantile": HIGH_VORT_QUANTILE,
            "incomp_tol": INCOMP_TOL,
            "seed_label": SEED_LABEL,
            "n11_reference_nu": N11_NU_SHARED,
            "n11_reference_mean_P": N11_MEAN_P_AT_N24,
        },
        "verdict": eval_result["verdict"],
        "F1": eval_result["F1"],
        "F2": eval_result["F2"],
        "F3": eval_result["F3"],
        "F4": eval_result["F4"],
        "F5": eval_result["F5"],
        "cross_resolution_flag": eval_result["cross_resolution_flag"],
        "per_run_summary": [
            {
                "nu": r["nu"],
                "re_eff": r["re_eff"],
                "max_div_L2": r["max_div_L2"],
                "BKM_T": r["BKM_T"],
                "max_Z": r["max_Z"],
                "max_omega_inf": r["max_omega_inf"],
                "mean_P": r["mean_P"],
                "mean_cos2_lambda1": r["mean_cos2_lambda1"],
                "mean_cos2_lambda2": r["mean_cos2_lambda2"],
                "mean_cos2_lambda3": r["mean_cos2_lambda3"],
                "mean_depletion_high": r["mean_depletion_high"],
                "snapshot_count": r["snapshot_count"],
                "snapshots": r["snapshots"],
                "times": r["times"],
                "P_series": r["P_series"],
                "Z_series": r["Z_series"],
                "omega_inf_series": r["omega_inf_series"],
            }
            for r in runs
        ],
        "timings_seconds": timings,
    }

    out_dir = _REPO_ROOT / "results" / "reynolds_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "higher_re_navier_stokes_3d_taylor_green_n48.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Results JSON written to: {out_path}")
    return output


if __name__ == "__main__":
    main()
