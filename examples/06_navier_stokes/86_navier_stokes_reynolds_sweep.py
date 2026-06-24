"""N11 demo — Reynolds sweep: CF eigenframe shift across viscosity scales.

Status: empirical exploration; NOT a closure of NS-G4 or NS-G5.

Background
----------
N9 (commit fbbcaa34, examples/84) shipped the Constantin-Fefferman (CF)
geometric-depletion diagnostics on a 3D Taylor-Green vortex at fixed
viscosity ``nu = 0.05``. The high-vorticity alignment cosines came out as

    <cos^2(omega_hat, e_{lambda1})>_high  =  0.294
    <cos^2(omega_hat, e_{lambda2})>_high  =  0.342
    <cos^2(omega_hat, e_{lambda3})>_high  =  0.364   (most compressing)

with the isotropic baseline at 1/3 = 0.333. Vorticity AVOIDED the
most-stretching direction (good, depletion mechanism active) but
preferred the most-compressing one, NOT the intermediate one as in
classical CF turbulence literature. N9 noted this discrepancy and
attributed it to TG being in the LAMINAR regime at this Reynolds number
(Re_eff ~ A * L / nu ~ 1 * 2*pi / 0.05 ~ 126).

This demo (N11) is the natural follow-up: vary ``nu`` over a decade and
test whether the CF eigenframe shift toward e_{lambda2} preference
emerges as the effective Reynolds number grows, i.e. whether the
discrete TNFR-NS operator reproduces the classical laminar->transitional
CF signature.

TNFR positioning
----------------
Like N9, this is a PRECURSOR to NS-G4 (structural TNFR construction of
(omega . grad) u). It probes the operator's ability to traverse the
laminar->transitional Reynolds regime on the alignment side without
requiring any new operator code. PURELY EMPIRICAL.

Setup
-----
3D Taylor-Green vortex on a periodic torus graph, n=24, dt=0.005,
T=1.0 (200 steps), A=1.0, INCOMP+advection ON. Sweep

    nu in {0.05, 0.02, 0.01, 0.005}

giving Re_eff ~ {126, 314, 628, 1257}. At ``nu < 0.005`` the Kolmogorov
scale eta ~ (nu^3 / eps)^{1/4} falls below the grid spacing h = 2*pi/24
~ 0.262 and the simulation becomes UNRESOLVED, so the sweep STOPS at
nu = 0.005 to avoid grid-scale artefacts being mistaken for physics.
This is an EXPLORATORY mid-Reynolds probe, not a high-Re DNS.

At 5 evenly-spaced snapshots per run we record per-snapshot:

  * <cos^2(omega_hat, e_{lambda_k})>_high   for k = 1, 2, 3
  * <D>_high = depletion ratio (N9 definition)
  * stretching production P(t), enstrophy Z(t), ||omega||_inf
  * INCOMP residual (must stay machine-eps)

then time-mean each quantity over the run and tabulate the trend.

PASS criteria (4)
-----------------
  C1  INCOMP holds across the sweep: max ||div||_L2 <= 1e-8 in every run.
  C2  Enstrophy finite: max Z(t) < +inf in every run, BKM integral finite.
  C3  Stretching grows with Re: time-mean P(t) is MONOTONICALLY increasing
      as nu decreases (canonical: lower viscosity -> stronger nonlinear
      transfer -> larger vortex stretching).
  C4  Alignment evolves with Re: at least ONE of the three time-mean
      cos^2_lambda_k values shifts MONOTONICALLY across the sweep
      (detects whether the operator's alignment statistics respond to
      Reynolds at all).

Honest scope
------------
Probing a 4-point Re sweep at moderate resolution is NOT a closure of
NS-G4 (let alone NS-G5 / Clay):

  * The whole sweep is in the LAMINAR-to-LOW-TRANSITIONAL regime
    (max Re ~ 1300). Fully turbulent CF statistics live at Re ~ 1e4-1e6
    and require DNS at n >= 256.
  * TG is a single, symmetric, smooth initial condition. Real CF
    statistics in literature are averages over forced isotropic
    turbulence in statistical steady state.
  * Whether the operator REPRODUCES classical CF preference for e_{lambda2}
    at high Re is genuinely OPEN here. The C4 criterion only asks for
    monotonic shift, not specifically for e_{lambda2} dominance. Even a
    NULL result (no monotonic shift) is informative.
  * NS-G1..G5 all remain OPEN.

This demo is the simplest possible probe of "does the alignment respond
to Reynolds?". A positive answer strengthens the case for the operator
encoding genuine CF physics; a negative answer flags that more
resolution (n>=48) or longer evolution is required before the alignment
statistics develop.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from tnfr.navier_stokes.operator import TNFRNavierStokesOperator, build_torus_graph_3d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 24
DT = 0.005
T_FINAL = 1.0
STEPS = int(round(T_FINAL / DT))  # 200
AMPLITUDE = 1.0
VISCOSITY_SWEEP = [0.05, 0.02, 0.01, 0.005]
RECORD_EVERY = 40  # 5 snapshots per run
HIGH_VORT_QUANTILE = 0.75
ISOTROPIC_BASELINE = 1.0 / 3.0
EPS = 1e-14

INCOMP_TOL = 1e-8
DIVERGENCE_PROBE_EVERY = 50  # cheap sanity sample


# ---------------------------------------------------------------------------
# Helpers (mirror N9)
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
    h = 2.0 * math.pi / N

    times: list[float] = []
    diags: list[dict[str, float]] = []
    P_series: list[float] = []
    Z_series: list[float] = []
    omega_inf_series: list[float] = []
    max_div_observed = 0.0

    def record(t: float) -> None:
        omega = op.vorticity_3d()
        u, v, w = velocity_grids(op, N)
        S = strain_tensor(u, v, w, h)
        d = alignment_diagnostics(omega, S, HIGH_VORT_QUANTILE)
        diags.append(d)
        times.append(t)
        mag = np.linalg.norm(np.moveaxis(omega, 0, -1), axis=-1)
        omega_inf_series.append(float(mag.max()))
        Z_series.append(float(0.5 * (mag**2).sum() * (h**3)))
        P_series.append(op.stretching_production())

    record(0.0)
    for k in range(1, STEPS + 1):
        op.step(DT, advection=True, incompressible=True)
        if k % DIVERGENCE_PROBE_EVERY == 0:
            div = float(op.divergence_residual())  # already L2 norm
            if div > max_div_observed:
                max_div_observed = div
        if k % RECORD_EVERY == 0:
            record(k * DT)

    return {
        "nu": nu,
        "Re_eff": AMPLITUDE * 2.0 * math.pi / nu,
        "times": np.array(times),
        "diags": diags,
        "P_series": np.array(P_series),
        "Z_series": np.array(Z_series),
        "omega_inf_series": np.array(omega_inf_series),
        "max_div_observed": max_div_observed,
        "BKM_integral": float(np.trapezoid(omega_inf_series, times)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("N11 demo — Reynolds sweep: CF alignment vs viscosity")
    print("=" * 78)
    print(
        f"\nConfig: n={N}, dt={DT}, T={T_FINAL} ({STEPS} steps), A={AMPLITUDE}, "
        f"INCOMP+adv ON.\n"
        f"Viscosity sweep: {VISCOSITY_SWEEP}\n"
        f"Re_eff = A*L/nu = 2*pi/nu : "
        f"{[round(2 * math.pi / nu) for nu in VISCOSITY_SWEEP]}\n"
        f"Isotropic baseline <cos^2> = 1/3 = {ISOTROPIC_BASELINE:.4f}\n"
    )

    t0 = time.perf_counter()
    runs = []
    for nu in VISCOSITY_SWEEP:
        print(f"[run] nu = {nu:.4f}  Re_eff ~ {round(2 * math.pi / nu)} ...")
        r = run_at_viscosity(nu)
        runs.append(r)
        print(
            f"      Done. max ||div||_L2 = {r['max_div_observed']:.3e}, "
            f"max |P| = {np.abs(r['P_series']).max():.3e}, "
            f"max Z = {r['Z_series'].max():.3e}, "
            f"BKM = {r['BKM_integral']:.3e}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.2f}s\n")

    # ----------------------------------------------------- per-snapshot tables
    for r in runs:
        print("-" * 78)
        print(
            f"Per-snapshot diagnostics (nu = {r['nu']:.4f}, "
            f"Re_eff ~ {round(r['Re_eff'])})"
        )
        print("-" * 78)
        print(
            f"{'t':>6} {'||om||_inf':>11} {'<|om|>':>8} "
            f"{'cos^2_l1':>9} {'cos^2_l2':>9} {'cos^2_l3':>9} "
            f"{'<D>':>8} {'P(t)':>10} {'Z(t)':>10}"
        )
        for i, t in enumerate(r["times"]):
            d = r["diags"][i]
            print(
                f"{t:>6.3f} {r['omega_inf_series'][i]:>11.4e} "
                f"{d['mean_mag_omega']:>8.4f} "
                f"{d['mean_cos2_lambda1']:>9.4f} "
                f"{d['mean_cos2_lambda2']:>9.4f} "
                f"{d['mean_cos2_lambda3']:>9.4f} "
                f"{d['mean_depletion_high']:>8.4f} "
                f"{r['P_series'][i]:>10.4e} "
                f"{r['Z_series'][i]:>10.4e}"
            )
        print()

    # ---------------------------------------------- time-mean across the sweep
    print("=" * 78)
    print("Time-mean aggregates across viscosity sweep")
    print("=" * 78)
    print(
        f"{'nu':>8} {'Re_eff':>8} "
        f"{'<cos2_l1>':>10} {'<cos2_l2>':>10} {'<cos2_l3>':>10} "
        f"{'<D>':>8} {'<P>':>10} {'maxZ':>10} {'BKM':>10} {'maxDiv':>10}"
    )
    tmean_cos2_l1 = []
    tmean_cos2_l2 = []
    tmean_cos2_l3 = []
    tmean_P = []
    for r in runs:
        c1 = float(np.mean([d["mean_cos2_lambda1"] for d in r["diags"]]))
        c2 = float(np.mean([d["mean_cos2_lambda2"] for d in r["diags"]]))
        c3 = float(np.mean([d["mean_cos2_lambda3"] for d in r["diags"]]))
        D = float(np.mean([d["mean_depletion_high"] for d in r["diags"]]))
        Pmean = float(np.mean(r["P_series"]))
        tmean_cos2_l1.append(c1)
        tmean_cos2_l2.append(c2)
        tmean_cos2_l3.append(c3)
        tmean_P.append(Pmean)
        print(
            f"{r['nu']:>8.4f} {round(r['Re_eff']):>8d} "
            f"{c1:>10.4f} {c2:>10.4f} {c3:>10.4f} "
            f"{D:>8.4f} {Pmean:>10.4e} "
            f"{r['Z_series'].max():>10.4e} "
            f"{r['BKM_integral']:>10.4e} "
            f"{r['max_div_observed']:>10.3e}"
        )

    # ------------------------------------------------------------- PASS criteria
    print()
    print("=" * 78)
    print("PASS criteria")
    print("=" * 78)

    max_div = max(r["max_div_observed"] for r in runs)
    c1_pass = max_div <= INCOMP_TOL
    print(
        f"C1 (INCOMP across sweep): max ||div||_L2 = {max_div:.3e} "
        f"<= {INCOMP_TOL:.0e} -> {'PASS' if c1_pass else 'FAIL'}"
    )

    all_finite = all(
        np.isfinite(r["Z_series"]).all()
        and np.isfinite(r["BKM_integral"])
        and np.isfinite(r["P_series"]).all()
        for r in runs
    )
    c2_pass = all_finite
    print(
        f"C2 (finite enstrophy / BKM / P across sweep): "
        f"{'PASS' if c2_pass else 'FAIL'}"
    )

    # C3: time-mean P monotonically increasing as nu decreases
    # (P-series sorted by viscosity descending corresponds to indices
    #  in tmean_P given VISCOSITY_SWEEP is already descending)
    p_monotone = all(tmean_P[i] < tmean_P[i + 1] for i in range(len(tmean_P) - 1))
    c3_pass = p_monotone
    print(
        f"C3 (P monotonic with Re): <P> = "
        f"{[round(p, 4) for p in tmean_P]} -> "
        f"{'PASS' if c3_pass else 'FAIL'}"
    )

    # C4: at least one of the three cos2 means is monotonic across the sweep
    def is_monotone(seq: list[float]) -> bool:
        inc = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
        dec = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
        return inc or dec

    mono_l1 = is_monotone(tmean_cos2_l1)
    mono_l2 = is_monotone(tmean_cos2_l2)
    mono_l3 = is_monotone(tmean_cos2_l3)
    c4_pass = mono_l1 or mono_l2 or mono_l3
    flags = (
        f"l1={'MONO' if mono_l1 else 'no'}, "
        f"l2={'MONO' if mono_l2 else 'no'}, "
        f"l3={'MONO' if mono_l3 else 'no'}"
    )
    print(
        f"C4 (alignment responds to Re): {flags} -> " f"{'PASS' if c4_pass else 'FAIL'}"
    )

    print()
    n_pass = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print(f"OVERALL: {n_pass}/4 -> {'PASS' if n_pass == 4 else 'FAIL'}")

    # --------------------------------------------------------- structural read
    print()
    print("=" * 78)
    print("Structural interpretation")
    print("=" * 78)
    # Locate which eigenvector dominates at lowest viscosity
    lowest_idx = len(runs) - 1
    cos2_at_lowest = [
        tmean_cos2_l1[lowest_idx],
        tmean_cos2_l2[lowest_idx],
        tmean_cos2_l3[lowest_idx],
    ]
    dominant = int(np.argmax(cos2_at_lowest)) + 1
    print(
        f"At nu = {runs[lowest_idx]['nu']:.4f} "
        f"(Re_eff ~ {round(runs[lowest_idx]['Re_eff'])}):\n"
        f"  dominant eigenvector preference for omega_hat = "
        f"e_lambda_{dominant} "
        f"(<cos^2> = {max(cos2_at_lowest):.4f}, "
        f"baseline 1/3 = {ISOTROPIC_BASELINE:.4f})"
    )
    if dominant == 2:
        print(
            "  -> Matches classical Constantin-Fefferman intermediate-eigenvector\n"
            "     preference in fully turbulent flows."
        )
    elif dominant == 1:
        print(
            "  -> Alignment with MOST-stretching eigenvector. Anomalous vs.\n"
            "     classical CF; possible operator anomaly or insufficient Re."
        )
    elif dominant == 3:
        print(
            "  -> Alignment with MOST-compressing eigenvector. Consistent with\n"
            "     N9 finding at nu=0.05 and suggests TG at n=24 is still below\n"
            "     the transition where intermediate-eigenvector preference\n"
            "     emerges. Higher resolution (n>=48) or longer evolution\n"
            "     required to test CF-canonical eigenframe."
        )

    print()
    print("=" * 78)
    print("Honest scope")
    print("=" * 78)
    print(
        "This is an EMPIRICAL EXPLORATION at moderate Reynolds (~125-1300).\n"
        "It does NOT close NS-G4 (TNFR construction of (omega.grad)u) or any\n"
        "other NS-Gk. A PASS only certifies that the operator's CF-style\n"
        "alignment statistics RESPOND to the viscosity parameter; it does\n"
        "NOT certify that the response reproduces the classical fully-\n"
        "turbulent CF eigenframe preference, which requires DNS at n>=256\n"
        "with forced isotropic statistics in statistical steady state.\n"
        "\n"
        "Global regularity of 3D incompressible NS remains OPEN.\n"
        "NS-G1, NS-G2, NS-G3, NS-G4, NS-G5 all remain OPEN.\n"
    )


if __name__ == "__main__":
    main()
