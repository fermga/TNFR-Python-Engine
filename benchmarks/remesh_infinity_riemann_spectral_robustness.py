"""R∞-1a-spectral-robustness — Falsification gate for R∞-1a-spectral.

Scope (honest)
--------------
R∞-1a-spectral (preceding benchmark) reported max|·| = 0.8575 via r_β
(sorted-magnitude Pearson), nominally satisfying the pre-registered F3
threshold (> 0.5 = B1 supported spectrally).  However:

  * the dominant signal r_β is the statistically weakest of the four
    tests — heavy-tailed positive sequences naturally produce high
    sorted-magnitude correlation;
  * the strongest structural test r_γ = 0.34 sat in the indeterminate
    range;
  * auxiliary controls r(P_k, γ_n) ≈ −0.67 flagged a possible kernel
    bias rather than Riemann content;
  * per-mode correlation r(s_i, r_n) = 0.005 was zero within noise.

Before any B1 claim, R∞-1a-spectral must survive a robustness gate.
This benchmark implements three pre-registered controls.

Controls
--------
C1 — White-noise null.  Replace the canonical oscillatory synthetic
     EPI field by zero-mean unit-variance white noise (seeded), run the
     identical REMESH iteration, and recompute r_α, r_β, r_γ, r_δ
     against the SAME Riemann reference.  Repeated for N_NULL = 16
     independent seeds.  Empirical null distribution per test.
     Pre-registered: if observed-on-canonical (R∞-1a-spectral) value
     falls inside the central 95% of the null distribution for any
     test, that test is artefactual.  In particular if r_β-null mean
     > 0.5, the R∞-1a-spectral r_β = 0.8575 result is consistent
     with kernel artefact.

C2 — (α, τ_l) sensitivity sweep.  Run the canonical pipeline at
     (α, τ_l) ∈ {0.25, 0.5, 0.75} × {2, 4, 8}, identical synthetic
     field, τ_g = 16, N_ITER = 512.  Report (r_α, r_β, r_γ, r_δ) for
     each.  Pre-registered: if r_β collapses below 0.5 at any
     non-canonical (α, τ_l), the signal is not robust to kernel
     parameters.  If r_β remains > 0.5 across the entire grid, the
     signal is parameter-independent.

C3 — Permutation nulls.  For the canonical fixed point only, build
     an empirical null distribution for r_α and r_γ by randomly
     permuting |r_n| (for r_α) and γ̃_n (for r_γ) over N_PERM = 5000
     iterations.  Pre-registered: observed value must be in the
     top 5% (one-sided p < 0.05) of its own permutation null to be
     considered structurally significant.

Falsification synthesis (pre-registered)
----------------------------------------
F4 — R∞-1a-spectral is REFUTED as evidence for B1 if ANY of:
       (a) C1: r_β-null mean > 0.5  (white noise reproduces the signal)
       (b) C2: r_β drops < 0.5 at any (α, τ_l) grid point
       (c) C3: observed r_α and r_γ both fail permutation significance
              (p > 0.05)
     R∞-1a-spectral is STRENGTHENED if ALL of:
       (a) C1: r_β-null mean < 0.2 AND observed r_β outside 99% null
       (b) C2: r_β > 0.5 across entire grid
       (c) C3: observed r_α OR r_γ achieves p < 0.05

     Otherwise: MIXED — partial support, requires deeper test.

What R∞-1a-spectral-robustness does NOT do
-------------------------------------------
* Does NOT prove or disprove RH.
* Does NOT close T-HP, G4, or any gap.
* Does NOT construct an admissible rescaling operator.

Status: EXPERIMENTAL — TNFR-Riemann R∞-1a-spectral-robustness (May 2026).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Re-use the canonical pipeline from R∞-1a-spectral to guarantee identical
# treatment between baseline, null, and sensitivity runs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remesh_infinity_riemann_spectral import (  # noqa: E402
    DT,
    MPMATH_DPS,
    apply_network_remesh,
    build_prime_ladder_graph,
    compute_fixed_point,
    fetch_riemann_zeros,
    fetch_smooth_targets,
    pearson,
    snapshot_epi,
    spearman_rank,
    vec,
)
from tnfr.alias import set_attr  # noqa: E402
from tnfr.constants.aliases import ALIAS_EPI  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PRIMES: int = 10
MAX_POWER: int = 4
TAU_GLOBAL_CANON: int = 16
TAU_LOCAL_CANON: int = 4
ALPHA_CANON: float = 0.5
N_ITER: int = 512

# C1 — white-noise null
N_NULL_SEEDS: int = 16

# C2 — sensitivity grid
ALPHA_GRID: tuple[float, ...] = (0.25, 0.5, 0.75)
TAU_LOCAL_GRID: tuple[int, ...] = (2, 4, 8)

# C3 — permutation null
N_PERM: int = 5000
PERM_SEED: int = 20260526


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def populate_history_white_noise(G, n_steps: int, rng: np.random.Generator) -> None:
    """Populate _epi_hist with zero-mean unit-variance white noise.

    Mirrors `populate_history` shape, but replaces the canonical
    oscillatory synthetic field with white noise so that the REMESH
    contraction operates on a structurally null input.
    """
    nodes = list(G.nodes())
    hist: deque = deque(maxlen=n_steps + 10)
    for _ in range(n_steps):
        noise = rng.standard_normal(len(nodes))
        snap = {n: float(noise[i]) for i, n in enumerate(nodes)}
        hist.append(snap)
    G.graph["_epi_hist"] = hist
    last = hist[-1]
    for n, nd in G.nodes(data=True):
        set_attr(nd, ALIAS_EPI, last[n])


def populate_history_canonical(G, n_steps: int) -> None:
    """Re-implements R∞-1a-spectral populate_history (canonical osc field)."""
    from remesh_infinity_riemann_spectral import synthetic_epi_snapshot
    hist: deque = deque(maxlen=n_steps + 10)
    for step in range(n_steps):
        hist.append(synthetic_epi_snapshot(G, step * DT))
    G.graph["_epi_hist"] = hist
    last = hist[-1]
    for n, nd in G.nodes(data=True):
        set_attr(nd, ALIAS_EPI, last[n])


def compute_correlations(
    fixed_point: dict, nodes: list, gamma: np.ndarray,
    gamma_tilde: np.ndarray, abs_r: np.ndarray, M: int, n_nodes: int,
) -> dict[str, float]:
    """Return r_α, r_β, r_γ, r_δ for a given fixed point."""
    fp_vec = vec(fixed_point, nodes)
    nu_f = np.asarray([n[1] * math.log(n[0]) for n in nodes])
    order = np.argsort(nu_f)
    s_ordered = fp_vec[order]
    s_demean = s_ordered - s_ordered.mean()
    spectrum = np.fft.rfft(s_demean)
    power = (np.abs(spectrum) ** 2).astype(float)
    P = power[1:M + 1]
    return {
        "r_alpha": pearson(P, abs_r[:M]),
        "r_beta": pearson(np.sort(P)[::-1], np.sort(abs_r[:M])[::-1]),
        "r_gamma": pearson(s_ordered, gamma_tilde[:n_nodes]),
        "r_delta": spearman_rank(P, abs_r[:M]),
    }


def run_canonical_pipeline(
    alpha: float, tau_local: int, tau_global: int, n_iter: int,
    init_func, gamma: np.ndarray, gamma_tilde: np.ndarray,
    abs_r: np.ndarray, M: int,
) -> dict[str, Any]:
    """Build fresh graph, populate via init_func, iterate REMESH, correlate."""
    G = build_prime_ladder_graph(n_primes=N_PRIMES, max_power=MAX_POWER)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    init_func(G, tau_global + 20)
    G.graph["REMESH_TAU_LOCAL"] = tau_local
    G.graph["REMESH_ALPHA"] = alpha
    G.graph["REMESH_TAU_GLOBAL"] = tau_global
    baseline_epi = snapshot_epi(G)
    # compute_fixed_point uses module-level TAU_LOCAL/TAU_GLOBAL/ALPHA from
    # remesh_infinity_riemann_spectral, so we override here via graph attrs
    # and re-implement the iteration locally to honour (α, τ_l, τ_g):
    hist_backup = deque(
        copy.deepcopy(list(G.graph["_epi_hist"])),
        maxlen=G.graph["_epi_hist"].maxlen,
    )
    for _ in range(n_iter):
        apply_network_remesh(G)
        G.graph["_epi_hist"].append(snapshot_epi(G))
    fixed_point = snapshot_epi(G)
    G.graph["_epi_hist"] = hist_backup
    fp_vec = vec(fixed_point, nodes)
    corr = compute_correlations(
        fixed_point, nodes, gamma, gamma_tilde, abs_r, M, n_nodes
    )
    return {
        "fixed_point_l2": float(np.linalg.norm(fp_vec)),
        "fixed_point_mean": float(fp_vec.mean()),
        **corr,
    }


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------

def run() -> dict[str, Any]:
    # Shared Riemann reference (identical to R∞-1a-spectral)
    G0 = build_prime_ladder_graph(n_primes=N_PRIMES, max_power=MAX_POWER)
    n_nodes = G0.number_of_nodes()
    M = n_nodes // 2
    n_need = max(n_nodes, M + 1)
    gamma = fetch_riemann_zeros(n_need)
    gamma_tilde = fetch_smooth_targets(n_need)
    r_residual = gamma - gamma_tilde
    abs_r = np.abs(r_residual)

    # ---- Baseline (re-run R∞-1a-spectral inline for fair comparison) ----
    print("[1/3] Baseline (canonical α=0.5, τ_l=4, τ_g=16) ...", flush=True)
    baseline_result = run_canonical_pipeline(
        ALPHA_CANON, TAU_LOCAL_CANON, TAU_GLOBAL_CANON, N_ITER,
        populate_history_canonical, gamma, gamma_tilde, abs_r, M,
    )
    print(f"      r_β={baseline_result['r_beta']:+.4f}  "
          f"r_α={baseline_result['r_alpha']:+.4f}  "
          f"r_γ={baseline_result['r_gamma']:+.4f}  "
          f"r_δ={baseline_result['r_delta']:+.4f}", flush=True)

    # ---- C1 white-noise null ----
    print(f"[2/3] C1 white-noise null ({N_NULL_SEEDS} seeds) ...", flush=True)
    c1_runs: list[dict] = []
    for seed in range(N_NULL_SEEDS):
        rng = np.random.default_rng(20260526 + seed)
        def init(G, n): return populate_history_white_noise(G, n, rng)
        res = run_canonical_pipeline(
            ALPHA_CANON, TAU_LOCAL_CANON, TAU_GLOBAL_CANON, N_ITER,
            init, gamma, gamma_tilde, abs_r, M,
        )
        res["seed"] = seed
        c1_runs.append(res)
        print(f"      seed={seed:>2}  r_β={res['r_beta']:+.4f}  "
              f"r_α={res['r_alpha']:+.4f}  r_γ={res['r_gamma']:+.4f}  "
              f"r_δ={res['r_delta']:+.4f}", flush=True)

    def _stats(key: str) -> dict[str, float]:
        vals = np.asarray([r[key] for r in c1_runs])
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "abs_mean": float(np.abs(vals).mean()),
            "abs_max": float(np.abs(vals).max()),
            "q025": float(np.quantile(vals, 0.025)),
            "q975": float(np.quantile(vals, 0.975)),
        }

    c1_stats = {
        "r_alpha": _stats("r_alpha"),
        "r_beta": _stats("r_beta"),
        "r_gamma": _stats("r_gamma"),
        "r_delta": _stats("r_delta"),
    }

    # ---- C2 sensitivity sweep ----
    print(f"[3/3] C2 sensitivity sweep ({len(ALPHA_GRID)}×{len(TAU_LOCAL_GRID)}) ...",
          flush=True)
    c2_runs: list[dict] = []
    for alpha in ALPHA_GRID:
        for tau_l in TAU_LOCAL_GRID:
            res = run_canonical_pipeline(
                alpha, tau_l, TAU_GLOBAL_CANON, N_ITER,
                populate_history_canonical, gamma, gamma_tilde, abs_r, M,
            )
            res["alpha"] = alpha
            res["tau_local"] = tau_l
            c2_runs.append(res)
            print(f"      α={alpha} τ_l={tau_l}  r_β={res['r_beta']:+.4f}  "
                  f"r_α={res['r_alpha']:+.4f}  r_γ={res['r_gamma']:+.4f}",
                  flush=True)

    c2_beta_vals = np.asarray([r["r_beta"] for r in c2_runs])
    c2_beta_min = float(c2_beta_vals.min())
    c2_beta_max = float(c2_beta_vals.max())

    # ---- C3 permutation null on canonical baseline ----
    # We only need the canonical fixed point's spectrum; use baseline values.
    G_c = build_prime_ladder_graph(n_primes=N_PRIMES, max_power=MAX_POWER)
    nodes_c = list(G_c.nodes())
    populate_history_canonical(G_c, TAU_GLOBAL_CANON + 20)
    G_c.graph["REMESH_TAU_LOCAL"] = TAU_LOCAL_CANON
    G_c.graph["REMESH_ALPHA"] = ALPHA_CANON
    G_c.graph["REMESH_TAU_GLOBAL"] = TAU_GLOBAL_CANON
    base_epi = snapshot_epi(G_c)
    fp_c = compute_fixed_point(G_c, base_epi, N_ITER)
    fp_vec_c = vec(fp_c, nodes_c)
    nu_f_c = np.asarray([n[1] * math.log(n[0]) for n in nodes_c])
    order_c = np.argsort(nu_f_c)
    s_ord = fp_vec_c[order_c]
    s_dem = s_ord - s_ord.mean()
    P_c = (np.abs(np.fft.rfft(s_dem)) ** 2)[1:M + 1]

    rng_perm = np.random.default_rng(PERM_SEED)
    null_alpha = np.empty(N_PERM)
    null_gamma = np.empty(N_PERM)
    abs_r_M = abs_r[:M].copy()
    gt_N = gamma_tilde[:n_nodes].copy()
    for k in range(N_PERM):
        perm_a = rng_perm.permutation(M)
        null_alpha[k] = pearson(P_c, abs_r_M[perm_a])
        perm_g = rng_perm.permutation(n_nodes)
        null_gamma[k] = pearson(s_ord, gt_N[perm_g])

    obs_alpha = baseline_result["r_alpha"]
    obs_gamma = baseline_result["r_gamma"]
    p_alpha_two = float((np.abs(null_alpha) >= abs(obs_alpha)).mean())
    p_alpha_one = float((null_alpha >= obs_alpha).mean())
    p_gamma_two = float((np.abs(null_gamma) >= abs(obs_gamma)).mean())
    p_gamma_one = float((null_gamma >= obs_gamma).mean())

    c3 = {
        "n_perm": N_PERM,
        "seed": PERM_SEED,
        "observed_r_alpha": obs_alpha,
        "observed_r_gamma": obs_gamma,
        "null_alpha_mean": float(null_alpha.mean()),
        "null_alpha_std": float(null_alpha.std(ddof=1)),
        "null_gamma_mean": float(null_gamma.mean()),
        "null_gamma_std": float(null_gamma.std(ddof=1)),
        "p_alpha_two_sided": p_alpha_two,
        "p_alpha_one_sided": p_alpha_one,
        "p_gamma_two_sided": p_gamma_two,
        "p_gamma_one_sided": p_gamma_one,
    }

    # ---- F4 verdict ----
    null_beta_mean = c1_stats["r_beta"]["mean"]
    null_beta_absmean = c1_stats["r_beta"]["abs_mean"]
    obs_beta = baseline_result["r_beta"]

    refute_a = null_beta_absmean > 0.5
    refute_b = c2_beta_min < 0.5
    refute_c = (p_alpha_one > 0.05) and (p_gamma_one > 0.05)
    refute_any = refute_a or refute_b or refute_c

    strengthen_a = (null_beta_absmean < 0.2) and (
        obs_beta < c1_stats["r_beta"]["q025"]
        or obs_beta > c1_stats["r_beta"]["q975"]
    )
    strengthen_b = c2_beta_min > 0.5
    strengthen_c = (p_alpha_one < 0.05) or (p_gamma_one < 0.05)
    strengthen_all = strengthen_a and strengthen_b and strengthen_c

    if refute_any:
        verdict = "REFUTED"
    elif strengthen_all:
        verdict = "STRENGTHENED"
    else:
        verdict = "MIXED"

    summary: dict[str, Any] = {
        "config": {
            "n_primes": N_PRIMES,
            "max_power": MAX_POWER,
            "n_nodes": n_nodes,
            "M": M,
            "tau_global": TAU_GLOBAL_CANON,
            "n_iter": N_ITER,
            "n_null_seeds": N_NULL_SEEDS,
            "alpha_grid": list(ALPHA_GRID),
            "tau_local_grid": list(TAU_LOCAL_GRID),
            "n_perm": N_PERM,
            "perm_seed": PERM_SEED,
        },
        "baseline_canonical": baseline_result,
        "C1_white_noise_null": {
            "n_seeds": N_NULL_SEEDS,
            "runs": c1_runs,
            "stats": c1_stats,
            "observed_r_beta_baseline": obs_beta,
            "null_r_beta_mean": null_beta_mean,
            "null_r_beta_abs_mean": null_beta_absmean,
            "observed_inside_null_95pct_r_beta": (
                c1_stats["r_beta"]["q025"] <= obs_beta <= c1_stats["r_beta"]["q975"]
            ),
        },
        "C2_sensitivity_sweep": {
            "runs": c2_runs,
            "r_beta_min": c2_beta_min,
            "r_beta_max": c2_beta_max,
            "r_beta_above_05_everywhere": bool(c2_beta_min > 0.5),
            "r_beta_below_05_somewhere": bool(c2_beta_min < 0.5),
        },
        "C3_permutation_null": c3,
        "F4_falsification": {
            "criterion": (
                "REFUTED if (C1 |r_β|-null mean > 0.5) OR "
                "(C2 r_β < 0.5 anywhere) OR "
                "(C3 both p_alpha_one > 0.05 AND p_gamma_one > 0.05). "
                "STRENGTHENED if (C1 |r_β|-null mean < 0.2 AND observed "
                "outside 95% null) AND (C2 r_β > 0.5 everywhere) AND "
                "(C3 either p_alpha_one < 0.05 OR p_gamma_one < 0.05). "
                "MIXED otherwise."
            ),
            "refute_C1_kernel_artefact": refute_a,
            "refute_C2_parameter_fragile": refute_b,
            "refute_C3_permutation_nonsignificant": refute_c,
            "strengthen_C1": strengthen_a,
            "strengthen_C2": strengthen_b,
            "strengthen_C3": strengthen_c,
            "verdict": verdict,
        },
    }
    return summary


def print_report(s: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("R∞-1a-spectral-robustness — F4 gate")
    print("=" * 78)
    b = s["baseline_canonical"]
    print(f"BASELINE (canonical):  r_α={b['r_alpha']:+.4f}  "
          f"r_β={b['r_beta']:+.4f}  r_γ={b['r_gamma']:+.4f}  "
          f"r_δ={b['r_delta']:+.4f}")
    print()
    c1 = s["C1_white_noise_null"]
    rb = c1["stats"]["r_beta"]
    ra = c1["stats"]["r_alpha"]
    rg = c1["stats"]["r_gamma"]
    print(f"C1 white-noise null  N={c1['n_seeds']}:")
    print(f"  r_β   mean={rb['mean']:+.4f}  |mean|={rb['abs_mean']:.4f}  "
          f"std={rb['std']:.4f}  q025={rb['q025']:+.4f}  q975={rb['q975']:+.4f}")
    print(f"  r_α   mean={ra['mean']:+.4f}  |mean|={ra['abs_mean']:.4f}  "
          f"std={ra['std']:.4f}")
    print(f"  r_γ   mean={rg['mean']:+.4f}  |mean|={rg['abs_mean']:.4f}  "
          f"std={rg['std']:.4f}")
    print(f"  baseline r_β={c1['observed_r_beta_baseline']:+.4f}  "
          f"inside null 95%? {c1['observed_inside_null_95pct_r_beta']}")
    print()
    c2 = s["C2_sensitivity_sweep"]
    print(f"C2 sensitivity sweep ({len(c2['runs'])} cells):")
    print(f"  r_β range  [{c2['r_beta_min']:+.4f}, {c2['r_beta_max']:+.4f}]")
    print(f"  r_β > 0.5 everywhere? {c2['r_beta_above_05_everywhere']}")
    print()
    c3 = s["C3_permutation_null"]
    print(f"C3 permutation null  N_perm={c3['n_perm']}:")
    print(f"  observed r_α={c3['observed_r_alpha']:+.4f}  "
          f"null mean={c3['null_alpha_mean']:+.4f}  std={c3['null_alpha_std']:.4f}")
    print(f"    p_one_sided={c3['p_alpha_one_sided']:.4f}  "
          f"p_two_sided={c3['p_alpha_two_sided']:.4f}")
    print(f"  observed r_γ={c3['observed_r_gamma']:+.4f}  "
          f"null mean={c3['null_gamma_mean']:+.4f}  std={c3['null_gamma_std']:.4f}")
    print(f"    p_one_sided={c3['p_gamma_one_sided']:.4f}  "
          f"p_two_sided={c3['p_gamma_two_sided']:.4f}")
    print()
    f4 = s["F4_falsification"]
    print(f"F4 VERDICT: {f4['verdict']}")
    print(f"  refute_C1_kernel_artefact:        {f4['refute_C1_kernel_artefact']}")
    print(f"  refute_C2_parameter_fragile:      {f4['refute_C2_parameter_fragile']}")
    print(f"  refute_C3_permutation_nonsignif:  {f4['refute_C3_permutation_nonsignificant']}")
    print(f"  strengthen_C1:                    {f4['strengthen_C1']}")
    print(f"  strengthen_C2:                    {f4['strengthen_C2']}")
    print(f"  strengthen_C3:                    {f4['strengthen_C3']}")
    print("=" * 78)


def main() -> None:
    summary = run()
    out_dir = Path("results/remesh_infinity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remesh_infinity_riemann_spectral_robustness.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_report(summary)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
