"""R∞-1a-spectral — Spectral projection of REMESH-∞ fixed point onto Riemann basis.

Scope (honest)
--------------
Follow-up to ``benchmarks/remesh_infinity_riemann_baseline.py`` (R∞-1a).
That benchmark established (Track B, N=512) that iterated REMESH on
prime-ladder synthetic dynamics admits a non-trivial fixed point
EPI* with structured FFT content (top-3 bins {16,19,20} of 21,
fractions ~{10.6%, 9.7%, 9.3%}; dc_fraction post-demean ~ 3e-33).

Established (R∞-1a):
    - iterated REMESH is contractive (step_decay = 6.82e-6)
    - fixed point ≠ time-average (rel→avg = 0.2808)
    - spectral content concentrated in high-νf bins (necessary
      condition for B1 reframe satisfied)

NOT established (open after R∞-1a):
    - whether the spectral content of EPI* has ANY measurable
      correspondence with Riemann data (γ_n or residuals r_n)

Hypothesis under test (B1 spectral)
-----------------------------------
H2: There exists a non-trivial linear correlation between the
    magnitudes of the fixed-point spectrum (ordered by some
    canonical TNFR axis) and a Riemann-side quantity built from
    the first N zeros, where N is the spectral dimension.

Falsification criterion (pre-registered)
----------------------------------------
F3: If max(|r_α|, |r_β|, |r_γ|, |r_δ|) < 0.2 across all four
    correlation tests defined below, branch B1 is empirically
    refuted **at the spectral level** even though the fixed point
    is non-trivial.  This would be strong evidence for branch B2
    or B3 (§13octies of TNFR_RIEMANN_RESEARCH_NOTES.md).

    If max(...) ∈ [0.2, 0.5]: indeterminate — fixed point has
    weak/noisy Riemann signature; needs operator-level redesign.

    If max(...) > 0.5: B1 spectrally supported (still does NOT
    prove RH; only that REMESH-∞ on prime-ladder dynamics is
    measurably correlated with Riemann data).

Tests (pre-registered, none decisive on its own)
------------------------------------------------
Let s_i = EPI*[i] for i = 1..N nodes, sorted by νf = k·log(p)
(canonical TNFR energy axis).  Let P_k = |FFT(s - mean(s))|² for
k = 0..N/2 = M.  Let γ_n be the first N Riemann zeros (mpmath),
γ̃_n the P28 smooth approximations, and r_n = γ_n - γ̃_n the
oscillatory residuals.

    r_α  = Pearson(P_1..P_M, |r_1..r_M|)            [index-aligned]
    r_β  = Pearson(sort(P_1..P_M, desc),
                   sort(|r_1..r_M|, desc))           [magnitude dist.]
    r_γ  = Pearson(s_1..s_N, γ̃_1..γ̃_N)             [node-ordered vs smooth γ]
    r_δ  = Spearman-rank(P_1..P_M, |r_1..r_M|)       [monotone alignment]

What R∞-1a-spectral does NOT do
-------------------------------
* Does NOT prove or disprove RH.
* Does NOT close T-HP, G4, or any gap.
* Does NOT build the admissible rescaling operator F.
* Does NOT modify the canonical engine.

Status: EXPERIMENTAL — TNFR-Riemann R∞-1a-spectral (May 2026).
"""

from __future__ import annotations

import copy
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.operators.remesh import apply_network_remesh
from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph
from tnfr.riemann.structural_zero_density import derive_smooth_zero_position

# ---------------------------------------------------------------------------
# Configuration (kept identical to R∞-1a for cross-comparison)
# ---------------------------------------------------------------------------

N_PRIMES: int = 10
MAX_POWER: int = 4
DT: float = 0.05
TAU_LOCAL: int = 4
ALPHA: float = 0.5
TAU_GLOBAL: int = 16
N_ITER: int = 512
MPMATH_DPS: int = 30


# ---------------------------------------------------------------------------
# Helpers (mirror R∞-1a)
# ---------------------------------------------------------------------------


def synthetic_epi_snapshot(G, t: float) -> dict:
    out: dict = {}
    for node in G.nodes():
        p, k = node
        log_p = math.log(p)
        nu_f = k * log_p
        out[node] = (log_p / k) * math.cos(nu_f * t)
    return out


def populate_history(G, n_steps: int) -> None:
    hist: deque = deque(maxlen=n_steps + 10)
    for step in range(n_steps):
        hist.append(synthetic_epi_snapshot(G, step * DT))
    G.graph["_epi_hist"] = hist
    last = hist[-1]
    for n, nd in G.nodes(data=True):
        set_attr(nd, ALIAS_EPI, last[n])


def snapshot_epi(G) -> dict:
    return {n: float(get_attr(nd, ALIAS_EPI, 0.0)) for n, nd in G.nodes(data=True)}


def restore_epi(G, snap: dict) -> None:
    for n, nd in G.nodes(data=True):
        set_attr(nd, ALIAS_EPI, snap[n])


def vec(d: dict, nodes: list) -> np.ndarray:
    return np.asarray([d[n] for n in nodes], dtype=float)


# ---------------------------------------------------------------------------
# Correlation utilities (pure NumPy / scipy-free)
# ---------------------------------------------------------------------------


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom < 1e-30:
        return float("nan")
    return float((xm * ym).sum() / denom)


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on ranks (no ties handling)."""

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks

    return pearson(_rank(x), _rank(y))


# ---------------------------------------------------------------------------
# Iterate REMESH^N to obtain the fixed point (replicates R∞-1a Track B)
# ---------------------------------------------------------------------------


def compute_fixed_point(G, baseline_epi: dict, n_iter: int) -> dict:
    G.graph["REMESH_TAU_LOCAL"] = TAU_LOCAL
    G.graph["REMESH_TAU_GLOBAL"] = TAU_GLOBAL
    G.graph["REMESH_ALPHA"] = ALPHA

    hist_backup = deque(
        copy.deepcopy(list(G.graph["_epi_hist"])),
        maxlen=G.graph["_epi_hist"].maxlen,
    )
    restore_epi(G, baseline_epi)
    for _ in range(n_iter):
        apply_network_remesh(G)
        G.graph["_epi_hist"].append(snapshot_epi(G))
    fixed_point = snapshot_epi(G)
    G.graph["_epi_hist"] = hist_backup
    restore_epi(G, baseline_epi)
    return fixed_point


# ---------------------------------------------------------------------------
# Riemann data
# ---------------------------------------------------------------------------


def fetch_riemann_zeros(n: int, dps: int = MPMATH_DPS) -> np.ndarray:
    mp.mp.dps = dps
    return np.asarray(
        [float(mp.im(mp.zetazero(k))) for k in range(1, n + 1)], dtype=float
    )


def fetch_smooth_targets(n: int) -> np.ndarray:
    """First n smooth Riemann-Siegel zero approximations via P28 API."""
    return np.asarray(
        [derive_smooth_zero_position(k) for k in range(1, n + 1)], dtype=float
    )


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    G = build_prime_ladder_graph(n_primes=N_PRIMES, max_power=MAX_POWER)
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    populate_history(G, n_steps=TAU_GLOBAL + 20)
    G.graph["REMESH_TAU_LOCAL"] = TAU_LOCAL
    G.graph["REMESH_ALPHA"] = ALPHA
    baseline_epi = snapshot_epi(G)

    # ---- Fixed point via iterated REMESH ----
    fixed_point = compute_fixed_point(G, baseline_epi, N_ITER)
    fp_vec_node_order = vec(fixed_point, nodes)

    # ---- νf-ordered fixed point ----
    nu_f = np.asarray([n[1] * math.log(n[0]) for n in nodes])
    order = np.argsort(nu_f)
    s_ordered = fp_vec_node_order[order]
    nu_f_ordered = nu_f[order]
    s_demean = s_ordered - s_ordered.mean()

    # ---- FFT power per bin ----
    spectrum = np.fft.rfft(s_demean)
    power = (np.abs(spectrum) ** 2).astype(float)
    # Drop DC bin (already removed by demean); keep bins 1..M
    P_full = power.copy()
    P = power[1:]
    M = len(P)  # = n_nodes // 2 = 20 for n_nodes=40

    # ---- Riemann zeros and smooth targets ----
    # Need enough zeros for all tests. M for spectral tests, n_nodes for
    # node-ordered test.
    n_need = max(n_nodes, M + 1)
    gamma = fetch_riemann_zeros(n_need)
    gamma_tilde = fetch_smooth_targets(n_need)
    r_residual = gamma - gamma_tilde
    abs_r = np.abs(r_residual)

    # ---- Pre-registered tests ----
    r_alpha = pearson(P, abs_r[:M])
    r_beta = pearson(np.sort(P)[::-1], np.sort(abs_r[:M])[::-1])
    r_gamma = pearson(s_ordered, gamma_tilde[:n_nodes])
    r_delta = spearman_rank(P, abs_r[:M])

    # Auxiliary controls (NOT in falsification criterion, only diagnostic)
    r_aux_smooth_pow = pearson(P, gamma_tilde[:M])
    r_aux_zero_pow = pearson(P, gamma[:M])
    r_aux_node_residual = pearson(s_ordered, r_residual[:n_nodes])

    max_pre_registered = max(abs(r_alpha), abs(r_beta), abs(r_gamma), abs(r_delta))
    F3_refuted = max_pre_registered < 0.2
    F3_supported = max_pre_registered > 0.5

    summary: dict[str, Any] = {
        "config": {
            "n_primes": N_PRIMES,
            "max_power": MAX_POWER,
            "n_nodes": n_nodes,
            "tau_global": TAU_GLOBAL,
            "tau_local": TAU_LOCAL,
            "alpha": ALPHA,
            "n_iter": N_ITER,
            "spectral_M": M,
            "mpmath_dps": MPMATH_DPS,
        },
        "riemann": {
            "gamma_first10": gamma[:10].tolist(),
            "gamma_tilde_first10": gamma_tilde[:10].tolist(),
            "residual_first10": r_residual[:10].tolist(),
            "residual_abs_mean": float(abs_r[:M].mean()),
            "residual_abs_max": float(abs_r[:M].max()),
        },
        "fixed_point": {
            "node_order_l2": float(np.linalg.norm(fp_vec_node_order)),
            "node_order_mean": float(fp_vec_node_order.mean()),
            "spectral_total_power": float(P_full.sum()),
            "spectral_top3_bins": np.argsort(P)[-3:][::-1].tolist(),
            "spectral_top3_power_fraction": [
                float(P[i] / (P.sum() + 1e-30))
                for i in np.argsort(P)[-3:][::-1].tolist()
            ],
        },
        "pre_registered_tests": {
            "r_alpha_pearson_power_vs_residual_index_aligned": r_alpha,
            "r_beta_pearson_sorted_power_vs_sorted_residual": r_beta,
            "r_gamma_pearson_nodefield_vs_smooth_targets": r_gamma,
            "r_delta_spearman_power_vs_residual": r_delta,
            "max_abs_pre_registered": max_pre_registered,
        },
        "auxiliary_controls_NOT_in_F3": {
            "r_power_vs_gamma_tilde": r_aux_smooth_pow,
            "r_power_vs_gamma": r_aux_zero_pow,
            "r_nodefield_vs_residual": r_aux_node_residual,
        },
        "falsification_F3": {
            "criterion": (
                "max(|r_alpha|,|r_beta|,|r_gamma|,|r_delta|) < 0.2 "
                "REFUTES B1 at spectral level; > 0.5 SUPPORTS B1 "
                "spectrally; in between indeterminate."
            ),
            "max_abs_pre_registered": max_pre_registered,
            "F3_refuted": F3_refuted,
            "F3_supported": F3_supported,
            "verdict": (
                "REFUTED"
                if F3_refuted
                else "SUPPORTED" if F3_supported else "INDETERMINATE"
            ),
        },
    }
    return summary


def print_report(summary: dict[str, Any]) -> None:
    print("=" * 78)
    print("R∞-1a-spectral — Spectral projection onto Riemann basis")
    print("=" * 78)
    cfg = summary["config"]
    print(
        f"Prime-ladder n_primes={cfg['n_primes']}, max_power={cfg['max_power']}, "
        f"n_nodes={cfg['n_nodes']}, M={cfg['spectral_M']}"
    )
    print(
        f"α={cfg['alpha']}, τ_l={cfg['tau_local']}, τ_g={cfg['tau_global']}, "
        f"N_iter={cfg['n_iter']}"
    )
    print()
    print("--- Riemann reference (mpmath, first 10) ---")
    rie = summary["riemann"]
    for i, (g, gt, r) in enumerate(
        zip(rie["gamma_first10"], rie["gamma_tilde_first10"], rie["residual_first10"])
    ):
        print(f"  n={i+1:>2}  γ={g:>10.6f}  γ̃={gt:>10.6f}  r={r:>+10.6f}")
    print(
        f"  Σ-stats:  |r|_mean={rie['residual_abs_mean']:.4f}  "
        f"|r|_max={rie['residual_abs_max']:.4f}"
    )
    print()
    print("--- Fixed point ---")
    fp = summary["fixed_point"]
    print(f"  ‖EPI*‖_L2 (node order) = {fp['node_order_l2']:.6f}")
    print(f"  mean(EPI*) = {fp['node_order_mean']:+.6e}")
    print(f"  spectral total power = {fp['spectral_total_power']:.4f}")
    print(
        f"  top-3 bins (1..M) = {fp['spectral_top3_bins']}  "
        f"fractions = {[f'{x:.3f}' for x in fp['spectral_top3_power_fraction']]}"
    )
    print()
    print("--- Pre-registered correlation tests ---")
    pr = summary["pre_registered_tests"]
    print(
        f"  r_α  (power[1..M] vs |r_n|[1..M], index-aligned)  = {pr['r_alpha_pearson_power_vs_residual_index_aligned']:+.4f}"
    )
    print(
        f"  r_β  (sort(power, desc) vs sort(|r_n|, desc))      = {pr['r_beta_pearson_sorted_power_vs_sorted_residual']:+.4f}"
    )
    print(
        f"  r_γ  (EPI*[νf-ordered] vs γ̃_n[1..N])              = {pr['r_gamma_pearson_nodefield_vs_smooth_targets']:+.4f}"
    )
    print(
        f"  r_δ  (Spearman power vs |r_n|)                     = {pr['r_delta_spearman_power_vs_residual']:+.4f}"
    )
    print(f"  max |·| = {pr['max_abs_pre_registered']:.4f}")
    print()
    print("--- Auxiliary controls (NOT in F3) ---")
    aux = summary["auxiliary_controls_NOT_in_F3"]
    for k, v in aux.items():
        print(f"  {k:>40s} = {v:+.4f}")
    print()
    print("--- F3 falsification verdict ---")
    f3 = summary["falsification_F3"]
    print(f"  Criterion: {f3['criterion']}")
    print(f"  max|·| = {f3['max_abs_pre_registered']:.4f}")
    print(f"  VERDICT: {f3['verdict']}")
    print("=" * 78)


def main() -> None:
    summary = run()
    print_report(summary)
    out_dir = Path("results/remesh_infinity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "remesh_infinity_riemann_spectral.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
