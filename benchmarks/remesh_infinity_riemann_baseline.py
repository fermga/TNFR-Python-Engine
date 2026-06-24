"""R∞-1a — REMESH global asymptotic limit baseline (Riemann side).

Scope (honest)
--------------
This benchmark tests a **falsifiable** numerical question about the
naive form of branch B1 of the REMESH global reframe (see
``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13vicies-novies and
``/memories/repo/tnfr-riemann-program-status.md``):

    Q: Does the canonical operator ``apply_network_remesh``, applied
       with τ_g → ∞ to a stationary oscillatory EPI field whose
       component frequencies are the prime-ladder structural
       frequencies ``νf = k·log(p)``, produce a perturbation whose
       deviation from the time average carries structured
       (non-time-averaged) information that could, in principle,
       encode the oscillatory residual

           r_n = γ_n - γ̃_n = π · S(γ̃_n)

       between true Riemann zeros γ_n and the P28/P30 smooth targets
       γ̃_n?

Hypothesis under test (naive B1)
--------------------------------
H1: ``REMESH^τ_g[EPI]`` retains structured non-trivial τ_g-dependence
    distinct from the time-average projection.  This would be
    **necessary** (not sufficient) for REMESH-∞ to have any chance
    of encoding S(T)-like oscillatory information.

Falsification criterion
-----------------------
F1: If ``‖REMESH^τ_g[EPI] - time_average(EPI)‖ → 0`` monotonically
    as τ_g increases, then REMESH global on stationary dynamics is
    just a Cesàro-style projection onto the time mean.  Naive B1 is
    refuted: such an operator cannot encode oscillatory residuals
    beyond what is already present in the time average.  Pivot
    required.

Prior expectation (honest)
--------------------------
The canonical mixing law is

    EPI_new = (1-α)·EPI_now + α·EPI[t-τ_l]
    EPI_new = (1-α)·EPI_new + α·EPI[t-τ_g]

with α = 0.5.  This is linear in the snapshots, so for a stationary
process the expectation of REMESH^τ_g approaches a fixed convex
combination of past expectations, which equals the time-average.
The agent therefore expects **F1 to trigger** in this baseline test.

A positive (non-falsified) result would force re-examination of the
mixing law's interaction with deterministic oscillatory snapshots
versus stationary stochastic ones, and would justify R∞-1b (NS side)
plus R∞-2 (analytical derivation of the τ_g → ∞ limit operator).

What R∞-1a does NOT do
----------------------
* Does NOT prove or disprove the Riemann Hypothesis.
* Does NOT prove or disprove branch B1 in its general form (only the
  naive stationary-dynamics version tested here).
* Does NOT touch the canonical engine; it only consumes
  ``apply_network_remesh`` and ``build_prime_ladder_graph``.
* Does NOT modify any operator semantics.

Status: EXPERIMENTAL — TNFR-Riemann R∞-1a (May 2026).
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.operators.remesh import apply_network_remesh
from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PRIMES: int = 10
MAX_POWER: int = 4
DT: float = 0.05  # Synthetic temporal step
TAU_LOCAL: int = 4
ALPHA: float = 0.5
TAU_GLOBAL_SWEEP: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def synthetic_epi_snapshot(G, t: float) -> dict:
    """Stationary oscillatory EPI field at time ``t``.

    For each node (p, k) the canonical structural frequency is
    νf = k · log(p).  The synthetic field is

        EPI(p, k; t) = (log(p) / k) · cos(νf · t)

    The amplitude log(p)/k reproduces the canonical von Mangoldt
    weighting Λ(p^k) = log(p) attenuated by the echo index k.  The
    cosine carries the canonical phase.  Mean over t equals 0.
    """
    out: dict = {}
    for node in G.nodes():
        p, k = node
        log_p = math.log(p)
        nu_f = k * log_p
        out[node] = (log_p / k) * math.cos(nu_f * t)
    return out


def populate_history(G, n_steps: int) -> None:
    """Pre-load ``_epi_hist`` with deterministic oscillatory snapshots.

    The most recent snapshot is also written into the live EPI
    attribute so that ``apply_network_remesh`` sees a consistent
    "present" state.
    """
    hist: deque = deque(maxlen=n_steps + 10)
    for step in range(n_steps):
        t = step * DT
        hist.append(synthetic_epi_snapshot(G, t))
    G.graph["_epi_hist"] = hist
    last = hist[-1]
    for n, nd in G.nodes(data=True):
        set_attr(nd, ALIAS_EPI, last[n])


def snapshot_epi(G) -> dict:
    return {n: float(get_attr(nd, ALIAS_EPI, 0.0)) for n, nd in G.nodes(data=True)}


def restore_epi(G, snap: dict) -> None:
    for n, nd in G.nodes(data=True):
        set_attr(nd, ALIAS_EPI, snap[n])


def time_average_field(G) -> dict:
    hist = G.graph["_epi_hist"]
    nodes = list(G.nodes())
    avg: dict = {n: 0.0 for n in nodes}
    for snap in hist:
        for n in nodes:
            avg[n] += snap[n]
    inv = 1.0 / len(hist)
    return {n: v * inv for n, v in avg.items()}


def vec(d: dict, nodes: list) -> np.ndarray:
    return np.asarray([d[n] for n in nodes], dtype=float)


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------


def run_track_A_single_application(
    G,
    nodes,
    baseline_epi: dict,
    avg_vec: np.ndarray,
    baseline_to_avg: float,
) -> list[dict[str, Any]]:
    """Track A — single application of REMESH at sweeping τ_g.

    Tests the *naive* notion of REMESH-∞ as one-shot operator with
    τ_g → ∞.  Expected (and confirmed empirically in the first run):
    no well-defined limit — output depends on the phase of the specific
    past snapshot sampled at lag τ_g.
    """
    G.graph["REMESH_TAU_LOCAL"] = TAU_LOCAL
    G.graph["REMESH_ALPHA"] = ALPHA
    baseline_vec = vec(baseline_epi, nodes)

    results: list[dict[str, Any]] = []
    for tau_g in TAU_GLOBAL_SWEEP:
        G.graph["REMESH_TAU_GLOBAL"] = tau_g
        restore_epi(G, baseline_epi)
        apply_network_remesh(G)
        post = snapshot_epi(G)
        post_vec = vec(post, nodes)
        dist_to_avg = float(np.linalg.norm(post_vec - avg_vec))
        relative_to_avg = dist_to_avg / (baseline_to_avg + 1e-12)
        delta = post_vec - baseline_vec
        results.append(
            {
                "tau_g": tau_g,
                "dist_to_time_average": dist_to_avg,
                "relative_to_baseline_avg_distance": relative_to_avg,
                "delta_l2": float(np.linalg.norm(delta)),
                "delta_max": float(np.max(np.abs(delta))),
                "delta_var": float(np.var(delta)),
            }
        )
    return results


def run_track_B_iterated(
    G,
    nodes,
    baseline_epi: dict,
    avg_vec: np.ndarray,
    baseline_to_avg: float,
    tau_g_fixed: int = 16,
    n_iter_max: int = 64,
) -> list[dict[str, Any]]:
    """Track B — iterated REMESH at fixed τ_g.

    Tests the *canonical* operationalization of REMESH-∞ as
    REMESH^N in the N → ∞ limit, with τ_g held fixed.  After each
    application, the new EPI snapshot is appended to ``_epi_hist``
    so that the next iteration sees an updated past.  This is the
    natural Banach iteration of the REMESH map; if the map is a
    contraction the iterates converge to a fixed point (attractor).

    Restores the history to its initial state at the end (the
    benchmark must not have side effects on the caller's view of
    ``G``).
    """
    import copy

    G.graph["REMESH_TAU_LOCAL"] = TAU_LOCAL
    G.graph["REMESH_TAU_GLOBAL"] = tau_g_fixed
    G.graph["REMESH_ALPHA"] = ALPHA

    hist_backup = deque(
        copy.deepcopy(list(G.graph["_epi_hist"])), maxlen=G.graph["_epi_hist"].maxlen
    )
    restore_epi(G, baseline_epi)

    iter_log: list[dict[str, Any]] = []
    prev_vec = vec(baseline_epi, nodes)
    for n_iter in range(1, n_iter_max + 1):
        apply_network_remesh(G)
        post = snapshot_epi(G)
        # Feed the new state into the history so the next iteration
        # is genuinely iterated (otherwise REMESH always sees the
        # same fixed past snapshot).
        G.graph["_epi_hist"].append(post)

        post_vec = vec(post, nodes)
        step_delta = float(np.linalg.norm(post_vec - prev_vec))
        dist_to_avg = float(np.linalg.norm(post_vec - avg_vec))
        relative_to_avg = dist_to_avg / (baseline_to_avg + 1e-12)
        iter_log.append(
            {
                "n_iter": n_iter,
                "step_delta_l2": step_delta,
                "dist_to_time_average": dist_to_avg,
                "relative_to_baseline_avg_distance": relative_to_avg,
                "norm": float(np.linalg.norm(post_vec)),
            }
        )
        prev_vec = post_vec

    # Restore history & EPI so subsequent tracks see clean state.
    G.graph["_epi_hist"] = hist_backup
    restore_epi(G, baseline_epi)
    return iter_log


def run() -> dict[str, Any]:
    G = build_prime_ladder_graph(n_primes=N_PRIMES, max_power=MAX_POWER)
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    max_tau = max(TAU_GLOBAL_SWEEP)
    populate_history(G, n_steps=max_tau + 20)

    G.graph["REMESH_TAU_LOCAL"] = TAU_LOCAL
    G.graph["REMESH_ALPHA"] = ALPHA

    baseline_epi = snapshot_epi(G)
    time_avg = time_average_field(G)

    baseline_vec = vec(baseline_epi, nodes)
    avg_vec = vec(time_avg, nodes)
    baseline_to_avg = float(np.linalg.norm(baseline_vec - avg_vec))

    # -----------------------------------------------------------------
    # Track A: single application, sweeping τ_g
    # -----------------------------------------------------------------
    track_A = run_track_A_single_application(
        G, nodes, baseline_epi, avg_vec, baseline_to_avg
    )

    # -----------------------------------------------------------------
    # Track B: iterated REMESH at fixed τ_g (canonical Banach iteration)
    # -----------------------------------------------------------------
    track_B = run_track_B_iterated(
        G,
        nodes,
        baseline_epi,
        avg_vec,
        baseline_to_avg,
        tau_g_fixed=16,
        n_iter_max=512,
    )

    # -----------------------------------------------------------------
    # Track C: spectral diagnostic of the Track B late-time state
    # -----------------------------------------------------------------
    # Re-run a short iteration to recover the late state and analyse
    # its spectral content along the νf-ordered axis.  If the iterated
    # operator carries any prime-ladder oscillatory structure, the
    # power spectrum of the late state (ordered by νf = k·log(p))
    # should show non-trivial peaks beyond the DC component.
    import copy

    hist_backup = deque(
        copy.deepcopy(list(G.graph["_epi_hist"])), maxlen=G.graph["_epi_hist"].maxlen
    )
    G.graph["REMESH_TAU_GLOBAL"] = 16
    restore_epi(G, baseline_epi)
    for _ in range(256):
        apply_network_remesh(G)
        G.graph["_epi_hist"].append(snapshot_epi(G))
    late_state = snapshot_epi(G)
    G.graph["_epi_hist"] = hist_backup
    restore_epi(G, baseline_epi)

    late_vec = vec(late_state, nodes)
    nu_f = np.asarray([n[1] * math.log(n[0]) for n in nodes])
    order = np.argsort(nu_f)
    late_ordered = late_vec[order]
    # Subtract mean to isolate oscillatory content
    late_demean = late_ordered - np.mean(late_ordered)
    spectrum = np.fft.rfft(late_demean)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    dc_fraction = float(power[0] / (total_power + 1e-12)) if total_power > 0 else 0.0
    top3_idx = np.argsort(power)[-3:][::-1].tolist()
    top3_power = [float(power[i]) for i in top3_idx]
    top3_fraction = [float(p / (total_power + 1e-12)) for p in top3_power]

    track_C_spectral = {
        "iterations": 256,
        "tau_g_fixed": 16,
        "ordered_axis": "nu_f = k * log(p)",
        "total_oscillatory_power": total_power,
        "dc_fraction_post_demean": dc_fraction,
        "top3_freq_bins": top3_idx,
        "top3_power": top3_power,
        "top3_power_fraction": top3_fraction,
        "late_state_l2_norm": float(np.linalg.norm(late_vec)),
        "late_state_mean": float(np.mean(late_vec)),
    }

    results = track_A  # Backward-compatible alias

    # ---------------------------------------------------------------
    # Falsification check (F1) — Track A
    # ---------------------------------------------------------------
    rel_seq = [r["relative_to_baseline_avg_distance"] for r in results]
    monotone_decreasing = all(
        rel_seq[i] >= rel_seq[i + 1] for i in range(len(rel_seq) - 1)
    )
    final_rel = rel_seq[-1]
    F1_triggered = monotone_decreasing and final_rel < 0.1

    # ---------------------------------------------------------------
    # Convergence check (F2) — Track B
    # ---------------------------------------------------------------
    # F2: REMESH^N converges to a fixed point (step delta → 0).
    # If F2 triggers, the iterated operator has a well-defined limit
    # and the question becomes: does the fixed point carry any
    # oscillatory residual information?  If F2 fails, iterated REMESH
    # diverges or orbits and a different operationalization is needed.
    final_step_delta = track_B[-1]["step_delta_l2"]
    initial_step_delta = track_B[0]["step_delta_l2"]
    step_decay_ratio = (
        final_step_delta / initial_step_delta
        if initial_step_delta > 1e-12
        else float("nan")
    )
    F2_triggered = final_step_delta < 1e-6 or step_decay_ratio < 0.01
    track_B_final_rel = track_B[-1]["relative_to_baseline_avg_distance"]

    summary = {
        "config": {
            "n_primes": N_PRIMES,
            "max_power": MAX_POWER,
            "n_nodes": n_nodes,
            "dt": DT,
            "alpha": ALPHA,
            "tau_local": TAU_LOCAL,
            "tau_global_sweep": list(TAU_GLOBAL_SWEEP),
            "track_B_tau_global_fixed": 16,
            "track_B_n_iter": 512,
        },
        "baseline_to_time_average_distance": baseline_to_avg,
        "track_A_single_application": track_A,
        "track_B_iterated": track_B,
        "track_C_spectral": track_C_spectral,
        "falsification": {
            "F1_criterion": (
                "REMESH single-application sweep → time_average "
                "(monotone decreasing distance, final < 10% of baseline gap)"
            ),
            "F1_monotone_decreasing": monotone_decreasing,
            "F1_final_relative_distance": final_rel,
            "F1_triggered": F1_triggered,
            "F1_interpretation": (
                "Naive single-application B1 REFUTED (Cesàro projection)"
                if F1_triggered
                else "Naive single-application B1 NOT refuted by F1"
            ),
            "F2_criterion": (
                "REMESH^N iterated converges to fixed point "
                "(step delta → 0 or decay ratio < 1%)"
            ),
            "F2_step_decay_ratio": step_decay_ratio,
            "F2_final_step_delta": final_step_delta,
            "F2_triggered": F2_triggered,
            "F2_track_B_final_rel_to_avg": track_B_final_rel,
            "F2_interpretation": (
                "Iterated REMESH has well-defined fixed point — " "examine its content"
                if F2_triggered
                else "Iterated REMESH does NOT converge to fixed point "
                "within tested horizon"
            ),
        },
    }
    return summary


def print_table(summary: dict[str, Any]) -> None:
    print("=" * 78)
    print("R∞-1a — REMESH global asymptotic baseline (Riemann side)")
    print("=" * 78)
    cfg = summary["config"]
    print(
        f"Prime-ladder: n_primes={cfg['n_primes']}, max_power={cfg['max_power']}, "
        f"n_nodes={cfg['n_nodes']}"
    )
    print(f"α = {cfg['alpha']}, τ_l = {cfg['tau_local']}, dt = {cfg['dt']}")
    print(
        f"Baseline-to-time-average distance: "
        f"{summary['baseline_to_time_average_distance']:.6e}"
    )
    print()
    print("--- Track A: single application, τ_g sweep ---")
    print(
        f"{'τ_g':>6} {'dist→avg':>14} {'rel→avg':>10} "
        f"{'δ_L2':>12} {'δ_max':>12} {'δ_var':>12}"
    )
    print("-" * 78)
    for row in summary["track_A_single_application"]:
        print(
            f"{row['tau_g']:>6} "
            f"{row['dist_to_time_average']:>14.6e} "
            f"{row['relative_to_baseline_avg_distance']:>10.4f} "
            f"{row['delta_l2']:>12.6e} "
            f"{row['delta_max']:>12.6e} "
            f"{row['delta_var']:>12.6e}"
        )
    print()
    print(f"--- Track B: iterated REMESH at τ_g={cfg['track_B_tau_global_fixed']} ---")
    print(
        f"{'N':>4} {'step_delta':>14} {'dist→avg':>14} {'rel→avg':>10} "
        f"{'‖EPI‖':>12}"
    )
    print("-" * 78)
    # Print first 5, every 8th, and last 3 to keep output readable
    rows = summary["track_B_iterated"]
    indices = sorted(
        set(
            list(range(5))
            + list(range(7, len(rows), 8))
            + list(range(len(rows) - 3, len(rows)))
        )
    )
    for i in indices:
        if 0 <= i < len(rows):
            row = rows[i]
            print(
                f"{row['n_iter']:>4} "
                f"{row['step_delta_l2']:>14.6e} "
                f"{row['dist_to_time_average']:>14.6e} "
                f"{row['relative_to_baseline_avg_distance']:>10.4f} "
                f"{row['norm']:>12.6e}"
            )
    print()
    fals = summary["falsification"]
    print("--- Falsification summary ---")
    print(
        f"F1 (single-application Cesàro): triggered={fals['F1_triggered']} | "
        f"monotone={fals['F1_monotone_decreasing']} | "
        f"final_rel={fals['F1_final_relative_distance']:.4f}"
    )
    print(f"  → {fals['F1_interpretation']}")
    print(
        f"F2 (iterated fixed point):      triggered={fals['F2_triggered']} | "
        f"step_decay={fals['F2_step_decay_ratio']:.4e} | "
        f"final_step_delta={fals['F2_final_step_delta']:.4e}"
    )
    print(f"  → {fals['F2_interpretation']}")
    print(f"  Track B final rel→avg: {fals['F2_track_B_final_rel_to_avg']:.4f}")

    spec = summary["track_C_spectral"]
    print()
    print("--- Track C: spectral diagnostic of late-iterated state ---")
    print(
        f"After {spec['iterations']} iterations at τ_g={spec['tau_g_fixed']}, "
        f"late state ordered by νf = k·log(p):"
    )
    print(
        f"  ‖late_state‖_L2 = {spec['late_state_l2_norm']:.4e}, "
        f"mean = {spec['late_state_mean']:.4e}"
    )
    print(
        f"  Total oscillatory power (post-demean): {spec['total_oscillatory_power']:.4e}"
    )
    print(f"  DC fraction post-demean: {spec['dc_fraction_post_demean']:.4e}")
    print(f"  Top-3 freq bins: {spec['top3_freq_bins']}")
    print(
        f"  Top-3 power fractions: "
        f"{[f'{f:.4f}' for f in spec['top3_power_fraction']]}"
    )
    print("=" * 78)


def main() -> None:
    summary = run()
    out_dir = Path("results") / "remesh_infinity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remesh_infinity_riemann_baseline.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print_table(summary)
    print(f"JSON output: {out_path}")


if __name__ == "__main__":
    main()
