#!/usr/bin/env python3
"""Structural Conservation Experiment — Minimal Reproducible Benchmark.

Central experiment for:
  "A Structural Continuity Law for Grammar-Constrained Dynamics in TNFR"

Runs three regimes (valid, perturbed, invalid) across five graph topologies
with fixed seeds and exports:
  - results/metrics.csv          (per-topology summary table)
  - results/figures/*.png        (publication-ready figures)

Usage:
    python src/run_conservation_experiment.py

All outputs are deterministic given the fixed seeds below.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve imports: works both from repo root and from this package directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import networkx as nx  # noqa: E402

from tnfr.constants import inject_defaults  # noqa: E402
from tnfr.physics.conservation import (  # noqa: E402
    ConservationTracker,
    capture_conservation_snapshot,
    compute_energy_functional,
    compute_lyapunov_derivative,
    compute_noether_charge,
    verify_conservation_balance,
)

# ---------------------------------------------------------------------------
# Configuration — all seeds fixed for reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
N_STEPS = 40
DT = 0.01

TOPOLOGIES: dict[str, dict] = {
    "path":    {"builder": "path",   "n": 20, "seed": GLOBAL_SEED},
    "cycle":   {"builder": "cycle",  "n": 20, "seed": GLOBAL_SEED},
    "grid":    {"builder": "grid",   "n": 25, "seed": GLOBAL_SEED},
    "tree":    {"builder": "tree",   "n": 15, "seed": GLOBAL_SEED},
    "erdos":   {"builder": "erdos",  "n": 25, "seed": GLOBAL_SEED},
}

# Output paths (relative to this script's parent package)
_PKG = Path(__file__).resolve().parent.parent
RESULTS_DIR = _PKG / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_CSV = RESULTS_DIR / "metrics.csv"


# ===================================================================
# Graph construction
# ===================================================================

def build_graph(cfg: dict) -> nx.Graph:
    """Build a TNFR-initialized graph from topology configuration."""
    b = cfg["builder"]
    n = cfg["n"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    if b == "path":
        G = nx.path_graph(n)
    elif b == "cycle":
        G = nx.cycle_graph(n)
    elif b == "grid":
        side = int(math.isqrt(n))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif b == "tree":
        G = nx.balanced_tree(2, int(math.log2(n + 1)))
        G = nx.convert_node_labels_to_integers(G)
    elif b == "erdos":
        G = nx.erdos_renyi_graph(n, 0.3, seed=seed)
        # Ensure connected
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(f"Unknown builder: {b}")

    inject_defaults(G)
    for nd in G.nodes():
        G.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
        G.nodes[nd]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[nd]["nu_f"] = G.nodes[nd]["frequency"]
        G.nodes[nd]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[nd]["EPI"] = rng.uniform(0.5, 2.0)
    return G


# ===================================================================
# Evolution regimes
# ===================================================================

def _evolve_valid(G: nx.Graph, dt: float) -> None:
    """One step of grammar-compliant nodal-equation evolution.

    Phase update + diffusive ΔNFR coupling (U2-safe: stabilising).
    """
    for nd in G.nodes():
        nu_f = G.nodes[nd].get("nu_f", 1.0)
        dnfr = G.nodes[nd].get("delta_nfr", 0.0)
        G.nodes[nd]["phase"] += dt * nu_f * dnfr * 0.1
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
        nbrs = list(G.neighbors(nd))
        if nbrs:
            mean_dnfr = float(
                np.mean([G.nodes[j].get("delta_nfr", 0.0) for j in nbrs])
            )
            G.nodes[nd]["delta_nfr"] += dt * 0.1 * (mean_dnfr - dnfr)


def _evolve_perturbed(G: nx.Graph, dt: float, rng: np.random.Generator) -> None:
    """Valid evolution + small admissible perturbation (stress test)."""
    _evolve_valid(G, dt)
    # Add small random phase jitter (stays within grammar bounds)
    for nd in G.nodes():
        G.nodes[nd]["phase"] += rng.normal(0, 0.01)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]


def _evolve_invalid(G: nx.Graph, dt: float, rng: np.random.Generator) -> None:
    """Grammar-violating evolution (control arm).

    Breaks U2 (unbounded ΔNFR growth) and U3 (random phase jumps).
    """
    for nd in G.nodes():
        nu_f = G.nodes[nd].get("nu_f", 1.0)
        dnfr = G.nodes[nd].get("delta_nfr", 0.0)
        # Amplify ΔNFR instead of stabilising (breaks U2 convergence)
        G.nodes[nd]["delta_nfr"] += dt * 0.5 * abs(dnfr)
        G.nodes[nd]["phase"] += dt * nu_f * dnfr * 0.3
        # Random large phase jumps (breaks U3 phase compatibility)
        if rng.random() < 0.1:
            G.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]


# ===================================================================
# Experiment runner
# ===================================================================

def run_regime(
    topo_name: str, cfg: dict, regime: str
) -> dict:
    """Run a single topology × regime experiment.

    Returns dict with per-step time series and summary metrics.
    """
    G = build_graph(cfg)
    rng = np.random.default_rng(cfg["seed"] + hash(regime) % 2**31)

    Q0 = compute_noether_charge(G)
    E0 = compute_energy_functional(G)

    tracker = ConservationTracker(G)
    tracker.record(t=0.0)

    charge_series = [Q0]
    energy_series = [E0]
    quality_series: list[float] = []
    lyapunov_series: list[float] = []

    for step in range(1, N_STEPS + 1):
        before = capture_conservation_snapshot(G)

        if regime == "valid":
            _evolve_valid(G, DT)
        elif regime == "perturbed":
            _evolve_perturbed(G, DT, rng)
        elif regime == "invalid":
            _evolve_invalid(G, DT, rng)
        else:
            raise ValueError(f"Unknown regime: {regime}")

        after = capture_conservation_snapshot(G)
        tracker.record(t=step * DT)

        Q_t = compute_noether_charge(G)
        E_t = compute_energy_functional(G)
        charge_series.append(Q_t)
        energy_series.append(E_t)

        balance = verify_conservation_balance(before, after, dt=DT)
        quality_series.append(balance.conservation_quality)

        lyap = compute_lyapunov_derivative(before, after, dt=DT)
        lyapunov_series.append(lyap.energy_derivative)

    report = tracker.report()

    # Summary metrics
    Q_final = charge_series[-1]
    rel_drift = abs(Q_final - Q0) / max(abs(Q0), 1e-15)
    mean_quality = float(np.mean(quality_series)) if quality_series else 0.0
    lyap_arr = np.array(lyapunov_series)
    stable_pct = float(np.mean(lyap_arr <= 0.0)) * 100.0 if len(lyap_arr) > 0 else 0.0

    return {
        "topology": topo_name,
        "regime": regime,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "Q0": Q0,
        "Q_final": Q_final,
        "rel_drift": rel_drift,
        "mean_quality": mean_quality,
        "stable_pct": stable_pct,
        "mean_dE_dt": float(np.mean(lyap_arr)) if len(lyap_arr) > 0 else 0.0,
        "charge_series": charge_series,
        "energy_series": energy_series,
        "quality_series": quality_series,
        "lyapunov_series": lyapunov_series,
    }


# ===================================================================
# Plotting
# ===================================================================

def generate_figures(all_results: list[dict]) -> None:
    """Generate publication-ready PNG figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ts = np.arange(0, N_STEPS + 1) * DT

    # --- Figure 1: Charge vs time (valid regime, all topologies) -----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in all_results:
        if r["regime"] == "valid":
            q = np.array(r["charge_series"])
            q_norm = (q - q[0]) / max(abs(q[0]), 1e-15)
            ax.plot(ts, q_norm, label=r["topology"], linewidth=1.5)
    ax.set_xlabel("Time (structural units)")
    ax.set_ylabel("Relative charge drift (Q(t)−Q₀)/|Q₀|")
    ax.set_title("Structural Charge Conservation — Valid Regime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "charge_vs_time.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Energy vs time (valid regime, all topologies) -----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in all_results:
        if r["regime"] == "valid":
            e = np.array(r["energy_series"])
            e_norm = e / max(abs(e[0]), 1e-15)
            ax.plot(ts, e_norm, label=r["topology"], linewidth=1.5)
    ax.set_xlabel("Time (structural units)")
    ax.set_ylabel("Normalised energy E(t)/E(0)")
    ax.set_title("Lyapunov Energy Evolution — Valid Regime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "energy_vs_time.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Valid vs Invalid comparison (one topology) --------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ref_topo = "erdos"
    for r in all_results:
        if r["topology"] == ref_topo:
            q = np.array(r["charge_series"])
            q_norm = (q - q[0]) / max(abs(q[0]), 1e-15)
            lbl = r["regime"]
            style = "-" if lbl == "valid" else ("--" if lbl == "perturbed" else ":")
            axes[0].plot(ts, q_norm, label=lbl, linestyle=style, linewidth=1.5)

            e = np.array(r["energy_series"])
            e_norm = e / max(abs(e[0]), 1e-15)
            axes[1].plot(ts, e_norm, label=lbl, linestyle=style, linewidth=1.5)

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Relative charge drift")
    axes[0].set_title(f"Charge — {ref_topo} topology")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Normalised energy")
    axes[1].set_title(f"Energy — {ref_topo} topology")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "valid_vs_invalid.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Topology summary bar chart ------------------------------
    valid_results = [r for r in all_results if r["regime"] == "valid"]
    invalid_results = [r for r in all_results if r["regime"] == "invalid"]

    topos = [r["topology"] for r in valid_results]
    v_drift = [r["rel_drift"] for r in valid_results]
    i_drift = [r["rel_drift"] for r in invalid_results]

    x = np.arange(len(topos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, v_drift, width, label="Valid", color="#2196F3")
    ax.bar(x + width / 2, i_drift, width, label="Invalid", color="#F44336")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Relative charge drift")
    ax.set_title("Conservation Quality: Valid vs Invalid")
    ax.set_xticks(x)
    ax.set_xticklabels(topos)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "topology_summary.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {FIGURES_DIR}/")


# ===================================================================
# CSV export
# ===================================================================

def export_csv(all_results: list[dict]) -> None:
    """Write per-topology summary to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "topology", "regime", "nodes", "edges",
        "Q0", "Q_final", "rel_drift",
        "mean_quality", "stable_pct", "mean_dE_dt",
    ]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)

    print(f"  Metrics saved to {METRICS_CSV}")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run the full experiment."""
    print("=" * 65)
    print("  STRUCTURAL CONSERVATION EXPERIMENT")
    print("  Minimal Reproducible Benchmark for Zenodo Deposit")
    print("=" * 65)
    print()

    regimes = ["valid", "perturbed", "invalid"]
    all_results: list[dict] = []

    for topo_name, cfg in TOPOLOGIES.items():
        for regime in regimes:
            print(f"  Running {topo_name:>8s} / {regime:<10s} ... ", end="", flush=True)
            result = run_regime(topo_name, cfg, regime)
            all_results.append(result)
            print(
                f"drift={result['rel_drift']:.2e}  "
                f"quality={result['mean_quality']:.4f}  "
                f"stable={result['stable_pct']:.0f}%"
            )
        print()

    # --- Summary table ---
    print("=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    print(
        f"{'Topology':<10} {'Regime':<12} {'Nodes':>5} "
        f"{'Rel_drift':>12} {'Quality':>10} {'Stable%':>8}"
    )
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['topology']:<10} {r['regime']:<12} {r['nodes']:>5} "
            f"{r['rel_drift']:>12.2e} {r['mean_quality']:>10.4f} "
            f"{r['stable_pct']:>7.1f}%"
        )
    print()

    # --- Export ---
    export_csv(all_results)
    generate_figures(all_results)

    print()
    print("  Experiment complete. All outputs in results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
