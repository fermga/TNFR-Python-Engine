#!/usr/bin/env python3
"""Conservation Law Validation — Systematic Benchmark.

Validates the Structural Conservation Theorem across topologies, sizes,
and dynamics regimes.  Reports conservation quality, Lyapunov stability,
sector coupling, and scaling behaviour.

Key predictions verified:
1. Charge drift < 0.1% across all topologies
2. Energy monotonically non-increasing (Lyapunov)
3. Cross-coupling kappa ~ 0.5-0.8 (Psi unification)
4. Conservation quality q(N) ~ 1 - C/sqrt(N) (scaling)

References:
    theory/STRUCTURAL_CONSERVATION_THEOREM.md
    src/tnfr/physics/conservation.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import networkx as nx  # noqa: E402

from tnfr.constants import inject_defaults  # noqa: E402
from tnfr.physics.conservation import (  # noqa: E402
    ConservationTracker,
    capture_conservation_snapshot,
    compute_conservation_scaling,
    compute_energy_functional,
    compute_lyapunov_derivative,
    compute_noether_charge,
    compute_spectral_conservation,
    analyze_sector_coupling,
    verify_conservation_balance,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_graph(n: int, topology: str, seed: int) -> nx.Graph:
    """Build a TNFR graph with specified topology."""
    if topology == "ws":
        k = max(4, min(n - 1, 4))
        G = nx.watts_strogatz_graph(n, k, 0.3, seed=seed)
    elif topology == "ba":
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif topology == "grid":
        side = int(math.sqrt(n))
        G = nx.grid_2d_graph(side, side)
    elif topology == "complete":
        G = nx.complete_graph(n)
    else:
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)

    rng = np.random.default_rng(seed)
    inject_defaults(G)
    for nd in G.nodes():
        G.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
        G.nodes[nd]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[nd]["nu_f"] = G.nodes[nd]["frequency"]
        G.nodes[nd]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[nd]["EPI"] = rng.uniform(0.5, 2.0)
    return G


def _evolve(G: nx.Graph, dt: float, n_steps: int) -> None:
    """Evolve the network for n_steps via nodal equation dynamics."""
    for _ in range(n_steps):
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


# ---------------------------------------------------------------------------
# Benchmark 1: Topology sweep
# ---------------------------------------------------------------------------

def benchmark_topology_sweep() -> None:
    """Measure conservation across topologies at fixed size."""
    print("=" * 65)
    print("  BENCHMARK 1: Topology Sweep (N=30, 20 steps)")
    print("=" * 65)

    topologies = ["ws", "ba", "grid", "complete"]
    sizes = [30, 30, 25, 30]  # grid must be square
    dt = 0.01
    n_steps = 20

    print(f"{'Topology':<12} {'Q_drift':>10} {'Rel_drift':>12} "
          f"{'Quality':>10} {'E_decay':>10} {'kappa':>8}")
    print("-" * 62)

    for topo, n in zip(topologies, sizes):
        G = _build_graph(n, topo, seed=42)
        Q0 = compute_noether_charge(G)
        E0 = compute_energy_functional(G)

        tracker = ConservationTracker(G)
        tracker.record(t=0.0)

        for step in range(n_steps):
            _evolve(G, dt, 1)
            tracker.record(t=(step + 1) * dt)

        report = tracker.report()
        Q_final = compute_noether_charge(G)

        # Sector coupling from last pair
        snaps = tracker._snapshots
        coupling = analyze_sector_coupling(
            snaps[-2][1], snaps[-1][1], dt=dt
        )
        kappa = coupling["cross_coupling_strength"]

        q_drift = abs(Q_final - Q0)
        rel_drift = q_drift / max(abs(Q0), 1e-15)
        e_decay = E0 - compute_energy_functional(G)

        print(
            f"{topo:<12} {q_drift:>10.6f} {rel_drift:>12.2e} "
            f"{report.mean_quality:>10.4f} {e_decay:>10.4f} {kappa:>8.4f}"
        )

    print()


# ---------------------------------------------------------------------------
# Benchmark 2: Lyapunov stability check
# ---------------------------------------------------------------------------

def benchmark_lyapunov_stability() -> None:
    """Verify dE/dt <= 0 across many steps and topologies."""
    print("=" * 65)
    print("  BENCHMARK 2: Lyapunov Stability (dE/dt <= 0)")
    print("=" * 65)

    configs = [
        ("ws", 30, 50), ("ba", 30, 50),
        ("grid", 25, 50), ("complete", 20, 50),
    ]

    dt = 0.01

    print(f"{'Topology':<12} {'Steps':>6} {'Stable':>8} {'Pct':>8} "
          f"{'Mean_dE/dt':>12} {'Max_dE/dt':>12}")
    print("-" * 58)

    for topo, n, n_steps in configs:
        G = _build_graph(n, topo, seed=42)
        stable = 0
        de_dt_values = []

        for step in range(n_steps):
            before = capture_conservation_snapshot(G)
            _evolve(G, dt, 1)
            after = capture_conservation_snapshot(G)
            lyap = compute_lyapunov_derivative(before, after, dt=dt)

            de_dt_values.append(lyap.energy_derivative)
            if lyap.is_stable:
                stable += 1

        arr = np.array(de_dt_values)
        print(
            f"{topo:<12} {n_steps:>6} {stable:>8} "
            f"{100 * stable / n_steps:>7.1f}% "
            f"{float(np.mean(arr)):>+12.4f} {float(np.max(arr)):>+12.6f}"
        )

    print()


# ---------------------------------------------------------------------------
# Benchmark 3: Spectral analysis
# ---------------------------------------------------------------------------

def benchmark_spectral_analysis() -> None:
    """Analyse conservation in the graph Laplacian eigenbasis."""
    print("=" * 65)
    print("  BENCHMARK 3: Spectral Conservation Analysis")
    print("=" * 65)

    configs = [("ws", 30), ("ba", 30), ("grid", 25)]

    print(f"{'Topology':<12} {'Gap':>8} {'Conserved':>12} {'Mode0_rho':>12}")
    print("-" * 44)

    for topo, n in configs:
        G = _build_graph(n, topo, seed=42)
        spec = compute_spectral_conservation(G)
        total = len(spec.eigenvalues)
        print(
            f"{topo:<12} {spec.spectral_gap:>8.4f} "
            f"{spec.dominant_conservation_modes:>5}/{total:<5} "
            f"{spec.rho_spectrum[0]:>12.4f}"
        )

    print()


# ---------------------------------------------------------------------------
# Benchmark 4: Scaling behaviour  q(N) ~ 1 - C/sqrt(N)
# ---------------------------------------------------------------------------

def benchmark_scaling() -> None:
    """Verify conservation quality scaling with network size."""
    print("=" * 65)
    print("  BENCHMARK 4: Conservation Quality Scaling q(N)")
    print("=" * 65)

    sizes = [10, 20, 40, 80, 150]
    seed = 42

    topologies = []
    for n in sizes:
        k = max(4, min(n - 1, 4))
        G = _build_graph(n, "ws", seed=seed)
        topologies.append((G, f"WS({n})"))

    result = compute_conservation_scaling(
        topologies, dt=0.01, n_steps=15, seed=seed
    )

    print(f"{'Label':<12} {'N':>6} {'Quality':>10}")
    print("-" * 28)
    for label, n, q in zip(result["labels"], result["sizes"],
                           result["qualities"]):
        print(f"{label:<12} {n:>6} {q:>10.4f}")

    print()
    print(f"Fit: q(N) ~ 1 - {result['fit_C']:.3f} / sqrt(N)")
    print(f"R-squared: {result['fit_R2']:.4f}")

    # Prediction for N=1000
    q_pred = 1.0 - result["fit_C"] / math.sqrt(1000)
    print(f"Predicted q(1000) = {q_pred:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR STRUCTURAL CONSERVATION LAW — BENCHMARK SUITE")
    print("  Systematic validation across topologies and sizes")
    print("*" * 65)
    print()

    benchmark_topology_sweep()
    benchmark_lyapunov_stability()
    benchmark_spectral_analysis()
    benchmark_scaling()

    print("=" * 65)
    print("  BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
