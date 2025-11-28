#!/usr/bin/env python3
"""
K_φ Safety Demo: print Φ_s drift, |∇φ| mean, and K_φ multiscale safety verdict.

This demo creates a small graph, initializes TNFR nodes, applies a brief
operator sequence, and reports three telemetry metrics side-by-side:
- Φ_s drift (U6 safety)
- |∇φ| mean (local stress)
- K_φ multiscale safety verdict + alpha fit

Usage: python benchmarks/k_phi_safety_demo.py
"""

import sys
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import (  # noqa: E402
    compute_structural_potential,
    compute_phase_gradient,
    k_phi_multiscale_safety,
)
from benchmarks.benchmark_utils import (  # noqa: E402
    create_tnfr_topology,
    initialize_tnfr_nodes,
)
from src.tnfr.operators.definitions import Dissonance, Coherence  # noqa: E402


def main():
    print("🛡️ TNFR Safety Triad Demo (Φ_s, |∇φ|, K_φ)")
    topo = 'ws'
    seed = 1234
    n_nodes = 40

    G = create_tnfr_topology(topo, n_nodes, seed)
    initialize_tnfr_nodes(G, seed=seed)

    # Baseline metrics
    phi_before = compute_structural_potential(G)
    grad_before = compute_phase_gradient(G)

    # Apply a small sequence
    rng = random.Random(seed)
    nodes = list(G.nodes())
    for node in nodes:
        if rng.random() < 0.4:
            Dissonance()(G, node)
        if rng.random() < 0.2:
            Coherence()(G, node)

    # Post metrics
    phi_after = compute_structural_potential(G)
    grad_after = compute_phase_gradient(G)

    # Φ_s drift
    drift = np.mean([abs(phi_after[n] - phi_before[n]) for n in G.nodes()])

    # |∇φ| means
    grad_mean_before = float(np.mean(list(grad_before.values())))
    grad_mean_after = float(np.mean(list(grad_after.values())))

    # K_φ safety (use global alpha_hint from Task 3)
    safety = k_phi_multiscale_safety(G, scales=(1, 2, 3, 5), alpha_hint=2.76)

    print("\nResults:")
    print(f"- Topology: {topo}, nodes={n_nodes}, seed={seed}")
    print(f"- Φ_s drift (U6 threshold < 2.0): {drift:.3f}")
    print(f"- |∇φ| mean: {grad_mean_before:.3f} → {grad_mean_after:.3f}")
    print("- K_φ multiscale safety:")
    print(
        f"   safe={safety['safe']} fit: α={safety['fit']['alpha']:.2f}, "
        f"R²={safety['fit']['r_squared']:.3f}, n={safety['fit']['n_points']}"
    )
    if safety.get('violations'):
        print(f"   tolerance violations at scales: {safety['violations']}")


if __name__ == "__main__":
    main()
