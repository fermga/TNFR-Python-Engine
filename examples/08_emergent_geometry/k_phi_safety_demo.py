#!/usr/bin/env python3
"""
K_φ Safety Demo: print Φ_s drift, |∇φ| mean, and K_φ multiscale safety verdict.

This demo creates a small graph, initializes TNFR nodes, applies a brief
operator sequence, and reports three telemetry metrics side-by-side:
- Φ_s drift (U6 safety)
- |∇φ| mean (local stress)
- K_φ multiscale safety verdict + alpha fit

Usage: python examples/08_emergent_geometry/k_phi_safety_demo.py
"""

import math
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tnfr.config import (  # noqa: E402
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT  # noqa: E402
from tnfr.operators.definitions import Coherence, Dissonance  # noqa: E402
from tnfr.physics.fields import (  # noqa: E402
    compute_phase_gradient,
    compute_structural_potential,
    k_phi_multiscale_safety,
)


def main():
    print("TNFR configured safety observations (Phi_s, abs(grad phi), K_phi)")
    topo = "ws"
    seed = 1234
    n_nodes = 40

    # Supplied preparation, independent of the later operator events.
    G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.3, seed=seed)
    inject_defaults(G)
    G.graph["RANDOM_SEED"] = seed
    preparation = random.Random(seed)
    for node in G:
        G.nodes[node].update(
            {
                EPI_PRIMARY: preparation.uniform(0.2, 0.8),
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: preparation.uniform(0.0, 2 * math.pi),
                DNFR_PRIMARY: preparation.uniform(0.01, 0.05),
            }
        )

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

    # Use the field owner's configured fit policy; no exponent is a derived law.
    safety = k_phi_multiscale_safety(G)

    print("\nResults:")
    print(f"- Topology: {topo}, nodes={n_nodes}, seed={seed}")
    print(
        f"- Phi_s drift (selected U6 policy < "
        f"{U6_STRUCTURAL_POTENTIAL_LIMIT:.3f}): {drift:.3f}"
    )
    print(f"- abs(grad phi) mean: {grad_mean_before:.3f} -> {grad_mean_after:.3f}")
    print("- K_phi multiscale safety (configured advisory):")
    print(
        f"   available={safety['available']} safe={safety['safe']} "
        f"fit: alpha={safety['fit']['alpha']:.2f}, "
        f"R_squared={safety['fit']['r_squared']:.3f}, "
        f"n={len(safety['variance_by_scale'])}"
    )
    if safety.get("violations"):
        print(f"   tolerance violations at scales: {safety['violations']}")


if __name__ == "__main__":
    main()
