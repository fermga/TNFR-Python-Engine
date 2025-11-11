"""Demo script for research-phase structural fields (TNFR physics).

Usage (from repo root):
    python tools/fields_demo.py --topology ring --n 20 --seed 42

Generates a graph, assigns synthetic phase and coherence values, then
computes:
- Structural potential Φ_s
- Phase gradient |∇φ|
- Phase curvature K_φ
- Coherence length ξ_C

Outputs summary statistics. All quantities are NON-CANONICAL telemetry.
"""
from __future__ import annotations

import argparse
import math
import random

import networkx as nx

# Ensure 'src' is on sys.path when running from repo root
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)


def _build_graph(topology: str, n: int, seed: int) -> nx.Graph:
    random.seed(seed)
    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "star":
        G = nx.star_graph(n - 1)
    elif topology == "ws":
        # Watts-Strogatz small-world
        k = max(2, min(6, n // 4))
        G = nx.watts_strogatz_graph(n, k, 0.2, seed=seed)
    elif topology == "scale_free":
        G = nx.scale_free_graph(n, seed=seed).to_undirected()
    else:
        raise ValueError(f"Unknown topology '{topology}'")
    return G


def _assign_synthetic_fields(G: nx.Graph, seed: int) -> None:
    rng = random.Random(seed)
    for i, node in enumerate(G.nodes()):
        # Phase: structured gradient + noise
        base_phase = (2 * math.pi * i / max(len(G), 1))
        noise = rng.uniform(-0.2, 0.2)
        phase = (base_phase + noise) % (2 * math.pi)
        # Coherence: center-biased for ring/star, random for others
        if isinstance(G, nx.Graph) and G.number_of_nodes() > 0:
            coherence = 0.6 + 0.4 * math.exp(-abs(i - len(G)/2) / (0.3 * len(G)))
        else:
            coherence = 0.5 + 0.5 * rng.random()
        delta_nfr = rng.uniform(-0.5, 0.5)
        G.nodes[node]["phase"] = phase
        G.nodes[node]["coherence"] = coherence
        G.nodes[node]["delta_nfr"] = delta_nfr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="ring", choices=["ring", "star", "ws", "scale_free"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    G = _build_graph(args.topology, args.n, args.seed)
    _assign_synthetic_fields(G, args.seed)

    phi_s = compute_structural_potential(G)
    grad = compute_phase_gradient(G)
    curvature = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    def summarize(name, values):
        arr = list(values.values())
        if not arr:
            return f"{name}: empty"
        return (
            f"{name}: min={min(arr):.3f} max={max(arr):.3f} mean={sum(arr)/len(arr):.3f}"
        )

    print(f"Topology: {args.topology} | n={args.n} | seed={args.seed}")
    print(summarize("Phi_s", phi_s))
    print(summarize("|∇φ|", grad))
    print(summarize("K_φ", curvature))
    print(f"ξ_C (coherence length): {xi_c:.3f}")


if __name__ == "__main__":
    main()
