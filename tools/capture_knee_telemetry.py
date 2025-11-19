"""Capture canonical structural fields around the performance knee.

Purpose
-------
Samples the CANONICAL structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) for a
set of (nodes, p) regimes bracketing the single‑process performance knee
identified in LARGE_SIMULATIONS.md.

Regimes
-------
Default target combos (can be overridden):
  (2048, 0.05), (4096, 0.05), (8192, 0.05), (4096, 0.02)

Physics Alignment
-----------------
All computations are read‑only telemetry (grammar safe). We assign
ΔNFR values using a degree‑centered heuristic:

    ΔNFR_i := deg(i) - mean_degree

This preserves a plausible structural pressure distribution without
attempting full ΔNFR dynamics integration (avoids non‑telemetry side
effects). Phase values φ_i are drawn uniformly in [0, 2π) to provide a
generic synchrony landscape.

Output
------
Writes JSONL records with fields:
  nodes, p, seed, mean_phi_s, std_phi_s,
  mean_phase_grad, max_phase_grad,
  mean_k_phi, std_k_phi, k_phi_variance_multiscale (dict),
  coherence_length, build_seconds, field_seconds, total_seconds

Each record enables correlation of structural stress indicators with
observed performance knee documented in LARGE_SIMULATIONS.md.

Usage
-----
  python tools/capture_knee_telemetry.py --output results/telemetry_knee.jsonl

Optionally specify a different seed and custom regimes:
    python tools/capture_knee_telemetry.py \
            --regimes 2048:0.05 4096:0.05 8192:0.05 4096:0.02

TNFR Invariants
---------------
Invariant #1 respected (no direct EPI mutation). Invariant #5 respected
since phase metrics are read‑only; coupling verification not required.
All four canonical fields computed through public API in
`src/tnfr/physics/fields.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    raise RuntimeError("networkx required for telemetry capture")

from tnfr.physics.fields import (  # type: ignore[import-not-found]
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
    compute_k_phi_multiscale_variance,
    fit_k_phi_asymptotic_alpha,
)


def _parse_regimes(regimes: List[str]) -> List[Tuple[int, float]]:
    parsed: List[Tuple[int, float]] = []
    for item in regimes:
        if ':' not in item:
            raise ValueError(f"Invalid regime format '{item}' (expected N:p)")
        n_str, p_str = item.split(':', 1)
        parsed.append((int(n_str), float(p_str)))
    return parsed


def _assign_phase_and_dnfr(G: nx.Graph, rng: np.random.Generator) -> None:
    degrees = dict(G.degree())
    mean_deg = float(np.mean(list(degrees.values()))) if degrees else 0.0
    for n in G.nodes():
        # Uniform random phase in [0, 2π)
        G.nodes[n]["phase"] = float(rng.uniform(0.0, 2.0 * math.pi))
        # Degree‑centered ΔNFR proxy (telemetry-only heuristic)
        G.nodes[n]["delta_nfr"] = float(degrees.get(n, 0) - mean_deg)


def capture(
    regimes: List[Tuple[int, float]],
    seed: int,
    out_path: Path,
    phi_landmark_ratio: float | None,
    skip_kphi_multiscale: bool,
) -> None:
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for nodes, p in regimes:
            build_t0 = time.perf_counter()
            G = nx.fast_gnp_random_graph(nodes, p, seed=seed, directed=False)
            _assign_phase_and_dnfr(G, rng)
            build_t1 = time.perf_counter()

            field_t0 = time.perf_counter()
            print(
                f"[telemetry] start nodes={nodes} p={p} seed={seed}"
            )
            # Φ_s computation via public API (optional landmark ratio)
            phi_s = compute_structural_potential(
                G,
                alpha=2.0,
                landmark_ratio=phi_landmark_ratio,
                validate=False,
            )
            grad = compute_phase_gradient(G)         # |∇φ|
            k_phi = compute_phase_curvature(G)       # K_φ
            xi_c = estimate_coherence_length(G)      # ξ_C (scalar)
            if skip_kphi_multiscale:
                k_phi_variance = {}
                alpha_fit = None
            else:
                k_phi_variance = compute_k_phi_multiscale_variance(G)
                alpha_fit = fit_k_phi_asymptotic_alpha(k_phi_variance)
            field_t1 = time.perf_counter()

            rec = {
                "nodes": nodes,
                "p": p,
                "seed": seed,
                "mean_phi_s": (
                    float(np.mean(list(phi_s.values()))) if phi_s else 0.0
                ),
                "std_phi_s": (
                    float(np.std(list(phi_s.values()))) if phi_s else 0.0
                ),
                "mean_phase_grad": (
                    float(np.mean(list(grad.values()))) if grad else 0.0
                ),
                "max_phase_grad": (
                    float(np.max(list(grad.values()))) if grad else 0.0
                ),
                "mean_k_phi": (
                    float(np.mean(list(k_phi.values()))) if k_phi else 0.0
                ),
                "std_k_phi": (
                    float(np.std(list(k_phi.values()))) if k_phi else 0.0
                ),
                "k_phi_variance_multiscale": k_phi_variance,
                "k_phi_alpha_fit": alpha_fit,
                "coherence_length": (
                    float(xi_c) if xi_c is not None else 0.0
                ),
                "build_seconds": build_t1 - build_t0,
                "field_seconds": field_t1 - field_t0,
                "total_seconds": (field_t1 - build_t0),
            }
            f.write(json.dumps(rec) + "\n")
            print(f"[tele] n={nodes} p={p} t={rec['total_seconds']:.2f}s")
            print(
                f"  phi={rec['mean_phi_s']:.3f} "
                f"g={rec['mean_phase_grad']:.3f} "
                f"k={rec['mean_k_phi']:.3f} "
                f"xi={rec['coherence_length']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture TNFR structural field telemetry at performance knee "
            "regimes."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/telemetry_knee.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=["2048:0.05", "4096:0.05", "8192:0.05", "4096:0.02"],
        help="List of N:p specifications (e.g. 2048:0.05)",
    )
    parser.add_argument(
        "--phi-landmark-ratio",
        type=float,
        default=None,
        help=(
            "Override landmark ratio for Φ_s approximation (e.g. 0.03). "
            "Applies only when N>500."
        ),
    )
    parser.add_argument(
        "--skip-kphi-multiscale",
        action="store_true",
        help="Skip multiscale K_φ variance (faster telemetry capture).",
    )
    args = parser.parse_args()
    regimes = _parse_regimes(args.regimes)
    capture(
        regimes,
        args.seed,
        args.output,
        args.phi_landmark_ratio,
        args.skip_kphi_multiscale,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
