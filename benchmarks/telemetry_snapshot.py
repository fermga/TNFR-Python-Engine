"""
TNFR Telemetry Snapshot

Generates JSONL telemetry for the Structural Field Tetrad:
- Φ_s (structural potential)
- |∇φ| (phase gradient)
- K_φ (phase curvature)
- ξ_C (coherence length)

Across configurable graph sizes and topologies.

Usage (PowerShell):
  $env:PYTHONPATH = "src";
    C:/TNFR-Python-Engine/test-env/Scripts/python.exe \
        benchmarks/telemetry_snapshot.py \
    --sizes 1000 2000 4000 8000 \
    --topologies er ws ba \
    --seeds 42 43 \
    --avg-degree 6 \
    --alpha 2.0 \
    --landmark-ratio 0.02 \
    --validate \
    --output benchmarks/results/telemetry_snapshot.jsonl

Notes:
- Uses canonical field functions from tnfr.physics.canonical and
    k_phi_multiscale_safety from tnfr.physics.fields.
- Populates per-node 'theta' (phase) and 'delta_nfr' (ΔNFR) attributes.
- Defaults favor speed and stability; tune --landmark-ratio and
    --validate for Φ_s accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
try:
    import networkx as nx
except Exception as e:  # pragma: no cover
    print(f"ERROR: networkx is required: {e}", file=sys.stderr)
    sys.exit(2)

# Canonical fields
from tnfr.physics.canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from tnfr.physics.fields import k_phi_multiscale_safety

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _avg_degree(G: Any) -> float:
    n = max(1, G.number_of_nodes())
    return 2.0 * G.number_of_edges() / n

def _estimate_mean_distance(
    G: Any,
    samples: int = 16,
    rng: np.random.RandomState | None = None,
) -> float:
    """Estimate mean shortest-path distance via sampled BFS roots.

    Returns NaN for graphs with <2 nodes or completely disconnected.
    """
    if G.number_of_nodes() < 2:
        return float("nan")
    nodes = list(G.nodes())
    if not nodes:
        return float("nan")
    if rng is None:
        rng = np.random.RandomState(0)
    if samples > len(nodes):
        samples = len(nodes)
    roots = rng.choice(nodes, size=samples, replace=False)
    dists_sum = 0.0
    count = 0
    for r in roots:
        lengths = nx.single_source_shortest_path_length(G, r)
        for v, d in lengths.items():
            if v != r and d > 0:
                dists_sum += float(d)
                count += 1
    if count == 0:
        return float("nan")
    return dists_sum / count

def _select_landmarks(
    G: Any,
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    num_landmarks: int,
    rng: np.random.RandomState,
) -> List[Any]:
    scores = []
    for node in nodes:
        degree = G.degree(node)
        dnfr_contrib = abs(delta_nfr[node])
        score = degree * (1.0 + dnfr_contrib)
        scores.append((score, node))
    scores.sort(reverse=True)
    top_candidates = [n for _, n in scores[: max(1, num_landmarks * 2)]]
    k = min(num_landmarks, len(top_candidates))
    if k <= 0:
        return []
    idx = rng.choice(len(top_candidates), size=k, replace=False)
    return [top_candidates[i] for i in idx]


def _phi_s_sampled_landmarks(
    G: Any,
    alpha: float,
    landmark_ratio: float,
    cap_landmarks: int,
    sample_sources: int,
    sample_size_exact: int,
    rng: np.random.RandomState,
) -> tuple[Dict[Any, float], Dict[str, float]]:
    """Compute Φ_s for a subset of sources using a landmark approximation."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, {}
    # Determine landmarks (respect cap)
    L = max(3, int(min(landmark_ratio * n, cap_landmarks)))
    delta_nfr = {
        node: float(G.nodes[node].get("delta_nfr", 0.0)) for node in nodes
    }
    landmarks = _select_landmarks(G, nodes, delta_nfr, L, rng)
    # Precompute distances from landmarks
    landmark_distances: Dict[Any, Dict[Any, float]] = {}
    for lm in landmarks:
        distances = (
            nx.single_source_dijkstra_path_length(G, lm, weight="weight")
            if G.number_of_edges() > 0
            else {lm: 0.0}
        )
        landmark_distances[lm] = distances
    # Sample sources
    k_src = min(sample_sources, n)
    src_nodes = list(rng.choice(nodes, size=k_src, replace=False))

    potentials: Dict[Any, float] = {}
    for src in src_nodes:
        total = 0.0
        # Landmark exact contributions
        for lm in landmarks:
            if lm == src:
                continue
            d = landmark_distances[lm].get(src, math.inf)
            if math.isfinite(d) and d > 0.0:
                total += delta_nfr[lm] / (d ** alpha)
        # Non-landmarks approximated via min over landmarks
        for dst in nodes:
            if dst == src or dst in landmarks:
                continue
            min_approx = math.inf
            for lm in landmarks:
                d_src = landmark_distances[lm].get(src, math.inf)
                d_dst = landmark_distances[lm].get(dst, math.inf)
                if math.isfinite(d_src) and math.isfinite(d_dst):
                    approx = abs(d_src - d_dst)
                    if approx <= 0.0:
                        approx = 1.0
                    if approx < min_approx:
                        min_approx = approx
            if math.isfinite(min_approx) and min_approx > 0.0:
                total += delta_nfr[dst] / (min_approx ** alpha)
        potentials[src] = total

    # Optional exact subset RMAE for meta (no refinement here)
    meta: Dict[str, float] = {"phi_s_landmark_ratio": float(L / max(1, n))}
    if sample_size_exact > 0 and n >= 50:
        subset = list(
            rng.choice(nodes, size=min(sample_size_exact, n), replace=False)
        )
        exact_vals: Dict[Any, float] = {}
        for src in subset:
            lengths = (
                nx.single_source_dijkstra_path_length(G, src, weight="weight")
                if G.number_of_edges() > 0
                else {src: 0.0}
            )
            accum = 0.0
            for dst in nodes:
                if dst == src:
                    continue
                d = lengths.get(dst, math.inf)
                if not math.isfinite(d) or d <= 0.0:
                    continue
                accum += delta_nfr[dst] / (d ** alpha)
            exact_vals[src] = accum
        overlap = [s for s in subset if s in potentials]
        if overlap:
            abs_errors = []
            exact_ref = []
            for s in overlap:
                abs_errors.append(abs(exact_vals[s] - potentials[s]))
                exact_ref.append(abs(exact_vals[s]))
            denom = (sum(exact_ref) / len(exact_ref)) if exact_ref else 1.0
            rmae = (sum(abs_errors) / len(abs_errors)) / denom if denom else 0.0
            meta["phi_s_rmae"] = float(rmae)
    return potentials, meta

def _seed_node_attributes(
    G: Any,
    rng: np.random.RandomState,
    dnfr_mode: str = "bimodal",
) -> None:
    """Attach canonical node attributes: 'theta' (phase) and
    'delta_nfr' (ΔNFR).

    dnfr_mode:
      - 'bimodal' (default): half positive ~N(+1,0.1), half negative ~N(-1,0.1)
      - 'normal':  ~N(0, 1)
      - 'uniform0': all zeros (degenerate; Φ_s will be zero)
    """
    nodes = list(G.nodes())
    if not nodes:
        return
    # Phase: uniform in [0, 2π)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(nodes))

    # ΔNFR patterns
    if dnfr_mode == "uniform0":
        dnfr = np.zeros(len(nodes), dtype=float)
    elif dnfr_mode == "normal":
        dnfr = rng.normal(loc=0.0, scale=1.0, size=len(nodes))
    else:  # bimodal
        half = len(nodes) // 2
        pos = rng.normal(loc=+1.0, scale=0.1, size=half)
        neg = rng.normal(loc=-1.0, scale=0.1, size=len(nodes) - half)
        dnfr = np.concatenate([pos, neg])
        rng.shuffle(dnfr)

    for i, node in enumerate(nodes):
        G.nodes[node]["theta"] = float(phases[i])
        G.nodes[node]["delta_nfr"] = float(dnfr[i])


def _build_graph(
    topology: str,
    n: int,
    avg_degree: int,
    rng: np.random.RandomState,
) -> Any:
    topology = topology.lower()
    if topology == "er":
        p = max(0.0, min(1.0, float(avg_degree) / max(1, n - 1)))
        return nx.fast_gnp_random_graph(
            n, p, seed=int(rng.randint(0, 2**31 - 1))
        )
    if topology == "ba":
        m = max(1, int(max(1, avg_degree) // 2))
        return nx.barabasi_albert_graph(
            n, m, seed=int(rng.randint(0, 2**31 - 1))
        )
    if topology == "ws":
        k = max(2, int(avg_degree))
        if k % 2 == 1:
            k += 1  # watts_strogatz requires even k
        beta = 0.1
        return nx.watts_strogatz_graph(
            n,
            min(k, max(2, n - 1 - (n - 1) % 2)),
            beta,
            seed=int(rng.randint(0, 2**31 - 1)),
        )
    raise ValueError(f"Unsupported topology: {topology}")


def _summarize(values: Dict[Any, float]) -> Dict[str, float]:
    arr = np.array(
        [v for k, v in values.items() if not str(k).startswith("__")],
        dtype=float,
    )
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "max": float("nan"),
            "abs_max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "abs_max": float(np.max(np.abs(arr))),
    }


def run_snapshot(
    sizes: List[int],
    topologies: List[str],
    seeds: List[int],
    avg_degree: int,
    alpha: float,
    landmark_ratio: float | None,
    validate: bool,
    error_epsilon: float,
    max_refinements: int,
    sample_size: int,
    dnfr_mode: str,
    output_path: str,
    phi_s_sample: int,
    phi_s_cap_landmarks: int,
    skip_xi_c: bool,
    skip_kphi: bool,
    allow_large: bool = False,
) -> None:
    _ensure_dir(output_path)
    schema = "tetrad_v1"
    with open(output_path, "w", encoding="utf-8") as f:
        for topology in topologies:
            for n in sizes:
                if int(n) > 4000 and not allow_large:
                    msg = (
                        f"[telemetry] Skipping n={n} (>4000). "
                        f"Use --allow-large to override."
                    )
                    print(msg, file=sys.stderr)
                    continue
                for seed in seeds:
                    rng = np.random.RandomState(int(seed))
                    G = _build_graph(topology, int(n), int(avg_degree), rng)
                    _seed_node_attributes(G, rng, dnfr_mode=dnfr_mode)

                    # Effective landmark ratio honoring absolute cap
                    eff_ratio = None
                    if landmark_ratio is not None:
                        eff_ratio = min(
                            float(landmark_ratio),
                            float(phi_s_cap_landmarks) / max(1.0, float(n)),
                        )

                    if phi_s_sample and phi_s_sample > 0:
                        phi_s, phi_meta = _phi_s_sampled_landmarks(
                            G,
                            alpha=alpha,
                            landmark_ratio=(
                                eff_ratio if eff_ratio is not None else 0.02
                            ),
                            cap_landmarks=int(phi_s_cap_landmarks),
                            sample_sources=int(phi_s_sample),
                            sample_size_exact=int(sample_size),
                            rng=rng,
                        )
                    else:
                        phi_s = compute_structural_potential(
                            G,
                            alpha=alpha,
                            landmark_ratio=eff_ratio,
                            validate=validate,
                            error_epsilon=error_epsilon,
                            max_refinements=max_refinements,
                            sample_size=sample_size,
                        )
                        phi_meta = {}

                    grad = compute_phase_gradient(G)
                    curv = compute_phase_curvature(G)
                    xi_c = (
                        estimate_coherence_length(G)
                        if not skip_xi_c
                        else float("nan")
                    )
                    kphi_safe = (
                        k_phi_multiscale_safety(G, alpha_hint=2.76)
                        if not skip_kphi
                        else {"safe": True, "fit": {}, "violations": []}
                    )

                    # Additional reference scales
                    mean_dist = _estimate_mean_distance(
                        G,
                        samples=min(16, max(4, n // 500)),
                        rng=rng,
                    )
                    coherence_length_ratio = (
                        float(xi_c / mean_dist)
                        if (
                            not math.isnan(xi_c)
                            and not math.isnan(mean_dist)
                            and mean_dist > 0
                        )
                        else float("nan")
                    )

                    record = {
                        "timestamp": _now_iso(),
                        "schema": schema,
                        "topology": topology,
                        "n": int(n),
                        "seed": int(seed),
                        "avg_degree": float(_avg_degree(G)),
                        "params": {
                            "alpha": float(alpha),
                            "landmark_ratio": (
                                None if eff_ratio is None else float(eff_ratio)
                            ),
                            "validate": bool(validate),
                            "error_epsilon": float(error_epsilon),
                            "max_refinements": int(max_refinements),
                            "sample_size": int(sample_size),
                            "dnfr_mode": dnfr_mode,
                            "phi_s_sample": int(phi_s_sample),
                            "phi_s_cap_landmarks": int(phi_s_cap_landmarks),
                            "skip_xi_c": bool(skip_xi_c),
                            "skip_kphi": bool(skip_kphi),
                        },
                        "metrics": {
                            "phi_s": _summarize(phi_s),
                            "phase_grad": _summarize(grad),
                            "phase_curv": _summarize(curv),
                            "xi_c": float(xi_c),
                            "coherence_length_ratio": coherence_length_ratio,
                            "k_phi_multiscale": {
                                "safe": bool(kphi_safe.get("safe", False)),
                                "fit": kphi_safe.get("fit", {}),
                                "violations": kphi_safe.get("violations", []),
                            },
                        },
                    }

                    # Include Φ_s validation metadata if present
                    if (
                        "__phi_s_landmark_ratio__" in phi_s
                        or "__phi_s_rmae__" in phi_s
                    ):
                        meta_obj = record.setdefault("meta", {})
                        if isinstance(meta_obj, dict):
                            if "__phi_s_landmark_ratio__" in phi_s:
                                meta_obj["phi_s_landmark_ratio"] = float(
                                    phi_s["__phi_s_landmark_ratio__"]
                                )  # type: ignore[index]
                            if "__phi_s_rmae__" in phi_s:
                                meta_obj["phi_s_rmae"] = float(
                                    phi_s["__phi_s_rmae__"]
                                )  # type: ignore[index]

                    # Include sampled meta if present
                    if phi_meta:
                        meta_obj2 = record.setdefault("meta", {})
                        if isinstance(meta_obj2, dict):
                            for k, v in phi_meta.items():
                                meta_obj2[k] = v

                    f.write(json.dumps(record) + "\n")
                    f.flush()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TNFR structural fields telemetry (JSONL)"
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[1000, 2000, 4000, 8000],
        help="List of graph sizes",
    )
    parser.add_argument(
        "--topologies",
        nargs="*",
        type=str,
        default=["er", "ws", "ba"],
        help="Topologies: er, ws, ba",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42],
        help="Random seeds to run",
    )
    parser.add_argument(
        "--avg-degree", type=int, default=6, help="Target average degree"
    )

    parser.add_argument(
        "--alpha", type=float, default=2.0, help="Φ_s exponent α"
    )
    parser.add_argument(
        "--landmark-ratio",
        type=float,
        default=0.02,
        help="Φ_s landmark ratio (0<r≤0.5); small values for speed",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable adaptive validation for Φ_s",
    )
    parser.add_argument(
        "--error-epsilon",
        type=float,
        default=0.05,
        help="Relative mean absolute error threshold for Φ_s validation",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=3,
        help="Max Φ_s landmark ratio doublings during validation",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="Exact-comparison sample size for Φ_s validation",
    )

    parser.add_argument(
        "--dnfr-mode",
        type=str,
        default="bimodal",
        choices=["bimodal", "normal", "uniform0"],
        help="ΔNFR initialization mode",
    )

    parser.add_argument(
        "--phi-s-sample",
        type=int,
        default=0,
        help="If >0, compute Φ_s only for this many sources",
    )
    parser.add_argument(
        "--phi-s-cap-landmarks",
        type=int,
        default=128,
        help="Absolute cap on number of landmarks used for Φ_s",
    )
    parser.add_argument(
        "--skip-xi-c",
        action="store_true",
        help="Skip ξ_C (coherence length) to speed up runs",
    )
    parser.add_argument(
        "--skip-kphi",
        action="store_true",
        help="Skip K_φ multiscale safety to speed up runs",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow n > 4000 (may be very slow)",
    )

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    default_out = os.path.join(
        "benchmarks",
        "results",
        f"telemetry_snapshot_{ts}.jsonl",
    )
    parser.add_argument(
        "--output", type=str, default=default_out, help="Output JSONL path"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_snapshot(
            sizes=list(ns.sizes),
            topologies=[t.lower() for t in ns.topologies],
            seeds=list(ns.seeds),
            avg_degree=int(ns.avg_degree),
            alpha=float(ns.alpha),
            landmark_ratio=(
                float(ns.landmark_ratio)
                if ns.landmark_ratio is not None
                else None
            ),
            validate=bool(ns.validate),
            error_epsilon=float(ns.error_epsilon),
            max_refinements=int(ns.max_refinements),
            sample_size=int(ns.sample_size),
            dnfr_mode=str(ns.dnfr_mode),
            output_path=str(ns.output),
            phi_s_sample=int(ns.phi_s_sample),
            phi_s_cap_landmarks=int(ns.phi_s_cap_landmarks),
            skip_xi_c=bool(ns.skip_xi_c),
            skip_kphi=bool(ns.skip_kphi),
            allow_large=bool(ns.allow_large),
        )
        print(f"Wrote telemetry to: {ns.output}")
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  # pragma: no cover
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
