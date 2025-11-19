"""
Fundamentals Precision Walk (TNFR)

High-precision evolution of the canonical nodal equation with structural
telemetry at each step. Designed to deepen insight into TNFR dynamics using
the Structural Field Tetrad and U2 integral convergence.

Equation: ∂EPI/∂t = νf · ΔNFR(t)

Outputs JSONL with per-step metrics:
- phi_s, phase_grad, phase_curv, xi_c
- integral_nuF_dnfr (global and per-node mean)
- mean_distance and coherence_length_ratio (xi_c/mean_distance)

Usage (PowerShell):
  $env:PYTHONPATH = "src"
  .\test-env\Scripts\python.exe benchmarks/fundamentals_precision_walk.py `
    --n 2000 --topology er --steps 50 --dt 0.01 `
    --avg-degree 6 --seed 42 `
    --landmark-ratio 0.02 --validate --sample-size 64 --max-refinements 2 `
    --output benchmarks/results/precision_walk_er_2000.jsonl

Notes:
- Guardrail: skips n>4000 unless --allow-large is specified.
- Uses canonical field functions and update_epi_via_nodal_equation().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, cast

import numpy as np
import networkx as nx

from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.physics.canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from tnfr.utils.cache import invalidate_function_cache


_FIELD_CACHE_FUNCS = (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)


def _invalidate_field_caches() -> None:
    for fn in _FIELD_CACHE_FUNCS:
        try:
            invalidate_function_cache(fn)
        except ValueError:
            pass


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _build_graph(topology: str, n: int, avg_degree: int, seed: int) -> Any:
    rng = np.random.RandomState(seed)
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
            k += 1
        beta = 0.1
        return nx.watts_strogatz_graph(
            n, min(k, max(2, n - 1 - (n - 1) % 2)), beta,
            seed=int(rng.randint(0, 2**31 - 1)),
        )
    raise ValueError(f"Unsupported topology: {topology}")


def _seed_state(
    G: Any,
    seed: int,
    vf: float = 1.0,
    dnfr_mode: str = "bimodal",
) -> None:
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(nodes))
    if dnfr_mode == "uniform0":
        dnfr: np.ndarray = np.zeros(len(nodes), dtype=float)
    elif dnfr_mode == "normal":
        dnfr = rng.normal(loc=0.0, scale=1.0, size=len(nodes))
    else:
        half = len(nodes) // 2
        pos = rng.normal(loc=+1.0, scale=0.1, size=half)
        neg = rng.normal(loc=-1.0, scale=0.1, size=len(nodes) - half)
        dnfr = np.concatenate([pos, neg])
        rng.shuffle(dnfr)
    for i, node in enumerate(nodes):
        nd = G.nodes[node]
        nd["theta"] = float(phases[i])
        nd["delta_nfr"] = float(dnfr[i])
        nd["vf"] = float(vf)
        nd["EPI"] = float(0.0)
        nd["dEPI_dt"] = float(0.0)


def _mean_distance(
    G: Any,
    rng: np.random.RandomState,
    samples: int = 16,
) -> float:
    if G.number_of_nodes() < 2:
        return float("nan")
    nodes = list(G.nodes())
    roots = rng.choice(nodes, size=min(samples, len(nodes)), replace=False)
    s = 0.0
    c = 0
    for r in roots:
        lengths = nx.single_source_shortest_path_length(G, r)
        for v, d in lengths.items():
            if v != r and d > 0:
                s += float(d)
                c += 1
    return s / c if c else float("nan")


def run(
    *,
    n: int,
    topology: str,
    steps: int,
    dt: float,
    avg_degree: int,
    seed: int,
    landmark_ratio: float,
    validate: bool,
    sample_size: int,
    max_refinements: int,
    output: str,
    dnfr_mode: str,
    allow_large: bool,
    export_node_fields: bool,
    oz_il: bool,
    oz_fraction: float,
    oz_every: int,
    oz_bursts: int,
    use_extended_dynamics: bool,
) -> None:
    if int(n) > 4000 and not allow_large:
        print(
            f"[precision-walk] Skipping n={n} (>4000). Use --allow-large to "
            "override.",
            file=sys.stderr,
        )
        return
    _ensure_dir(output)

    G = _build_graph(topology, n, avg_degree, seed)
    _seed_state(G, seed, vf=1.0, dnfr_mode=dnfr_mode)

    # Enable extended dynamics when explicitly requested or when
    # running OZ→IL sequences so phase/ΔNFR fields evolve each step.
    if use_extended_dynamics or oz_il:
        G.graph["use_extended_dynamics"] = True

    integ = 0.0  # ∫ νf · ΔNFR dt (global, mean over nodes per step)
    rng = np.random.RandomState(seed)

    _invalidate_field_caches()

    with open(output, "w", encoding="utf-8") as f:
        for step in range(int(steps)):
            # Canonical fields (high precision for Φ_s)
            phi_s = compute_structural_potential(
                G,
                alpha=2.0,
                landmark_ratio=landmark_ratio,
                validate=validate,
                error_epsilon=0.05,
                max_refinements=max_refinements,
                sample_size=sample_size,
            )
            grad = compute_phase_gradient(G)
            curv = compute_phase_curvature(G)
            xi_c = estimate_coherence_length(G)

            # Accumulate integral (mean νf·ΔNFR dt) structural pressure proxy
            vf_dnfr_mean = 0.0
            if G.number_of_nodes():
                vf_dnfr_mean = (
                    float(
                        sum(
                            float(G.nodes[i].get("vf", 0.0))
                            * float(G.nodes[i].get("delta_nfr", 0.0))
                            for i in G.nodes()
                        )
                    )
                    / float(G.number_of_nodes())
                )
            integ += vf_dnfr_mean * float(dt)

            md = _mean_distance(G, rng, samples=min(16, max(4, n // 500)))
            clr = (
                float(xi_c / md)
                if (not math.isnan(xi_c) and not math.isnan(md) and md > 0.0)
                else float("nan")
            )

            # Aggregate stats (means + stds) for variance convergence analysis
            phi_s_values = [
                v for k, v in phi_s.items() if not str(k).startswith("__")
            ]
            grad_values = list(grad.values())
            curv_values = list(curv.values())

            rec = {
                "t": step * float(dt),
                "timestamp": _now_iso(),
                "topology": topology,
                "n": int(n),
                "seed": int(seed),
                "dt": float(dt),
                "step": int(step),
                "metrics": {
                    "phi_s": {
                        "mean": float(np.mean(phi_s_values))
                        if phi_s_values else float("nan"),
                        "std": float(np.std(phi_s_values))
                        if phi_s_values else float("nan"),
                        "max": float(np.max(phi_s_values))
                        if phi_s_values else float("nan"),
                        "abs_max": float(np.max(np.abs(phi_s_values)))
                        if phi_s_values else float("nan"),
                    },
                    "phase_grad": {
                        "mean": float(np.mean(grad_values))
                        if grad_values else float("nan"),
                        "std": float(np.std(grad_values))
                        if grad_values else float("nan"),
                        "max": float(np.max(grad_values))
                        if grad_values else float("nan"),
                        "abs_max": float(np.max(np.abs(grad_values)))
                        if grad_values else float("nan"),
                    },
                    "phase_curv": {
                        "mean": float(np.mean(curv_values))
                        if curv_values else float("nan"),
                        "std": float(np.std(curv_values))
                        if curv_values else float("nan"),
                        "max": float(np.max(curv_values))
                        if curv_values else float("nan"),
                        "abs_max": float(np.max(np.abs(curv_values)))
                        if curv_values else float("nan"),
                    },
                    "xi_c": float(xi_c),
                    "coherence_length_ratio": clr,
                    "integral_vf_dnfr": float(integ),
                    "vf_dnfr_mean": float(vf_dnfr_mean),
                },
            }

            if export_node_fields:
                # Degree-based stratification bins:
                # top5%, next15%, mid60%, bottom20%
                degrees = dict(G.degree())
                sorted_nodes = sorted(
                    degrees.keys(), key=lambda x: degrees[x], reverse=True
                )
                total_nodes = len(sorted_nodes)

                def _slice_indices(frac: float) -> int:
                    return int(math.ceil(frac * total_nodes))
                top5_end = _slice_indices(0.05)
                next15_end = top5_end + _slice_indices(0.15)
                mid60_end = next15_end + _slice_indices(0.60)
                top5_nodes = sorted_nodes[:top5_end]
                next15_nodes = sorted_nodes[top5_end:next15_end]
                mid60_nodes = sorted_nodes[next15_end:mid60_end]
                bottom20_nodes = sorted_nodes[mid60_end:]

                def _stats(vals: list[float]) -> tuple[float, float]:
                    if not vals:
                        return float("nan"), float("nan")
                    arr = np.asarray(vals, dtype=float)
                    return float(np.mean(arr)), float(np.std(arr))

                strat = {
                    "top5": {
                        "count": len(top5_nodes),
                        "curv_mean": _stats([
                            curv[n] for n in top5_nodes if n in curv
                        ])[0],
                        "curv_std": _stats([
                            curv[n] for n in top5_nodes if n in curv
                        ])[1],
                        "grad_mean": _stats([
                            grad[n] for n in top5_nodes if n in grad
                        ])[0],
                        "grad_std": _stats([
                            grad[n] for n in top5_nodes if n in grad
                        ])[1],
                    },
                    "next15": {
                        "count": len(next15_nodes),
                        "curv_mean": _stats([
                            curv[n] for n in next15_nodes if n in curv
                        ])[0],
                        "curv_std": _stats([
                            curv[n] for n in next15_nodes if n in curv
                        ])[1],
                        "grad_mean": _stats([
                            grad[n] for n in next15_nodes if n in grad
                        ])[0],
                        "grad_std": _stats([
                            grad[n] for n in next15_nodes if n in grad
                        ])[1],
                    },
                    "mid60": {
                        "count": len(mid60_nodes),
                        "curv_mean": _stats([
                            curv[n] for n in mid60_nodes if n in curv
                        ])[0],
                        "curv_std": _stats([
                            curv[n] for n in mid60_nodes if n in curv
                        ])[1],
                        "grad_mean": _stats([
                            grad[n] for n in mid60_nodes if n in grad
                        ])[0],
                        "grad_std": _stats([
                            grad[n] for n in mid60_nodes if n in grad
                        ])[1],
                    },
                    "bottom20": {
                        "count": len(bottom20_nodes),
                        "curv_mean": _stats([
                            curv[n] for n in bottom20_nodes if n in curv
                        ])[0],
                        "curv_std": _stats([
                            curv[n] for n in bottom20_nodes if n in curv
                        ])[1],
                        "grad_mean": _stats([
                            grad[n] for n in bottom20_nodes if n in grad
                        ])[0],
                        "grad_std": _stats([
                            grad[n] for n in bottom20_nodes if n in grad
                        ])[1],
                    },
                }
                rec["node_stratification"] = strat

            # Emit any Φ_s meta if available
            if (
                "__phi_s_landmark_ratio__" in phi_s
                or "__phi_s_rmae__" in phi_s
            ):
                rec["meta"] = {}
                meta_dict = cast(
                    dict, rec["meta"]
                )  # explicit cast for type checker
                if "__phi_s_landmark_ratio__" in phi_s:
                    meta_dict["phi_s_landmark_ratio"] = float(
                        phi_s["__phi_s_landmark_ratio__"]
                    )
                if "__phi_s_rmae__" in phi_s:
                    meta_dict["phi_s_rmae"] = float(
                        phi_s["__phi_s_rmae__"]
                    )

            f.write(json.dumps(rec) + "\n")
            f.flush()

            # Apply optional OZ→IL sequence (destabilize then stabilize)
            if oz_il and (step % max(1, oz_every) == 0):
                nodes = list(G.nodes())
                if nodes:
                    import random as _r
                    k = max(
                        1,
                        int(
                            len(nodes)
                            * max(0.001, min(0.5, oz_fraction))
                        ),
                    )
                    targets = _r.sample(nodes, k)
                    # Apply via unified centralized grammar API
                    try:
                        from tnfr.operators.grammar import (  # type: ignore
                            apply_glyph_with_grammar as _apply_with_grammar,
                        )
                        bursts = max(1, int(oz_bursts))
                        for _ in range(bursts):
                            _apply_with_grammar(G, targets, 'OZ')
                        _apply_with_grammar(G, targets, 'IL')
                    except Exception:
                        # If grammar API unavailable, skip OZ→IL
                        pass

            # Advance dynamics via canonical nodal equation
            update_epi_via_nodal_equation(G, dt=float(dt), method="euler")
            _invalidate_field_caches()


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TNFR Fundamentals Precision Walk")
    p.add_argument("--n", type=int, default=2000, help="Number of nodes")
    p.add_argument(
        "--topology",
        type=str,
        default="er",
        choices=["er", "ws", "ba"],
    )
    p.add_argument("--steps", type=int, default=50, help="Integration steps")
    p.add_argument("--dt", type=float, default=0.01, help="Time step dt")
    p.add_argument("--avg-degree", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--landmark-ratio", type=float, default=0.02)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--max-refinements", type=int, default=2)
    p.add_argument(
        "--dnfr-mode",
        type=str,
        default="bimodal",
        choices=["bimodal", "normal", "uniform0"],
    )
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    p.add_argument(
        "--output",
        type=str,
        default=f"benchmarks/results/precision_walk_{ts}.jsonl",
    )
    p.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow n>4000 (may be slow)",
    )
    p.add_argument(
        "--use-extended-dynamics",
        action="store_true",
        help="Enable extended nodal dynamics (phase + ΔNFR updates)",
    )
    p.add_argument(
        "--export-node-fields",
        action="store_true",
        help="Export degree-stratified curvature/gradient bins per step",
    )
    p.add_argument(
        "--oz-il",
        action="store_true",
        help="Apply OZ→IL sequence each oz-every steps (U2/U4 compliant)",
    )
    p.add_argument(
        "--oz-fraction",
        type=float,
        default=0.02,
        help="Fraction of nodes targeted by OZ per application (0-0.5)",
    )
    p.add_argument(
        "--oz-every",
        type=int,
        default=1,
        help="Apply OZ→IL every k steps (k>=1)",
    )
    p.add_argument(
        "--oz-bursts",
        type=int,
        default=1,
        help="Number of OZ applications before IL when OZ is triggered",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse(sys.argv[1:] if argv is None else argv)
    try:
        run(
            n=int(ns.n),
            topology=str(ns.topology),
            steps=int(ns.steps),
            dt=float(ns.dt),
            avg_degree=int(ns.avg_degree),
            seed=int(ns.seed),
            landmark_ratio=float(ns.landmark_ratio),
            validate=bool(ns.validate),
            sample_size=int(ns.sample_size),
            max_refinements=int(ns.max_refinements),
            output=str(ns.output),
            dnfr_mode=str(ns.dnfr_mode),
            allow_large=bool(ns.allow_large),
            export_node_fields=bool(ns.export_node_fields),
            oz_il=bool(ns.oz_il),
            oz_fraction=float(ns.oz_fraction),
            oz_every=int(ns.oz_every),
            oz_bursts=int(ns.oz_bursts),
            use_extended_dynamics=bool(ns.use_extended_dynamics),
        )
        print(f"Wrote precision walk to: {ns.output}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
