"""Field Methods Battery
=========================

Compare alternative implementations ("approaches") for phase gradient and
phase curvature and record their differences across topologies. This lets
us "see what happens" when using naive linear methods vs canonical
circular (wrapped) methods on S1.

Outputs JSONL to a specified path with per-run metrics including
correlations and mean absolute deviations between approaches.

Usage (PowerShell):
    python benchmarks/field_methods_battery.py \
        --export results/field_methods_battery.jsonl \
        --topologies ring,ws,scale_free --sizes 30 --runs 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import networkx as nx

# Ensure local src is importable
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from tnfr.physics.fields import (  # type: ignore  # noqa: E402
    compute_phase_gradient,
    compute_phase_curvature,
)


def make_graph(topology: str, n: int, seed: int) -> nx.Graph:
    random.seed(seed)
    if topology == 'ring':
        return nx.cycle_graph(n)
    if topology == 'ws':
        return nx.watts_strogatz_graph(n, k=min(4, n-1), p=0.3, seed=seed)
    if topology == 'scale_free':
        return nx.scale_free_graph(n, seed=seed).to_undirected()
    if topology == 'grid':
        import math as _m
        side = max(2, int(_m.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)
    if topology == 'tree':
        G = nx.balanced_tree(r=2, h=max(1, int(math.log2(n))))
        if G.number_of_nodes() > n:
            nodes_to_remove = list(G.nodes())[n:]
            G.remove_nodes_from(nodes_to_remove)
        return G
    raise ValueError(f"Unknown topology: {topology}")


def init_state(G: nx.Graph, seed: int) -> None:
    random.seed(seed)
    for node in G.nodes:
        # Phases spread to hit wrap-around edge cases
        G.nodes[node]['phase'] = random.uniform(0.0, 2 * math.pi)
        G.nodes[node]['theta'] = G.nodes[node]['phase']


# Alternative (naive) methods -------------------------------------------------

def naive_phase_gradient(G: nx.Graph) -> Dict[Any, float]:
    grad: Dict[Any, float] = {}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        if not neigh:
            grad[i] = 0.0
            continue
        phi_i = float(G.nodes[i].get('phase', 0.0))
        diffs = [
            abs(phi_i - float(G.nodes[j].get('phase', 0.0))) for j in neigh
        ]
        grad[i] = float(np.mean(diffs))
    return grad


def naive_phase_curvature(G: nx.Graph) -> Dict[Any, float]:
    curv: Dict[Any, float] = {}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        if not neigh:
            curv[i] = 0.0
            continue
        phi_i = float(G.nodes[i].get('phase', 0.0))
        mean_lin = float(
            np.mean([float(G.nodes[j].get('phase', 0.0)) for j in neigh])
        )
        curv[i] = float(phi_i - mean_lin)
    return curv


# Metrics --------------------------------------------------------------------

def compare_fields(
    a: Dict[Any, float], b: Dict[Any, float]
) -> Dict[str, float]:
    nodes = list(set(a.keys()) & set(b.keys()))
    if not nodes:
        return {"corr": 0.0, "mad": 0.0, "rmse": 0.0}
    va = np.array([a[n] for n in nodes], dtype=float)
    vb = np.array([b[n] for n in nodes], dtype=float)
    if len(nodes) >= 3:
        C = np.corrcoef(va, vb)
        corr = float(C[0, 1])
    else:
        corr = 0.0
    mad = float(np.mean(np.abs(va - vb)))
    rmse = float(np.sqrt(np.mean((va - vb) ** 2)))
    return {"corr": corr, "mad": mad, "rmse": rmse}


# CLI ------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Field Methods Battery (compare naive vs canonical)"
    )
    p.add_argument('--topologies', type=str, default='ring,ws,scale_free')
    p.add_argument('--sizes', type=str, default='30')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--export', type=str, default='results/field_methods_battery.jsonl'
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    topologies = [t.strip() for t in args.topologies.split(',') if t.strip()]
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    out_path = Path(args.export)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as f_out:
        for topo in topologies:
            for n in sizes:
                for run in range(args.runs):
                    seed = args.seed + run + n
                    G = make_graph(topo, n, seed)
                    init_state(G, seed + 99)

                    # Canonical
                    grad_can = compute_phase_gradient(G)
                    curv_can = compute_phase_curvature(G)

                    # Naive
                    grad_nv = naive_phase_gradient(G)
                    curv_nv = naive_phase_curvature(G)

                    cmp_grad = compare_fields(grad_can, grad_nv)
                    cmp_curv = compare_fields(curv_can, curv_nv)

                    rec = {
                        "task": "field_methods_battery",
                        "topology": topo,
                        "n": n,
                        "seed": seed,
                        "cmp_grad": cmp_grad,
                        "cmp_curv": cmp_curv,
                    }
                    f_out.write(json.dumps(rec) + "\n")

    print(f"Field methods battery complete. Output: {out_path}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
