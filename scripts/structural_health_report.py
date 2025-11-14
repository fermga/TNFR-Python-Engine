"""CLI script: Generate TNFR structural health report.

Reads a TNFR graph produced by a simulation (optional) and an operator
sequence (optional) then prints a concise structural health summary.

Usage examples:
---------------
python scripts/structural_health_report.py \
    --graph examples/output/graph.pkl \
    --sequence examples/output/sequence.txt \
    --json results/reports/health_report.json

python scripts/structural_health_report.py --random 32 --edge-prob 0.15

Inputs
------
Graph formats supported:
 - Pickle (NetworkX graph)
 - Edge list (.edgelist) simple whitespace separated pairs

Sequence file: one operator mnemonic per line (e.g. AL, UM, IL, SHA).

Outputs
-------
STDOUT: Human-readable summary
JSON (optional): Machine-readable payload

All computations are telemetry-only; graph is never mutated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

try:  # networkx dependency
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

from tnfr.validation.health import compute_structural_health


def _load_graph(path: Path):
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx not available")
    if path.suffix == ".pkl":
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)
    if path.suffix == ".edgelist":
        G = nx.read_edgelist(path)  # type: ignore
        return G
    raise ValueError(f"Unsupported graph format: {path.suffix}")


def _load_sequence(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TNFR structural health report")
    p.add_argument(
        "--graph",
        type=Path,
        help="Graph pickle (.pkl) or edge list (.edgelist)",
        required=False,
    )
    p.add_argument(
        "--sequence",
        type=Path,
        help="Operator sequence file (one mnemonic per line)",
        required=False,
    )
    p.add_argument(
        "--json",
        type=Path,
        help="Optional JSON output path",
        required=False,
    )
    p.add_argument(
        "--random",
        type=int,
        help="Generate random Erdos-Renyi graph with N nodes",
    )
    p.add_argument(
        "--edge-prob",
        type=float,
        default=0.1,
        help="Probability for random graph edges",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if nx is None:  # pragma: no cover
        print("networkx required for health report", file=sys.stderr)
        return 2

    if args.graph and args.random:
        print("Specify either --graph or --random, not both", file=sys.stderr)
        return 2

    if args.random:
        nx.random.seed(args.seed)  # type: ignore[attr-defined]
        G = nx.erdos_renyi_graph(args.random, args.edge_prob)  # type: ignore
    elif args.graph:
        G = _load_graph(args.graph)
    else:
        print("Must supply --graph or --random", file=sys.stderr)
        return 2

    sequence = _load_sequence(args.sequence) if args.sequence else None

    health = compute_structural_health(G, sequence=sequence)

    # Human summary
    print("TNFR Structural Health Report")
    print("--------------------------------")
    print(f"Status            : {health['status']}")
    print(f"Risk Level        : {health['risk_level']}")
    subset = health["field_metrics_subset"]
    if subset["mean_phi_s"] is not None:
        print(f"Mean Φ_s          : {subset['mean_phi_s']:.4f}")
    else:
        print("Mean Φ_s          : NA")
    if subset["max_phase_gradient"] is not None:
        print(f"Max |∇φ|          : {subset['max_phase_gradient']:.4f}")
    else:
        print("Max |∇φ|          : NA")
    if subset["max_k_phi"] is not None:
        print(f"Max |K_φ|         : {subset['max_k_phi']:.4f}")
    else:
        print("Max |K_φ|         : NA")
    if subset["xi_c"] is not None:
        print(f"ξ_C               : {subset['xi_c']:.2f}")
    else:
        print("ξ_C               : NA")
    if subset["delta_phi_s"] is not None:
        print(f"ΔΦ_s drift        : {subset['delta_phi_s']:.4f}")
    print("Threshold Flags   :")
    for k, v in health["thresholds_exceeded"].items():
        print(f"  - {k}: {'EXCEEDED' if v else 'ok'}")
    if health["recommended_actions"]:
        print("Recommended Actions:")
        for act in health["recommended_actions"]:
            print(f"  * {act}")
    if health["notes"]:
        print("Notes:")
        for n in health["notes"]:
            print(f"  - {n}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(health, indent=2))
        print(f"JSON report written to {args.json}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
