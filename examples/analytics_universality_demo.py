"""Universality Analytics Demo

Reads exponent summary and clustering outputs produced by Phase 5 analytics
scripts and prints a concise universality report. Purely observational: does
not modify TNFR dynamics (invariants preserved).

Usage (after running benchmarks + analytics):
  python examples/analytics_universality_demo.py \
    --exponents results/exponents_summary.json \
    --clusters results/universality_clusters.json

If files are missing, the script will instruct how to generate them.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

METRICS = [
    "timing_phi_s",
    "timing_phase_grad",
    "timing_phase_curv",
    "timing_xi_c",
    "timing_tetrad_snapshot",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_exponents(
    exponents: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metrics = exponents.get("metrics", {})
    for m in METRICS:
        mdata = metrics.get(m)
        if not mdata:
            continue
        topo_data = mdata.get("topologies", {})
        for topo, stats in topo_data.items():
            out.setdefault(topo, {})[m] = stats.get("exponent")
    return out


def render_universality_table(
    exponents_map: Dict[str, Dict[str, float]]
) -> str:
    # Collect rows with metric exponents
    header = ["topology"] + METRICS
    lines = [" | ".join(header)]
    lines.append(" | ".join(["---"] * len(header)))
    for topo in sorted(exponents_map.keys()):
        row = [topo]
        for m in METRICS:
            val = exponents_map[topo].get(m)
            # NaN check via val != val idiom
            if val is None or (isinstance(val, float) and (val != val)):
                row.append("NA")
            else:
                row.append(f"{val:.3f}")
        lines.append(" | ".join(row))
    return "\n".join(lines)


def summarize_clusters(clusters: Dict[str, Any]) -> str:
    assignments = clusters.get("assignments", {})
    centroids = clusters.get("centroids", [])
    # Reverse map cluster -> topologies
    by_cluster: Dict[int, List[str]] = {}
    for topo, cid in assignments.items():
        by_cluster.setdefault(int(cid), []).append(topo)
    lines = []
    lines.append("Clusters (deterministic):")
    for cid in sorted(by_cluster.keys()):
        members = ", ".join(sorted(by_cluster[cid]))
        cvec = centroids[cid] if cid < len(centroids) else []
        centroid_fmt = ["{:.3f}".format(x) for x in cvec]
        lines.append(
            f"  Cluster {cid}: [{members}] :: centroids={centroid_fmt}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exponents", default="results/exponents_summary.json")
    ap.add_argument("--clusters", default="results/universality_clusters.json")
    args = ap.parse_args()

    if not os.path.isfile(args.exponents):
        print(
            f"[MISSING] {args.exponents}\n"
            "Run: python benchmarks/tetrad_scaling_exponents.py "
            "--csv <aggregate.csv> --json-out results/exponents_summary.json"
        )
        return
    if not os.path.isfile(args.clusters):
        print(
            f"[MISSING] {args.clusters}\n"
            "Run: python benchmarks/universality_clusters.py "
            "--exponents-json results/exponents_summary.json "
            "--clusters-out results/universality_clusters.json"
        )
        return

    exponents = load_json(args.exponents)
    clusters = load_json(args.clusters)

    exponents_map = summarize_exponents(exponents)
    print("\n== Universality Timing Exponents ==")
    print(render_universality_table(exponents_map))

    print("\n== Cluster Assignments ==")
    print(summarize_clusters(clusters))

    print("\n[OK] Observational only; TNFR physics unchanged.")


if __name__ == "__main__":  # pragma: no cover
    main()
