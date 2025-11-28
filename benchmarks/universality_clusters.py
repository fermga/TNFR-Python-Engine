"""Deterministic universality clustering for tetrad timing exponents.

Reads `exponents_summary.json` produced by `tetrad_scaling_exponents.py` and
derives a feature vector per topology using available timing exponents.

Feature vector per topology:
  [exp_phi_s, exp_phase_grad, exp_phase_curv, exp_xi_c, exp_tetrad_snapshot]

Clustering method (deterministic k-means-like with fixed initial centroids):
  1. Collect feature vectors (topologies).
  2. If requested cluster count K >= number of topologies, each topology
     becomes its own cluster (identity partition).
  3. Otherwise initialize centroids as first K topology vectors in sorted
     order of topology name for determinism.
  4. Iterate (max 50 steps or convergence) assigning by Euclidean distance
     and updating centroids to mean of members.
  5. Produce final mapping topology -> cluster_id, with cluster statistics.

All analysis is READ-ONLY; TNFR invariants preserved.

CLI Example:
  python benchmarks/universality_clusters.py \
      --exponents-file results/exponents_summary.json --clusters 2

Output:
  results/universality_clusters.json
  results/universality_clusters.md (table summary)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Any


FEATURE_KEYS = [
    "timing_phi_s",
    "timing_phase_grad",
    "timing_phase_curv",
    "timing_xi_c",
    "timing_tetrad_snapshot",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deterministic clustering of tetrad timing scaling exponents for "
            "universality insight"
        )
    )
    p.add_argument(
        "--exponents-file",
        required=True,
        help="Path to exponents_summary.json",
    )
    p.add_argument(
        "--clusters",
        type=int,
        default=2,
        help="Number of clusters (default: 2)",
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for clustering outputs (default: results)",
    )
    return p.parse_args()


def load_exponents(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_feature_vectors(data: Dict[str, Any]) -> Dict[str, List[float]]:
    metrics = data.get("metrics", {})
    # Build: topology -> list of exponents in FEATURE_KEYS order
    vectors: Dict[str, List[float]] = {}
    for fk in FEATURE_KEYS:
        metric_block = metrics.get(fk, {})
        topologies = metric_block.get("topologies", {})
        for topo, tdata in topologies.items():
            vectors.setdefault(topo, [])
            exp_val = tdata.get("exponent", math.nan)
            vectors[topo].append(exp_val)
    return vectors


def deterministic_kmeans(
    vectors: Dict[str, List[float]], k: int
) -> Dict[str, Any]:
    topo_names = sorted(vectors.keys())
    if k <= 1:
        # Single cluster case
        return {
            "clusters": {0: topo_names},
            "centroids": {0: _mean_vector([vectors[t] for t in topo_names])},
            "assignments": {t: 0 for t in topo_names},
            "iterations": 0,
        }
    if k >= len(topo_names):
        # Identity partition
        clusters = {i: [topo_names[i]] for i in range(len(topo_names))}
        centroids = {i: vectors[topo_names[i]] for i in range(len(topo_names))}
        return {
            "clusters": clusters,
            "centroids": centroids,
            "assignments": {t: i for i, t in enumerate(topo_names)},
            "iterations": 0,
        }
    # Initialize centroids deterministically
    centroids = {i: vectors[topo_names[i]][:] for i in range(k)}
    assignments: Dict[str, int] = {t: -1 for t in topo_names}
    max_iter = 50
    for it in range(max_iter):
        changed = False
        # Assign
        for t in topo_names:
            vec = vectors[t]
            best_cluster = min(
                centroids.keys(),
                key=lambda cid: _euclidean_squared(vec, centroids[cid]),
            )
            if assignments[t] != best_cluster:
                assignments[t] = best_cluster
                changed = True
        # Recompute centroids
        cluster_members: Dict[int, List[List[float]]] = {
            cid: [] for cid in centroids
        }
        for t, cid in assignments.items():
            cluster_members[cid].append(vectors[t])
        for cid, member_vecs in cluster_members.items():
            if member_vecs:
                centroids[cid] = _mean_vector(member_vecs)
        if not changed:
            return {
                "clusters": _invert_assignments(assignments),
                "centroids": centroids,
                "assignments": assignments,
                "iterations": it + 1,
            }
    return {
        "clusters": _invert_assignments(assignments),
        "centroids": centroids,
        "assignments": assignments,
        "iterations": max_iter,
    }


def _invert_assignments(assignments: Dict[str, int]) -> Dict[int, List[str]]:
    inv: Dict[int, List[str]] = {}
    for topo, cid in assignments.items():
        inv.setdefault(cid, []).append(topo)
    return inv


def _mean_vector(vecs: List[List[float]]) -> List[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    acc = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            acc[i] += 0.0 if math.isnan(x) else x
    count = len(vecs)
    return [a / count for a in acc]


def _euclidean_squared(a: List[float], b: List[float]) -> float:
    return sum(((_nz(a[i]) - _nz(b[i])) ** 2) for i in range(len(a)))


def _nz(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def write_outputs(
    result: Dict[str, Any], path_json: str, path_md: str
) -> None:
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    lines: List[str] = []
    lines.append("# Universality Clusters (Timing Exponents)")
    lines.append("")
    lines.append(f"Iterations: {result['iterations']}")
    lines.append("")
    lines.append(
        "| Cluster | Members | Centroid (phi_s, grad, curv, xi_c, snap) |"
    )
    lines.append(
        "|---------|---------|------------------------------------------|"
    )
    for cid, members in sorted(result["clusters"].items()):
        centroid = result["centroids"][cid]
        centroid_fmt = ", ".join(f"{v:.4g}" for v in centroid)
        lines.append(
            f"| {cid} | {', '.join(members)} | {centroid_fmt} |"
        )
    lines.append("")
    with open(path_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    data = load_exponents(args.exponents_file)
    vectors = extract_feature_vectors(data)
    if not vectors:
        print("[ERROR] No topology feature vectors extracted.")
        raise SystemExit(1)
    result = deterministic_kmeans(vectors, args.clusters)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "universality_clusters.json")
    md_path = os.path.join(args.output_dir, "universality_clusters.md")
    write_outputs(result, json_path, md_path)
    print(f"Clusters written: {json_path}")
    print(f"Markdown summary: {md_path}")
    print("Invariants preserved (read-only universality analysis).")


if __name__ == "__main__":  # pragma: no cover
    main()
