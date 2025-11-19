"""GPU vs CPU exponent timing comparison (optional Phase 5 extension).

Measures wall-clock time to compute exponent summary (log-log scaling fits)
using CPU (NumPy) and optionally GPU (CuPy) backends. Purely observational;
outputs JSON with timing metrics and exponents for cross-hardware comparison.

If CuPy is unavailable or no CUDA device is detected, GPU results are skipped.

Usage:
  python benchmarks/gpu_cpu_exponent_timing_compare.py \
      --csv results/tetrad_scaling_aggregate_phase5.csv \
      --out results/gpu_cpu_exponent_timing.json

Optional:
  python benchmarks/gpu_cpu_exponent_timing_compare.py --backend both

Physics alignment: Does not alter TNFR dynamics; only re-computes summaries.
Invariants preserved (read-only analytics).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import os
from typing import Dict, Any, List

import csv

try:
    import cupy as cp  # type: ignore
    _CUPY = True
except Exception:  # pragma: no cover
    _CUPY = False

# Columns required for exponent summary
TIMING_COLUMNS = [
    "timing_phi_s",
    "timing_phase_grad",
    "timing_phase_curv",
    "timing_xi_c",
    "timing_tetrad_snapshot",
]
REQUIRED = {"topology", "n_nodes", "seed", *TIMING_COLUMNS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GPU vs CPU timing for exponent summary")
    p.add_argument("--csv", required=True, help="Aggregated CSV input")
    p.add_argument("--backend", choices=["cpu", "gpu", "both"], default="both")
    p.add_argument("--out", default="results/gpu_cpu_exponent_timing.json")
    return p.parse_args()


def load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if REQUIRED.issubset(r.keys()):
                rows.append(r)
    return rows


def group(
    rows: List[Dict[str, Any]]
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for r in rows:
        topo = r["topology"]
        n = int(r["n_nodes"])
        out.setdefault(topo, {}).setdefault(n, []).append(r)
    return out


def fit_exponents_cpu(groups) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for topo, by_n in groups.items():
        metrics_info: Dict[str, Any] = {}
        node_sizes = sorted(by_n.keys())
        for col in TIMING_COLUMNS:
            means = []
            for n in node_sizes:
                vals = [float(r[col]) for r in by_n[n]]
                means.append(sum(vals) / len(vals))
            if len(node_sizes) >= 2:
                pairs = [
                    (math.log(float(n)), math.log(m))
                    for n, m in zip(node_sizes, means)
                    if m > 0
                ]
                if len(pairs) >= 2:
                    xs, ys = zip(*pairs)
                    mean_x = sum(xs) / len(xs)
                    mean_y = sum(ys) / len(ys)
                    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
                    den = sum((x - mean_x) ** 2 for x in xs)
                    slope = num / den if den else float("nan")
                else:
                    slope = float("nan")
            else:
                slope = float("nan")
            metrics_info[col] = {"exponent": slope}
        result[topo] = metrics_info
    return result


def fit_exponents_gpu(groups) -> Dict[str, Dict[str, Any]]:
    if not _CUPY:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for topo, by_n in groups.items():
        metrics_info: Dict[str, Any] = {}
        node_sizes = sorted(by_n.keys())
        for col in TIMING_COLUMNS:
            means = []
            for n in node_sizes:
                vals = [float(r[col]) for r in by_n[n]]
                means.append(sum(vals) / len(vals))
            if len(node_sizes) >= 2:
                xs = cp.asarray(node_sizes, dtype=cp.float64)
                ys = cp.asarray(means, dtype=cp.float64)
                mask = ys > 0
                xs = cp.log(xs[mask])
                ys = cp.log(ys[mask])
                if xs.size >= 2:
                    mean_x = cp.mean(xs)
                    mean_y = cp.mean(ys)
                    num = cp.sum((xs - mean_x) * (ys - mean_y))
                    den = cp.sum((xs - mean_x) ** 2)
                    slope = float(num / den) if den != 0 else float("nan")
                else:
                    slope = float("nan")
            else:
                slope = float("nan")
            metrics_info[col] = {"exponent": slope}
        result[topo] = metrics_info
    return result


def main() -> None:
    args = parse_args()
    rows = load_csv(args.csv)
    if not rows:
        print("[ERROR] No valid rows in CSV")
        return
    groups = group(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output: Dict[str, Any] = {"cpu": None, "gpu": None}
    if args.backend in ("cpu", "both"):
        t0 = time.perf_counter()
        cpu_res = fit_exponents_cpu(groups)
        cpu_time = time.perf_counter() - t0
        output["cpu"] = {"timing": cpu_time, "exponents": cpu_res}
    if args.backend in ("gpu", "both") and _CUPY:
        t0 = time.perf_counter()
        gpu_res = fit_exponents_gpu(groups)
        gpu_time = time.perf_counter() - t0
        output["gpu"] = {
            "timing": gpu_time,
            "exponents": gpu_res,
            "backend": "cupy",
        }
    elif args.backend in ("gpu", "both") and not _CUPY:
        output["gpu"] = {"error": "CuPy not available"}
    output["invariants"] = "read-only comparison"
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results written to {args.out}")
    if output.get("cpu"):
        print(f"CPU time: {output['cpu']['timing']:.6f}s")
    if output.get("gpu") and output["gpu"] and "timing" in output["gpu"]:
        print(f"GPU time: {output['gpu']['timing']:.6f}s")


if __name__ == "__main__":  # pragma: no cover
    main()
