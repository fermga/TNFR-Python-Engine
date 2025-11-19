"""Fit log-log scaling exponents for Structural Field Tetrad timings.

Reads aggregated CSV (from tetrad_results_aggregate.py) OR raw JSONL glob.
Purely observational: does not alter TNFR dynamics (invariants preserved).

For each topology and timing metric (timing_phi_s, timing_phase_grad,
timing_phase_curv, timing_xi_c, timing_tetrad_snapshot) we:
  1. Group rows by (topology, n_nodes)
  2. Compute mean and std over seeds
  3. Perform linear regression on log(mean) vs log(n_nodes):
       log(T) = a * log(N) + b  =>  T ≈ N^a * exp(b)
  4. Compute R^2 for fit quality

Outputs:
  JSON summary: results/exponents_summary.json
  Markdown table: results/exponents_summary.md

CLI Example:
  python benchmarks/tetrad_scaling_exponents.py \
    --csv results/tetrad_scaling_aggregate_*.csv

If multiple CSV files match, all are merged.
Alternatively:
  python benchmarks/tetrad_scaling_exponents.py \
    --jsonl-glob results/tetrad_scaling_*.jsonl

Requirements: numpy only (scipy optional but not required).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
# Removed time-based stamp; use deterministic hash for reproducibility
import hashlib
from typing import Dict, List, Any, Tuple

try:  # optional SciPy (placeholder for future CI intervals)
    import scipy  # type: ignore  # noqa: F401
    _SCIPY = True
except Exception:  # pragma: no cover
    _SCIPY = False


TIMING_COLUMNS = [
    "timing_phi_s",
    "timing_phase_grad",
    "timing_phase_curv",
    "timing_xi_c",
    "timing_tetrad_snapshot",
]

REQUIRED_COLUMNS = {
    "topology",
    "n_nodes",
    "seed",
    *TIMING_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit log-log scaling exponents for tetrad timing metrics"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--csv",
        nargs="+",
        help="Aggregated CSV file(s) (supports glob expansion by shell)",
    )
    g.add_argument(
        "--jsonl-glob",
        help="Glob for raw JSONL benchmark outputs",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum distinct node sizes required for a fit (default: 4)",
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for exponent summaries (default: results)",
    )
    return p.parse_args()


def load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            flat = _flatten_jsonl_record(d)
            out.append(flat)
    return out


def _flatten_jsonl_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    timings = rec.get("timings", {})
    return {
        "topology": rec.get("topology"),
        "n_nodes": rec.get("n_nodes"),
        "seed": rec.get("seed"),
        "timing_phi_s": timings.get("phi_s"),
        "timing_phase_grad": timings.get("phase_grad"),
        "timing_phase_curv": timings.get("phase_curv"),
        "timing_xi_c": timings.get("xi_c"),
        "timing_tetrad_snapshot": timings.get("tetrad_snapshot"),
    }


def filter_valid(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for r in records:
        if not REQUIRED_COLUMNS.issubset(r.keys()):
            continue
        try:
            int(r["n_nodes"])
        except Exception:
            continue
        # Convert timing to float, skip if any missing
        missing = False
        for c in TIMING_COLUMNS:
            if r.get(c) in (None, ""):
                missing = True
                break
        if missing:
            continue
        valid.append(r)
    return valid


def group_by_topology(
    records: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        topo = str(r["topology"])
        groups.setdefault(topo, []).append(r)
    return groups


def group_by_nodes(
    records: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    g: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        n = int(r["n_nodes"])
        g.setdefault(n, []).append(r)
    return g


def compute_means_stds(
    group: List[Dict[str, Any]]
) -> Dict[str, Tuple[float, float]]:
    # group is list of records with same n_nodes
    acc: Dict[str, List[float]] = {c: [] for c in TIMING_COLUMNS}
    for r in group:
        for c in TIMING_COLUMNS:
            try:
                acc[c].append(float(r[c]))
            except Exception:
                pass
    out: Dict[str, Tuple[float, float]] = {}
    for c, arr in acc.items():
        if not arr:
            out[c] = (math.nan, math.nan)
        else:
            mean = sum(arr) / len(arr)
            if len(arr) > 1:
                var = sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            out[c] = (mean, std)
    return out


def log_log_fit(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    # Returns (slope, intercept, r2). Ignores non-positive y.
    pairs = [(math.log(xi), math.log(yi)) for xi, yi in zip(x, y) if yi > 0]
    if len(pairs) < 2:
        return math.nan, math.nan, math.nan
    xs, ys = zip(*pairs)
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return math.nan, math.nan, math.nan
    slope = num / den
    intercept = mean_y - slope * mean_x
    # R^2
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pairs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return slope, intercept, r2


def build_summary(
    records: List[Dict[str, Any]], min_points: int
) -> Dict[str, Any]:
    topo_groups = group_by_topology(records)
    # Deterministic timestamp surrogate: stable hash of input records.
    # Ensures reproducibility (same input -> identical summary) for tests.
    _records_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()[:16]
    summary: Dict[str, Any] = {
        "generated_at": f"hash:{_records_hash}",
        "metrics": {},
    }
    for metric in TIMING_COLUMNS:
        metric_data: Dict[str, Any] = {"topologies": {}, "units": "seconds"}
        for topo, t_records in topo_groups.items():
            by_nodes = group_by_nodes(t_records)
            node_sizes = sorted(by_nodes.keys())
            means: List[float] = []
            stds: List[float] = []
            for n in node_sizes:
                stat = compute_means_stds(by_nodes[n])
                mean, std = stat[metric]
                means.append(mean)
                stds.append(std)
            if len(node_sizes) >= min_points:
                slope, intercept, r2 = log_log_fit(
                    [float(n) for n in node_sizes], means
                )
            else:
                slope = intercept = r2 = math.nan
            metric_data["topologies"][topo] = {
                "n_nodes": node_sizes,
                "means": means,
                "stds": stds,
                "exponent": slope,
                "intercept": intercept,
                "r2": r2,
                "point_count": len(node_sizes),
            }
        summary["metrics"][metric] = metric_data
    return summary


def write_json(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def write_markdown(summary: Dict[str, Any], path: str) -> None:
    lines: List[str] = []
    lines.append("# Tetrad Timing Scaling Exponents")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    for metric, data in summary["metrics"].items():
        lines.append(f"## {metric}")
        lines.append("")
        lines.append(
            "| Topology | Points | Exponent (a) | Intercept (b) | R^2 |"
        )
        lines.append(
            "|----------|--------|--------------|---------------|-----|"
        )
        for topo, tdata in data["topologies"].items():
            a = tdata["exponent"]
            b = tdata["intercept"]
            r2 = tdata["r2"]
            pc = tdata["point_count"]
            lines.append(
                f"| {topo} | {pc} | {a:.5g} | {b:.5g} | {r2:.5g} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.csv:
        # Expand shell globs already resolved; filter existing files
        csv_files = []
        for pat in args.csv:
            if os.path.isfile(pat):
                csv_files.append(pat)
            else:
                csv_files.extend(glob.glob(pat))
        if not csv_files:
            print("[ERROR] No CSV inputs found.")
            raise SystemExit(1)
        all_rows: List[Dict[str, Any]] = []
        for path in csv_files:
            all_rows.extend(load_csv(path))
    else:
        jsonl_files = glob.glob(args.jsonl_glob)
        if not jsonl_files:
            print("[ERROR] No JSONL inputs found.")
            raise SystemExit(1)
        all_rows = []
        for path in jsonl_files:
            all_rows.extend(load_jsonl(path))

    valid = filter_valid(all_rows)
    if not valid:
        print("[ERROR] No valid records after filtering.")
        raise SystemExit(2)

    summary = build_summary(valid, args.min_points)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "exponents_summary.json")
    md_path = os.path.join(args.output_dir, "exponents_summary.md")
    write_json(summary, json_path)
    write_markdown(summary, md_path)

    print(f"Summary written: {json_path}")
    print(f"Markdown table: {md_path}")
    print("Read-only analysis complete. Invariants preserved.")


if __name__ == "__main__":  # pragma: no cover
    main()
