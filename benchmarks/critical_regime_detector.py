"""Critical regime risk detector (Phase 5 Task 3).

Computes a READ-ONLY structural risk score per benchmark record combining:
    1. ξ_C divergence proxy: normalized coherence length (log-saturated)
  2. Phase gradient volatility: phase_grad_std / (|phase_grad_mean| + eps)
  3. Curvature hotspot proxy:
      min(1, |phase_curv_std| / (|phase_curv_mean| + eps))

All inputs come from previously generated benchmark outputs (JSONL or
aggregated CSV). No simulation or operator application occurs; TNFR
invariants preserved (pure telemetry transformation).

Risk score formula (log-saturated ξ_C):
    raw_norm_xi = xi_c / n_nodes
    norm_xi = min(1, log(1 + xi_c) / log(1 + n_nodes))  # saturation
    risk_raw = w_xi * norm_xi + w_pg * pg_vol + w_curv * curv_proxy
    risk = min(1.0, risk_raw / (w_xi + w_pg + w_curv))

Default weights (can be overridden):
    w_xi = 0.5, w_pg = 0.3, w_curv = 0.2

Critical flag if risk >= threshold (default 0.7).

Output: JSONL file with one line per record including risk metrics.

Example CLI usage:
  python benchmarks/critical_regime_detector.py \
      --jsonl-glob results/tetrad_scaling_*.jsonl

  python benchmarks/critical_regime_detector.py \
      --csv results/tetrad_scaling_aggregate_20251115_*.csv \
      --threshold 0.65 --w-xi 0.6 --w-pg 0.25 --w-curv 0.15

"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import datetime as _dt
from typing import List, Dict, Any


CSV_REQUIRED = {
    "topology",
    "n_nodes",
    "seed",
    "phase_grad_mean",
    "phase_grad_std",
    "phase_curv_mean",
    "phase_curv_std",
    "xi_c",
}

JSONL_REQUIRED = CSV_REQUIRED  # identical set after flattening


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute structural critical regime risk scores from benchmark "
            "data"
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--jsonl-glob",
        help="Glob pattern for raw benchmark JSONL files",
    )
    g.add_argument(
        "--csv",
        nargs="+",
        help="Aggregated CSV file(s) (shell globs expanded by shell)",
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for risk output JSONL (default: results)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Critical flag threshold (default: 0.7)",
    )
    p.add_argument(
        "--w-xi", type=float, default=0.5, help="Weight for ξ_C component"
    )
    p.add_argument(
        "--w-pg", type=float, default=0.3, help="Weight phase gradient"
    )
    p.add_argument(
        "--w-curv", type=float, default=0.2, help="Weight curvature proxy"
    )
    p.add_argument(
        "--xi-norm",
        choices=["raw", "log-sat"],
        default="log-sat",
        help=(
            "Normalization for xi_c: raw ratio or log-saturated "
            "(default: log-sat)"
        ),
    )
    return p.parse_args()


def load_csv(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            items.append(r)
    return items


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            flat = flatten_jsonl_record(rec)
            out.append(flat)
    return out


def flatten_jsonl_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    tv = rec.get("tetrad_values", {})
    return {
        "topology": rec.get("topology"),
        "n_nodes": rec.get("n_nodes"),
        "seed": rec.get("seed"),
        "phase_grad_mean": tv.get("phase_grad_mean"),
        "phase_grad_std": tv.get("phase_grad_std"),
        "phase_curv_mean": tv.get("phase_curv_mean"),
        "phase_curv_std": tv.get("phase_curv_std"),
        "xi_c": tv.get("xi_c"),
    }


def filter_valid(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for r in records:
        if not JSONL_REQUIRED.issubset(r.keys()):
            continue
        if r.get("n_nodes") in (None, ""):
            continue
        valid.append(r)
    return valid


def compute_risk(
    record: Dict[str, Any], w_xi: float, w_pg: float, w_curv: float,
    norm_mode: str
) -> Dict[str, Any]:
    eps = 1e-12
    try:
        n_nodes = float(record["n_nodes"])
    except Exception:
        n_nodes = float("nan")
    xi_c = record.get("xi_c")
    raw_norm_xi = 0.0
    norm_xi = 0.0
    if xi_c not in (None, ""):
        try:
            xi_val = float(xi_c)
            import math
            raw_norm_xi = xi_val / max(n_nodes, 1.0)
            if norm_mode == "log-sat":
                if n_nodes > 0 and xi_val > 0:
                    norm_xi = math.log1p(xi_val) / math.log1p(n_nodes)
                    if norm_xi > 1.0:
                        norm_xi = 1.0
                else:
                    norm_xi = 0.0
            else:  # raw
                norm_xi = raw_norm_xi
        except Exception:
            raw_norm_xi = 0.0
            norm_xi = 0.0
    # Phase gradient volatility
    try:
        pg_mean = abs(float(record.get("phase_grad_mean", 0.0)))
        pg_std = float(record.get("phase_grad_std", 0.0))
        pg_vol = pg_std / (pg_mean + eps)
    except Exception:
        pg_vol = 0.0
    # Curvature hotspot proxy
    try:
        curv_mean = abs(float(record.get("phase_curv_mean", 0.0)))
        curv_std = abs(float(record.get("phase_curv_std", 0.0)))
        curv_proxy = min(1.0, curv_std / (curv_mean + eps))
    except Exception:
        curv_proxy = 0.0
    risk_raw = w_xi * norm_xi + w_pg * pg_vol + w_curv * curv_proxy
    denom = w_xi + w_pg + w_curv
    risk = min(1.0, risk_raw / denom if denom > 0 else 0.0)
    risk_version = (
        "log_sat_xi_v1" if norm_mode == "log-sat" else "raw_xi_v1"
    )
    return {
        "norm_xi": norm_xi,
        "norm_xi_raw": raw_norm_xi,
        "phase_grad_volatility": pg_vol,
        "curvature_hotspot_proxy": curv_proxy,
        "risk": risk,
        "risk_version": risk_version,
        "norm_mode": norm_mode,
    }


def main() -> None:
    args = parse_args()
    if args.jsonl_glob:
        files = glob.glob(args.jsonl_glob)
        if not files:
            print("[ERROR] No JSONL files matched glob.")
            sys.exit(1)
        records: List[Dict[str, Any]] = []
        for f in files:
            records.extend(load_jsonl(f))
    else:
        csv_files: List[str] = []
        for pat in args.csv:
            if os.path.isfile(pat):
                csv_files.append(pat)
            else:
                csv_files.extend(glob.glob(pat))
        if not csv_files:
            print("[ERROR] No CSV inputs found.")
            sys.exit(1)
        records = []
        for f in csv_files:
            records.extend(load_csv(f))

    valid = filter_valid(records)
    if not valid:
        print("[ERROR] No valid records for risk computation.")
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir, f"critical_regime_risk_{timestamp}.jsonl"
    )
    flagged = 0
    total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in valid:
            metrics = compute_risk(
                r, args.w_xi, args.w_pg, args.w_curv, args.xi_norm
            )
            entry = {
                "topology": r.get("topology"),
                "n_nodes": r.get("n_nodes"),
                "seed": r.get("seed"),
                **metrics,
                "threshold": args.threshold,
                "critical": metrics["risk"] >= args.threshold,
            }
            if entry["critical"]:
                flagged += 1
            total += 1
            fh.write(json.dumps(entry) + "\n")

    print(f"Risk file: {out_path}")
    print(f"Records processed: {total}, critical flagged: {flagged}")
    print("Invariants preserved (telemetry-only risk assessment).")


if __name__ == "__main__":  # pragma: no cover
    main()
