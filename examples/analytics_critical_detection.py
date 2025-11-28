"""Critical Regime Detection Demo

Loads raw benchmark JSONL files, computes structural risk scores using the
Phase 5 critical regime detector output, and summarizes highest-risk runs.

If a precomputed risk JSONL file exists (default path), it is loaded directly.
Otherwise, the script will instruct how to generate it.

Usage:
  python examples/analytics_critical_detection.py \
     --risk-jsonl results/critical_regime_risk.jsonl \
     --top-n 10
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

RISK_KEYS = [
    "risk_score",
    "norm_xi",
    "phase_grad_volatility",
    "curvature_hotspot_ratio",
]


def parse_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def sort_by_risk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: r.get("risk_score", 0.0),
        reverse=True,
    )


def render_table(rows: List[Dict[str, Any]]) -> str:
    header = ["topology", "n_nodes", *RISK_KEYS]
    lines = [" | ".join(header)]
    lines.append(" | ".join(["---"] * len(header)))
    for r in rows:
        row = [
            str(r.get("topology", "?")),
            str(r.get("n_nodes", "?")),
        ]
        for k in RISK_KEYS:
            v = r.get(k)
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        lines.append(" | ".join(row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--risk-jsonl",
        default="results/critical_regime_risk.jsonl",
        help="Path to risk detector JSONL output",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of highest-risk rows to display",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.risk_jsonl):
        print(
            f"[MISSING] {args.risk_jsonl}\n"
            "Run: python benchmarks/critical_regime_detector.py "
            "--glob \"results/tetrad_scaling_*.jsonl\" "
            "--out results/critical_regime_risk.jsonl"
        )
        return

    rows = parse_jsonl(args.risk_jsonl)
    if not rows:
        print("[ERROR] No rows parsed from risk JSONL.")
        return

    sorted_rows = sort_by_risk(rows)[: args.top_n]
    print("\n== Highest Structural Risk Runs ==")
    print(render_table(sorted_rows))
    print("\n[OK] Observational only; TNFR physics unchanged.")


if __name__ == "__main__":  # pragma: no cover
    main()
