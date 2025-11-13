"""Summarize Field Methods Battery per topology.

Aggregates correlations and errors between canonical and naive
implementations of |grad(phi)| and K_phi across topologies.

Usage:
    python benchmarks/summarize_field_methods_battery.py \
        --input results/field_methods_battery_large.jsonl \
        --output results/field_methods_battery_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize field methods battery")
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    return p.parse_args(list(argv))


def update(acc: Dict[str, float], key: str, value: float) -> None:
    acc[key + "__sum"] = acc.get(key + "__sum", 0.0) + float(value)
    acc[key + "__sum_sq"] = (
        acc.get(key + "__sum_sq", 0.0) + float(value * value)
    )
    acc[key + "__count"] = acc.get(key + "__count", 0) + 1


def finalize(acc: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = sorted({k.split("__")[0] for k in acc.keys()})
    for k in keys:
        cnt = acc.get(k + "__count", 0)
        if cnt <= 0:
            out[k] = {"mean": 0.0, "std": 0.0}
            continue
        s = acc.get(k + "__sum", 0.0)
        s2 = acc.get(k + "__sum_sq", 0.0)
        mu = s / cnt
        var = max(0.0, s2 / cnt - mu * mu)
        out[k] = {"mean": mu, "std": var ** 0.5}
    return out


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    # Per-topology accumulation
    acc: Dict[str, Dict[str, float]] = defaultdict(dict)

    for r in rows:
        topo = str(r.get('topology', 'unknown'))
        cmp_grad = r.get('cmp_grad', {})
        cmp_curv = r.get('cmp_curv', {})
        for key, val in cmp_grad.items():
            update(acc.setdefault(topo, {}), 'grad_' + key, float(val))
        for key, val in cmp_curv.items():
            update(acc.setdefault(topo, {}), 'curv_' + key, float(val))

    summary: Dict[str, Any] = {}
    for topo, metrics in acc.items():
        summary[topo] = finalize(metrics)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {out_path}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
