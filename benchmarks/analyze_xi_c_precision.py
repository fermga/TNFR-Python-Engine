#!/usr/bin/env python3
"""ξ_C Precision Mode Comparative Analysis

Reads coherence length critical exponent JSONL outputs generated under
different precision modes (standard/high/research) and computes summary
statistics for ξ_C means and ranges, verifying qualitative consistency
with TNFR invariants (bounded drift, preserved ordering).

The script is passive (READ-ONLY) and provides a condensed table plus
JSONL output for downstream aggregation.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_precision_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for r in rows:
        precision = r.get('precision_mode') or r.get('precision')
        analysis = r.get('analysis', {})
        xi_entries = r.get('raw_data', {}).get('xi_c_data', [])
        xi_means = [e.get('mean', np.nan) for e in xi_entries]
        xi_mean_global = (
            float(np.nanmean(xi_means)) if xi_means else float('nan')
        )
        extracted.append(
            {
                'precision_mode': precision or 'unknown',
                'topology': r.get('topology'),
                'n_nodes': r.get('n_nodes'),
                'xi_c_global_mean': xi_mean_global,
                'xi_c_min': (
                    float(np.nanmin(xi_means)) if xi_means else float('nan')
                ),
                'xi_c_max': (
                    float(np.nanmax(xi_means)) if xi_means else float('nan')
                ),
                'n_samples': len(xi_means),
                'exponent_fit_success': analysis.get('success'),
            }
        )
    return extracted


def group_by_precision(
    rows: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r['precision_mode'], []).append(r)
    return groups


def summarize(groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for mode, recs in groups.items():
        means = [
            r['xi_c_global_mean']
            for r in recs
            if not np.isnan(r['xi_c_global_mean'])
        ]
        if means:
            drift_range = float(max(means) - min(means))
            mean_mean = float(np.mean(means))
            std_mean = float(np.std(means))
        else:
            drift_range = float('nan')
            mean_mean = float('nan')
            std_mean = float('nan')
        summaries.append(
            {
                'precision_mode': mode,
                'n_experiments': len(recs),
                'xi_c_mean_mean': mean_mean,
                'xi_c_mean_std': std_mean,
                'xi_c_mean_range': drift_range,
                'qualitative_consistency': (
                    True if drift_range < 0.05 * (mean_mean + 1e-9) else False
                ),
            }
        )
    return summaries


def print_table(summaries: List[Dict[str, Any]]):  # simple textual table
    headers = [
        'precision_mode', 'n_experiments', 'xi_c_mean_mean', 'xi_c_mean_std',
        'xi_c_mean_range', 'qualitative_consistency'
    ]
    col_widths = {h: max(len(h), 14) for h in headers}
    for row in summaries:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))
    line = ' '.join(f"{h:<{col_widths[h]}}" for h in headers)
    print(line)
    print('-' * len(line))
    for row in summaries:
        print(' '.join(f"{str(row[h]):<{col_widths[h]}}" for h in headers))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ξ_C statistics across precision modes"
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input JSONL files from coherence length exponent benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/xi_c_precision_summary.jsonl"),
        help="Output JSONL summary path"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress table printout"
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            if not args.quiet:
                print(f"[WARN] Missing input: {path}")
            continue
        raw = _load_jsonl(path)
        ext = extract_precision_rows(raw)
        all_rows.extend(ext)

    groups = group_by_precision(all_rows)
    summaries = summarize(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for row in summaries:
            f.write(json.dumps(row) + '\n')

    if not args.quiet:
        print_table(summaries)
        print(f"Saved summary: {args.output}")


if __name__ == '__main__':  # pragma: no cover
    main()
