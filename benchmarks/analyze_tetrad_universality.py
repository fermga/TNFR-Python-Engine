#!/usr/bin/env python3
"""Tetrad Universality Correlation Analysis

Reads one or more benchmark result files (JSONL or CSV) containing
observations of the structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and
coherence metrics (C(t), Si) and computes per-topology correlations.

The script is intentionally lightweight: it performs passive statistical
aggregation only (READ-ONLY) and never mutates TNFR state.

Output: JSONL summary rows, one per topology, including Pearson
correlations and simple ranking of predictive strength.

Can be extended later with partial correlations or regression models.
"""

from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(y) < 3:
        return float('nan')
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('nan')
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def _extract_records(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in raw:
        # Attempt to drill into known structures
        topo = row.get('topology') or row.get('raw_data', {}).get('topology')
        if topo is None:
            continue

        # Gather tetrad + coherence metrics if present
        rd = row.get('raw_data', {})
        # analysis section may exist but not required here

        # Flatten potential coherence length data
        xi_entries = rd.get('xi_c_data', [])
        xi_means = [e.get('mean', 0.0) for e in xi_entries if 'mean' in e]
        xi_global_mean = float(np.mean(xi_means)) if xi_means else math.nan

        rec = {
            'topology': topo,
            'phi_s_mean': rd.get(
                'phi_s_mean', row.get('phi_s_mean', math.nan)
            ),
            'phase_gradient_mean': rd.get('phase_gradient_mean', math.nan),
            'k_phi_mean': rd.get('k_phi_mean', math.nan),
            'xi_c_mean': xi_global_mean,
            'coherence_mean': rd.get(
                'coherence_mean', row.get('coherence_mean', math.nan)
            ),
            'sense_index_mean': rd.get(
                'sense_index_mean', row.get('sense_index_mean', math.nan)
            ),
        }
        records.append(rec)
    return records


def _group_by_topology(
    records: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r['topology'], []).append(r)
    return groups


def compute_correlations(
    groups: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for topo, recs in groups.items():
        def col(name: str) -> List[float]:
            return [
                r.get(name, math.nan)
                for r in recs
                if not math.isnan(r.get(name, math.nan))
            ]

        phi_s = col('phi_s_mean')
        grad = col('phase_gradient_mean')
        k_phi = col('k_phi_mean')
        xi_c = col('xi_c_mean')
        coh = col('coherence_mean')
        si = col('sense_index_mean')

        corr_phi_coh = _pearson(phi_s, coh)
        corr_grad_coh = _pearson(grad, coh)
        corr_kphi_coh = _pearson(k_phi, coh)
        corr_xic_coh = _pearson(xi_c, coh)
        corr_phi_si = _pearson(phi_s, si)
        corr_xic_si = _pearson(xi_c, si)

        ranking = {
            'coherence': sorted(
                [
                    ('phi_s', corr_phi_coh),
                    ('phase_gradient', corr_grad_coh),
                    ('k_phi', corr_kphi_coh),
                    ('xi_c', corr_xic_coh),
                ],
                key=lambda x: (abs(x[1]) if not math.isnan(x[1]) else -1),
                reverse=True,
            ),
            'sense_index': sorted(
                [
                    ('phi_s', corr_phi_si),
                    ('xi_c', corr_xic_si),
                ],
                key=lambda x: (abs(x[1]) if not math.isnan(x[1]) else -1),
                reverse=True,
            ),
        }

        summaries.append(
            {
                'topology': topo,
                'n_samples': len(recs),
                'corr_phi_s_coherence': corr_phi_coh,
                'corr_phase_gradient_coherence': corr_grad_coh,
                'corr_k_phi_coherence': corr_kphi_coh,
                'corr_xi_c_coherence': corr_xic_coh,
                'corr_phi_s_sense_index': corr_phi_si,
                'corr_xi_c_sense_index': corr_xic_si,
                'predictive_ranking': ranking,
            }
        )

    return summaries


def main():  # CLI entry
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-topology tetrad correlations with coherence metrics"
        )
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input JSONL/CSV files containing benchmark rows"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/results/tetrad_universality_summary.jsonl"
        ),
        help=(
            "Output JSONL path (default: benchmarks/results/"
            "tetrad_universality_summary.jsonl)"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output"
    )
    args = parser.parse_args()

    all_records: List[Dict[str, Any]] = []
    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            if not args.quiet:
                print(f"[WARN] Missing input: {path}")
            continue
        if path.suffix == '.jsonl':
            raw_rows = _load_jsonl(path)
        elif path.suffix == '.csv':
            # Simple CSV loader: expect header row; skip if mismatch
            import csv
            raw_rows = []
            with path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_rows.append(row)
        else:
            if not args.quiet:
                print(f"[WARN] Unsupported extension: {path}")
            continue
        extracted = _extract_records(raw_rows)
        all_records.extend(extracted)

    groups = _group_by_topology(all_records)
    summaries = compute_correlations(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for row in summaries:
            f.write(json.dumps(row) + '\n')

    if not args.quiet:
        print(f"Wrote {len(summaries)} topology summaries to {args.output}")
        for s in summaries:
            print(
                f"{s['topology']:<12} n={s['n_samples']:<4d} "
                f"corr(xi_c, C)={s['corr_xi_c_coherence']:.3f} "
                f"corr(phi_s, C)={s['corr_phi_s_coherence']:.3f}"
            )


if __name__ == '__main__':  # pragma: no cover
    main()
