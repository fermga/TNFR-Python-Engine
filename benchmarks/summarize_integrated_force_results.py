"""Summarize Integrated Force Regime Study results per topology.

Reads one or more JSONL files from the integrated study and aggregates
per-topology metrics:
- Task 3: regime distribution (intensity sweep)
- Task 2: mean/std of S_local and S_global
- Task 1: average selected correlations
- Task 5: per-operator average deltas for fields
- Task counts for visibility

Usage:
    python benchmarks/summarize_integrated_force_results.py \
        --inputs results/integrated_force_study_battery.jsonl \
        --output results/integrated_force_study_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize integrated force results"
    )
    p.add_argument('--inputs', type=str, required=True,
                   help='Comma-separated list of JSONL files')
    p.add_argument('--output', type=str, required=True,
                   help='Path to write JSON summary')
    return p.parse_args(list(argv))


def update_mean_std(acc: Dict[str, Any], key: str, value: float) -> None:
    s = acc.setdefault(key, {"count": 0, "sum": 0.0, "sum_sq": 0.0})
    s["count"] += 1
    s["sum"] += float(value)
    s["sum_sq"] += float(value * value)


def finalize_mean_std(s: Dict[str, Any]) -> Dict[str, float]:
    cnt = s.get("count", 0)
    if cnt <= 0:
        return {"mean": 0.0, "std": 0.0}
    mu = s["sum"] / cnt
    var = max(0.0, s["sum_sq"] / cnt - mu * mu)
    return {"mean": mu, "std": var ** 0.5}


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    inputs = [p.strip() for p in args.inputs.split(',') if p.strip()]

    rows: List[Dict[str, Any]] = []
    for path in inputs:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

    # Aggregation structures
    summary: Dict[str, Any] = {}
    task_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # Task 2 stats (S_local, S_global)
    stats_t2: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Task 3: regime distribution
    regimes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Task 1: selected correlations (averages)
    corr_keys = [
        "grad_phi__abs_k_phi",
        "phi_s__coh_local",
        "dnfr__coh_local",
        "grad_phi__dnfr",
    ]
    corr_sums: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    corr_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # Task 5: operator-field coupling averages per operator
    ops = [
        'emission', 'coherence', 'dissonance', 'mutation', 'expansion',
        'silence'
    ]
    deltas_keys = ["d_phi_s", "d_grad_phi", "d_abs_k_phi_max", "d_xi_c"]
    op_sums: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    op_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # Pass 1: accumulate
    for r in rows:
        topo = str(r.get("topology", "unknown"))
        task = str(r.get("task", "unknown"))
        task_counts[topo][task] += 1

        if task == 'composite_field_metrics':  # Task 2
            update_mean_std(
                stats_t2[topo], 'S_local', float(r.get('S_local', 0.0))
            )
            update_mean_std(
                stats_t2[topo], 'S_global', float(r.get('S_global', 0.0))
            )

        elif task == 'force_regime_phase_diagram':  # Task 3
            regime = str(r.get('regime', 'unknown'))
            regimes[topo][regime] += 1

        elif task == 'field_interaction_matrix':  # Task 1
            corr = r.get('corr', {})
            for ck in corr_keys:
                if ck in corr:
                    corr_sums[topo][ck] += float(corr[ck])
                    corr_counts[topo][ck] += 1

        elif task == 'operator_field_coupling':  # Task 5
            op = str(r.get('operator', 'unknown'))
            if op in ops:
                md = r.get('mean_deltas', {})
                for dk in deltas_keys:
                    if dk in md:
                        op_sums[topo][op][dk] += float(md[dk])
                op_counts[topo][op] += 1

    # Finalize per-topology summary
    for topo in sorted(task_counts.keys()):
        s: Dict[str, Any] = {}
        # Task counts
        s['task_counts'] = dict(task_counts[topo])
        # Task 2 stats
        t2_stats = {
            k: finalize_mean_std(v) for k, v in stats_t2[topo].items()
        }
        s['composite_field_metrics'] = t2_stats
        # Task 3 regimes
        s['regimes'] = dict(regimes[topo])
        # Task 1 correlations (means)
        s['correlations'] = {
            ck: (corr_sums[topo][ck] / corr_counts[topo][ck]
                 if corr_counts[topo][ck] > 0 else 0.0)
            for ck in corr_keys
        }
        # Task 5 operator deltas (means)
        op_block: Dict[str, Dict[str, float]] = {}
        for op in ops:
            if op_counts[topo][op] > 0:
                op_block[op] = {
                    dk: op_sums[topo][op][dk] / op_counts[topo][op]
                    for dk in deltas_keys
                }
        s['operator_field_coupling'] = op_block
        summary[topo] = s

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {out_path}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
