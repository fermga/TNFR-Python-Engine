"""Analyze Phase Gradient Correlation with Alternative Metrics
================================================================

Computes correlations between |∇φ| changes and alternative TNFR metrics
that capture dynamics missed by C(t).

Metrics analyzed:
1. Mean ΔNFR: System-wide reorganization pressure
2. Max ΔNFR: Peak node stress (fragmentation risk)
3. Sense Index (Si): Stable reorganization capacity

Usage:
    python tools/analyze_alternative_metrics.py \
        --file alternative_metrics_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    rows = []
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


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) 
                   for xi, yi in zip(x, y))
    
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def analyze_overall_correlations(data: List[dict]) -> Dict:
    """Compute overall correlations across all experiments."""
    # Extract deltas
    delta_grad_phi = [r['delta_grad_phi'] for r in data]
    delta_mean_dnfr = [r['delta_mean_dnfr'] for r in data]
    delta_max_dnfr = [r['delta_max_dnfr'] for r in data]
    delta_si = [r['delta_si'] for r in data]
    delta_phi_s = [r['delta_phi_s'] for r in data]
    delta_C = [r['delta_C'] for r in data]
    
    # Compute correlations
    corr_mean_dnfr = pearson_correlation(delta_grad_phi, delta_mean_dnfr)
    corr_max_dnfr = pearson_correlation(delta_grad_phi, delta_max_dnfr)
    corr_si = pearson_correlation(delta_grad_phi, delta_si)
    corr_phi_s = pearson_correlation(delta_grad_phi, delta_phi_s)
    corr_C = pearson_correlation(delta_grad_phi, delta_C)
    
    return {
        'mean_dnfr': corr_mean_dnfr,
        'max_dnfr': corr_max_dnfr,
        'si': corr_si,
        'phi_s': corr_phi_s,
        'C': corr_C,
    }


def analyze_by_condition(
    data: List[dict],
    condition_key: str
) -> Dict[str, Dict]:
    """Group by condition and compute correlations."""
    groups = {}
    for row in data:
        key = row[condition_key]
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    
    results = {}
    for key, group_data in groups.items():
        if len(group_data) < 3:
            continue
        
        delta_grad_phi = [r['delta_grad_phi'] for r in group_data]
        delta_mean_dnfr = [r['delta_mean_dnfr'] for r in group_data]
        delta_max_dnfr = [r['delta_max_dnfr'] for r in group_data]
        delta_si = [r['delta_si'] for r in group_data]
        
        results[key] = {
            'mean_dnfr': pearson_correlation(delta_grad_phi, 
                                            delta_mean_dnfr),
            'max_dnfr': pearson_correlation(delta_grad_phi, 
                                           delta_max_dnfr),
            'si': pearson_correlation(delta_grad_phi, delta_si),
            'n': len(group_data),
        }
    
    return results


def find_high_correlation_regimes(
    data: List[dict],
    threshold: float = 0.5
) -> List[Dict]:
    """Find (topology, intensity, sequence_type) with strong correlation."""
    regimes = {}
    
    for row in data:
        key = (row['topology'], row['intensity'], row['sequence_type'])
        if key not in regimes:
            regimes[key] = []
        regimes[key].append(row)
    
    strong_regimes = []
    
    for key, group_data in regimes.items():
        if len(group_data) < 3:
            continue
        
        delta_grad_phi = [r['delta_grad_phi'] for r in group_data]
        delta_mean_dnfr = [r['delta_mean_dnfr'] for r in group_data]
        delta_max_dnfr = [r['delta_max_dnfr'] for r in group_data]
        delta_si = [r['delta_si'] for r in group_data]
        
        corr_mean = pearson_correlation(delta_grad_phi, delta_mean_dnfr)
        corr_max = pearson_correlation(delta_grad_phi, delta_max_dnfr)
        corr_si = pearson_correlation(delta_grad_phi, delta_si)
        
        # Check if ANY metric exceeds threshold
        if (abs(corr_mean) >= threshold or 
            abs(corr_max) >= threshold or 
            abs(corr_si) >= threshold):
            
            topology, intensity, seq_type = key
            strong_regimes.append({
                'topology': topology,
                'intensity': intensity,
                'sequence_type': seq_type,
                'corr_mean_dnfr': corr_mean,
                'corr_max_dnfr': corr_max,
                'corr_si': corr_si,
                'n': len(group_data),
            })
    
    # Sort by strongest correlation
    strong_regimes.sort(
        key=lambda r: max(abs(r['corr_mean_dnfr']), 
                         abs(r['corr_max_dnfr']), 
                         abs(r['corr_si'])),
        reverse=True
    )
    
    return strong_regimes


def print_results(
    overall: Dict,
    by_topology: Dict,
    by_intensity: Dict,
    by_sequence: Dict,
    strong_regimes: List[Dict],
    data: List[dict]
) -> None:
    """Pretty-print analysis results."""
    print("\n" + "="*80)
    print("PHASE GRADIENT vs ALTERNATIVE METRICS CORRELATION ANALYSIS")
    print("="*80)
    
    print(f"\n📊 OVERALL CORRELATION (n={len(data)})")
    print(f"  |∇φ| vs Δ(mean_ΔNFR):  {overall['mean_dnfr']:+.4f}")
    print(f"  |∇φ| vs Δ(max_ΔNFR):   {overall['max_dnfr']:+.4f}")
    print(f"  |∇φ| vs Δ(Si):         {overall['si']:+.4f}")
    print(f"  |∇φ| vs Δ(Φ_s):        {overall['phi_s']:+.4f} [baseline]")
    print(f"  |∇φ| vs Δ(C):          {overall['C']:+.4f} [C(t) - known weak]")
    
    # Assessment
    max_corr = max(abs(overall['mean_dnfr']), 
                   abs(overall['max_dnfr']), 
                   abs(overall['si']))
    
    if max_corr >= 0.5:
        status = "✅ STRONG"
    elif max_corr >= 0.3:
        status = "🟡 MODERATE"
    else:
        status = "❌ WEAK"
    
    print(f"\n  {status}: Max |corr| = {max_corr:.4f}")
    
    # By topology
    print(f"\n🌐 BY TOPOLOGY")
    for topo in sorted(by_topology.keys()):
        stats = by_topology[topo]
        print(f"  {topo:12s}  mean_ΔNFR: {stats['mean_dnfr']:+.4f}  "
              f"max_ΔNFR: {stats['max_dnfr']:+.4f}  "
              f"Si: {stats['si']:+.4f}  (n={stats['n']})")
    
    # By intensity
    print(f"\n⚡ BY INTENSITY")
    for intensity in sorted(by_intensity.keys()):
        stats = by_intensity[intensity]
        print(f"  I={intensity:.1f}  mean_ΔNFR: {stats['mean_dnfr']:+.4f}  "
              f"max_ΔNFR: {stats['max_dnfr']:+.4f}  "
              f"Si: {stats['si']:+.4f}  (n={stats['n']})")
    
    # By sequence type
    print(f"\n🔄 BY SEQUENCE TYPE")
    for seq_type in sorted(by_sequence.keys()):
        stats = by_sequence[seq_type]
        print(f"  {seq_type:15s}  mean_ΔNFR: {stats['mean_dnfr']:+.4f}  "
              f"max_ΔNFR: {stats['max_dnfr']:+.4f}  "
              f"Si: {stats['si']:+.4f}  (n={stats['n']})")
    
    # Strong regimes
    print(f"\n🎯 HIGH CORRELATION REGIMES (|corr| ≥ 0.5): "
          f"{len(strong_regimes)} found")
    
    for i, regime in enumerate(strong_regimes[:10], 1):
        max_corr_regime = max(abs(regime['corr_mean_dnfr']),
                             abs(regime['corr_max_dnfr']),
                             abs(regime['corr_si']))
        
        # Identify which metric is strongest
        if abs(regime['corr_mean_dnfr']) == max_corr_regime:
            metric = "mean_ΔNFR"
            corr_val = regime['corr_mean_dnfr']
        elif abs(regime['corr_max_dnfr']) == max_corr_regime:
            metric = "max_ΔNFR"
            corr_val = regime['corr_max_dnfr']
        else:
            metric = "Si"
            corr_val = regime['corr_si']
        
        print(f"  {i:2d}. {regime['topology']:11s} | "
              f"I={regime['intensity']:.1f} | "
              f"{regime['sequence_type']:15s} → "
              f"{metric}: {corr_val:+.4f}")
    
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    if max_corr >= 0.5:
        print(f"  ✅ CANONICAL PROMOTION RECOMMENDED")
        print(f"  |∇φ| achieves |corr| ≥ 0.5 with alternative metrics")
    elif max_corr >= 0.3:
        print(f"  🟡 TELEMETRY STATUS (moderate predictive power)")
    else:
        print(f"  ❌ RESEARCH ONLY (weak predictive power)")
    
    print(f"\n  Best metric: ", end="")
    if abs(overall['mean_dnfr']) == max_corr:
        print(f"mean_ΔNFR (system-wide stress)")
    elif abs(overall['max_dnfr']) == max_corr:
        print(f"max_ΔNFR (peak node stress)")
    else:
        print(f"Si (reorganization capacity)")
    
    print("="*80 + "\n")


def main(argv: List[str]) -> int:
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze alternative metrics correlation"
    )
    parser.add_argument(
        '--file', type=str, required=True,
        help='JSONL results file'
    )
    
    args = parser.parse_args(argv)
    
    # Load data
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = (Path(__file__).parent.parent / "benchmarks" / 
                    "results" / file_path)
    
    print(f"Loading data from: {file_path}")
    data = load_jsonl(file_path)
    
    if not data:
        print("ERROR: No data loaded")
        return 1
    
    print(f"Loaded {len(data)} experiments")
    
    # Compute correlations
    overall = analyze_overall_correlations(data)
    by_topology = analyze_by_condition(data, 'topology')
    by_intensity = analyze_by_condition(data, 'intensity')
    by_sequence = analyze_by_condition(data, 'sequence_type')
    strong_regimes = find_high_correlation_regimes(data, threshold=0.5)
    
    # Print results
    print_results(overall, by_topology, by_intensity, by_sequence,
                 strong_regimes, data)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
