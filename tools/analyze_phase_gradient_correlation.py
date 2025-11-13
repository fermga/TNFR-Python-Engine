"""Analyze Phase Gradient Correlation Study Results
==================================================

Processes JSONL output from phase_gradient_correlation_study.py and computes:
- Pearson correlation between |∇φ| and ΔC across all conditions
- Correlation breakdown by topology, intensity, and sequence type
- Comparison with Φ_s baseline correlation
- Identification of regimes where |corr| > 0.3 (target threshold)

Usage (PowerShell):
    python tools/analyze_phase_gradient_correlation.py --file correlation_study.jsonl

Status: RESEARCH - Part of |∇φ| canonical validation process
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file into list of dictionaries."""
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
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def analyze_overall_correlation(data: List[dict]) -> Dict[str, float]:
    """Compute overall correlation between |∇φ| and ΔC."""
    delta_grad_phi = [row['delta_grad_phi'] for row in data]
    delta_C = [row['delta_C'] for row in data]
    delta_phi_s = [row['delta_phi_s'] for row in data]
    
    corr_grad_phi = pearson_correlation(delta_grad_phi, delta_C)
    corr_phi_s = pearson_correlation(delta_phi_s, delta_C)
    
    return {
        'corr_grad_phi_vs_C': corr_grad_phi,
        'corr_phi_s_vs_C': corr_phi_s,
        'n_samples': len(data),
    }


def analyze_by_condition(
    data: List[dict],
    condition_key: str
) -> Dict[str, Dict[str, float]]:
    """Analyze correlation grouped by a specific condition."""
    grouped = defaultdict(list)
    
    for row in data:
        key = row.get(condition_key, 'unknown')
        grouped[key].append(row)
    
    results = {}
    for key, rows in grouped.items():
        delta_grad_phi = [r['delta_grad_phi'] for r in rows]
        delta_C = [r['delta_C'] for r in rows]
        delta_phi_s = [r['delta_phi_s'] for r in rows]
        
        results[str(key)] = {
            'corr_grad_phi_vs_C': pearson_correlation(delta_grad_phi, delta_C),
            'corr_phi_s_vs_C': pearson_correlation(delta_phi_s, delta_C),
            'n_samples': len(rows),
            'mean_delta_C': sum(delta_C) / len(delta_C) if delta_C else 0.0,
        }
    
    return results


def find_high_correlation_regimes(
    data: List[dict],
    threshold: float = 0.3
) -> List[Dict]:
    """Find specific conditions where |corr| exceeds threshold."""
    # Group by combination of topology, intensity, and sequence_type
    grouped = defaultdict(list)
    
    for row in data:
        key = (
            row.get('topology', 'unknown'),
            row.get('intensity', 0.0),
            row.get('sequence_type', 'unknown'),
        )
        grouped[key].append(row)
    
    high_corr_regimes = []
    
    for key, rows in grouped.items():
        if len(rows) < 5:  # Need minimum samples
            continue
        
        topology, intensity, seq_type = key
        delta_grad_phi = [r['delta_grad_phi'] for r in rows]
        delta_C = [r['delta_C'] for r in rows]
        
        corr = pearson_correlation(delta_grad_phi, delta_C)
        
        if abs(corr) >= threshold:
            high_corr_regimes.append({
                'topology': topology,
                'intensity': intensity,
                'sequence_type': seq_type,
                'correlation': corr,
                'n_samples': len(rows),
                'mean_delta_C': sum(delta_C) / len(delta_C),
            })
    
    # Sort by absolute correlation (descending)
    high_corr_regimes.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return high_corr_regimes


def analyze_path_gradient_effectiveness(data: List[dict]) -> Dict[str, float]:
    """Analyze path-integrated gradient for RA-dominated sequences."""
    ra_data = [r for r in data if r.get('sequence_type') == 'RA_dominated']
    
    if not ra_data:
        return {'n_samples': 0}
    
    mean_pig = [r['mean_path_gradient'] for r in ra_data]
    delta_C = [r['delta_C'] for r in ra_data]
    
    corr = pearson_correlation(mean_pig, delta_C)
    
    return {
        'corr_path_gradient_vs_C': corr,
        'n_samples': len(ra_data),
        'mean_path_gradient': sum(mean_pig) / len(mean_pig) if mean_pig else 0.0,
    }


def print_results(results: Dict) -> None:
    """Pretty-print analysis results."""
    print("\n" + "=" * 80)
    print("PHASE GRADIENT CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Overall
    overall = results['overall']
    print(f"\n📊 OVERALL CORRELATION (n={overall['n_samples']})")
    print(f"  |∇φ| vs ΔC:  {overall['corr_grad_phi_vs_C']:+.4f}")
    print(f"  Φ_s vs ΔC:  {overall['corr_phi_s_vs_C']:+.4f}  [baseline]")
    
    # By topology
    print(f"\n🌐 BY TOPOLOGY")
    for topo, stats in results['by_topology'].items():
        print(f"  {topo:12s}  |∇φ|: {stats['corr_grad_phi_vs_C']:+.4f}  "
              f"Φ_s: {stats['corr_phi_s_vs_C']:+.4f}  (n={stats['n_samples']})")
    
    # By intensity
    print(f"\n⚡ BY INTENSITY")
    for intensity, stats in sorted(results['by_intensity'].items()):
        print(f"  I={intensity:4s}  |∇φ|: {stats['corr_grad_phi_vs_C']:+.4f}  "
              f"Φ_s: {stats['corr_phi_s_vs_C']:+.4f}  (n={stats['n_samples']})")
    
    # By sequence type
    print(f"\n🔄 BY SEQUENCE TYPE")
    for seq_type, stats in results['by_sequence'].items():
        print(f"  {seq_type:15s}  |∇φ|: {stats['corr_grad_phi_vs_C']:+.4f}  "
              f"Φ_s: {stats['corr_phi_s_vs_C']:+.4f}  (n={stats['n_samples']})")
    
    # High correlation regimes
    high_corr = results['high_correlation_regimes']
    print(f"\n🎯 HIGH CORRELATION REGIMES (|corr| ≥ 0.3)")
    if high_corr:
        for regime in high_corr[:10]:  # Top 10
            print(f"  {regime['topology']:12s} | I={regime['intensity']:.1f} | "
                  f"{regime['sequence_type']:15s} → corr = {regime['correlation']:+.4f}")
    else:
        print("  None found (no regimes exceed |corr| ≥ 0.3)")
    
    # Path gradient (RA sequences)
    path_grad = results['path_gradient']
    if path_grad['n_samples'] > 0:
        print(f"\n🌊 PATH-INTEGRATED GRADIENT (RA sequences only)")
        print(f"  PIG vs ΔC: {path_grad['corr_path_gradient_vs_C']:+.4f}  "
              f"(n={path_grad['n_samples']})")
        print(f"  Mean PIG:  {path_grad['mean_path_gradient']:.4f}")
    
    # Summary assessment
    print(f"\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)
    
    best_corr = overall['corr_grad_phi_vs_C']
    baseline_corr = overall['corr_phi_s_vs_C']
    
    print(f"  |∇φ| correlation: {best_corr:+.4f}")
    print(f"  Φ_s baseline:     {baseline_corr:+.4f}")
    
    if abs(best_corr) >= 0.5:
        print(f"\n  ✅ STRONG: |∇φ| achieves |corr| ≥ 0.5 (promotion criterion met)")
    elif abs(best_corr) >= 0.3:
        print(f"\n  🟡 MODERATE: |∇φ| achieves |corr| ≥ 0.3 (conditional use case)")
    else:
        print(f"\n  ❌ WEAK: |∇φ| remains < 0.3 (telemetry only)")
    
    if high_corr:
        print(f"  📌 Found {len(high_corr)} high-correlation regimes")
        print(f"     Best regime: {high_corr[0]['topology']} | "
              f"I={high_corr[0]['intensity']} | {high_corr[0]['sequence_type']}")
    
    print(f"\n" + "=" * 80 + "\n")


def main(argv: List[str]) -> int:
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze phase gradient correlation study"
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to JSONL results file'
    )
    
    args = parser.parse_args(argv)
    
    # Load data
    file_path = Path(args.file)
    if not file_path.is_absolute():
        # Assume relative to benchmarks/results/
        file_path = Path(__file__).parent.parent / "benchmarks" / "results" / file_path
    
    print(f"Loading data from: {file_path}")
    data = load_jsonl(file_path)
    
    if not data:
        print("ERROR: No data loaded")
        return 1
    
    print(f"Loaded {len(data)} experiments")
    
    # Run analyses
    results = {
        'overall': analyze_overall_correlation(data),
        'by_topology': analyze_by_condition(data, 'topology'),
        'by_intensity': analyze_by_condition(data, 'intensity'),
        'by_sequence': analyze_by_condition(data, 'sequence_type'),
        'high_correlation_regimes': find_high_correlation_regimes(data, threshold=0.3),
        'path_gradient': analyze_path_gradient_effectiveness(data),
    }
    
    # Print results
    print_results(results)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
