"""Safety Criterion Analysis for Phase Gradient |∇φ|
=====================================================

Analyzes whether |∇φ| provides unique early-warning signals for fragmentation
that are not captured by Φ_s alone.

Key Questions:
1. Does max(|∇φ|) spike before C(t) drops?
2. Is there a threshold max(|∇φ|) > threshold_gradient that predicts fragmentation?
3. Does |∇φ| provide information independent of Φ_s?

Usage:
    python tools/analyze_safety_criterion.py --file correlation_allnodes_test.jsonl

Status: RESEARCH - Part of |∇φ| canonical validation
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple


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


def analyze_fragmentation_events(data: List[dict]) -> Dict:
    """Identify fragmentation events (large negative ΔC)."""
    # Fragmentation defined as ΔC < -0.05 (5% coherence loss)
    fragmentation_threshold = -0.05
    
    fragmented = [r for r in data if r['delta_C'] < fragmentation_threshold]
    stable = [r for r in data if r['delta_C'] >= fragmentation_threshold]
    
    print(f"\n{'='*70}")
    print(f"FRAGMENTATION EVENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Threshold: ΔC < {fragmentation_threshold}")
    print(f"Fragmented runs: {len(fragmented)}/{len(data)} ({len(fragmented)/len(data)*100:.1f}%)")
    print(f"Stable runs: {len(stable)}/{len(data)} ({len(stable)/len(data)*100:.1f}%)")
    
    if not fragmented:
        print("\n⚠️  No fragmentation events detected (all sequences are stable)")
        return {
            'fragmented_count': 0,
            'stable_count': len(stable),
            'analysis': 'No fragmentation events for threshold analysis'
        }
    
    # Analyze |∇φ| in fragmented vs stable
    grad_phi_fragmented = [abs(r['grad_phi_final']) for r in fragmented]
    grad_phi_stable = [abs(r['grad_phi_final']) for r in stable]
    
    phi_s_fragmented = [abs(r['phi_s_final']) for r in fragmented]
    phi_s_stable = [abs(r['phi_s_final']) for r in stable]
    
    mean_grad_frag = sum(grad_phi_fragmented) / len(grad_phi_fragmented)
    mean_grad_stable = sum(grad_phi_stable) / len(grad_phi_stable)
    
    mean_phi_s_frag = sum(phi_s_fragmented) / len(phi_s_fragmented)
    mean_phi_s_stable = sum(phi_s_stable) / len(phi_s_stable)
    
    print(f"\n📊 FIELD COMPARISON")
    print(f"{'='*70}")
    print(f"  |∇φ| (fragmented): {mean_grad_frag:.4f}")
    print(f"  |∇φ| (stable):     {mean_grad_stable:.4f}")
    print(f"  Ratio:             {mean_grad_frag/mean_grad_stable:.2f}x")
    print(f"\n  Φ_s (fragmented):  {mean_phi_s_frag:.4f}")
    print(f"  Φ_s (stable):      {mean_phi_s_stable:.4f}")
    print(f"  Ratio:             {mean_phi_s_frag/mean_phi_s_stable:.2f}x")
    
    return {
        'fragmented_count': len(fragmented),
        'stable_count': len(stable),
        'mean_grad_phi_fragmented': mean_grad_frag,
        'mean_grad_phi_stable': mean_grad_stable,
        'mean_phi_s_fragmented': mean_phi_s_frag,
        'mean_phi_s_stable': mean_phi_s_stable,
    }


def calibrate_threshold(data: List[dict]) -> Dict:
    """Calibrate |∇φ| threshold for early warning."""
    # Try different thresholds and compute false positive/negative rates
    
    # Sort by final |∇φ|
    all_grad_phi = [(abs(r['grad_phi_final']), r['delta_C']) for r in data]
    all_grad_phi.sort()
    
    # Try percentile-based thresholds
    thresholds = []
    for percentile in [50, 75, 90, 95, 99]:
        idx = int(len(all_grad_phi) * percentile / 100)
        threshold_value = all_grad_phi[idx][0]
        thresholds.append((percentile, threshold_value))
    
    print(f"\n{'='*70}")
    print(f"|∇φ| THRESHOLD CALIBRATION")
    print(f"{'='*70}")
    
    best_threshold = None
    best_f1 = 0.0
    
    for percentile, threshold in thresholds:
        # Classify: high |∇φ| = predicted fragmentation
        true_positive = 0  # High |∇φ| and ΔC < 0
        false_positive = 0  # High |∇φ| but ΔC >= 0
        true_negative = 0  # Low |∇φ| and ΔC >= 0
        false_negative = 0  # Low |∇φ| but ΔC < 0
        
        for r in data:
            high_gradient = abs(r['grad_phi_final']) > threshold
            fragmented = r['delta_C'] < 0
            
            if high_gradient and fragmented:
                true_positive += 1
            elif high_gradient and not fragmented:
                false_positive += 1
            elif not high_gradient and not fragmented:
                true_negative += 1
            else:  # not high_gradient and fragmented
                false_negative += 1
        
        # Compute metrics
        total = len(data)
        accuracy = (true_positive + true_negative) / total if total > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPercentile {percentile}% → threshold = {threshold:.4f}")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-score:  {f1:.3f}")
        print(f"  TP={true_positive}, FP={false_positive}, TN={true_negative}, FN={false_negative}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = (percentile, threshold, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positive': true_positive,
                'false_positive': false_positive,
                'true_negative': true_negative,
                'false_negative': false_negative,
            })
    
    if best_threshold:
        print(f"\n{'='*70}")
        print(f"✅ BEST THRESHOLD")
        print(f"{'='*70}")
        percentile, threshold, metrics = best_threshold
        print(f"  Percentile: {percentile}%")
        print(f"  |∇φ|_max:   {threshold:.4f}")
        print(f"  F1-score:   {metrics['f1']:.3f}")
        print(f"  Precision:  {metrics['precision']:.3f}")
        print(f"  Recall:     {metrics['recall']:.3f}")
    
    return best_threshold


def compare_with_phi_s(data: List[dict]) -> None:
    """Compare |∇φ| and Φ_s predictive power."""
    print(f"\n{'='*70}")
    print(f"COMPARATIVE PREDICTIVE POWER")
    print(f"{'='*70}")
    
    # Correlation with ΔC
    delta_C = [r['delta_C'] for r in data]
    delta_grad_phi = [r['delta_grad_phi'] for r in data]
    delta_phi_s = [r['delta_phi_s'] for r in data]
    
    def pearson(x, y):
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom = math.sqrt(sum((xi - mean_x)**2 for xi in x) * sum((yi - mean_y)**2 for yi in y))
        return num / denom if denom != 0 else 0
    
    corr_grad_phi = pearson(delta_grad_phi, delta_C)
    corr_phi_s = pearson(delta_phi_s, delta_C)
    
    print(f"\nCorrelation with ΔC:")
    print(f"  |∇φ|: {corr_grad_phi:+.4f}")
    print(f"  Φ_s:  {corr_phi_s:+.4f}")
    print(f"\nRelative strength: {abs(corr_grad_phi)/abs(corr_phi_s):.1%} of Φ_s")
    
    # Check independence
    corr_fields = pearson(delta_grad_phi, delta_phi_s)
    print(f"\nIndependence check:")
    print(f"  corr(Δ|∇φ|, ΔΦ_s): {corr_fields:+.4f}")
    if abs(corr_fields) < 0.7:
        print(f"  ✅ Fields are relatively independent (|corr| < 0.7)")
    else:
        print(f"  ⚠️  Fields are highly correlated (redundant information)")


def main(argv: List[str]) -> int:
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(description="Safety criterion analysis")
    parser.add_argument('--file', type=str, required=True, help='JSONL results file')
    
    args = parser.parse_args(argv)
    
    # Load data
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent.parent / "benchmarks" / "results" / file_path
    
    print(f"Loading data from: {file_path}")
    data = load_jsonl(file_path)
    
    if not data:
        print("ERROR: No data loaded")
        return 1
    
    print(f"Loaded {len(data)} experiments")
    
    # Run analyses
    frag_stats = analyze_fragmentation_events(data)
    
    if frag_stats['fragmented_count'] > 0:
        threshold_info = calibrate_threshold(data)
    
    compare_with_phi_s(data)
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
