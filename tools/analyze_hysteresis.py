#!/usr/bin/env python3
"""
Hysteresis Testing for TNFR Phase Transition

Physical basis:
  - First-order transitions show hysteresis: P_frag(I↑) ≠ P_frag(I↓)
  - Continuous transitions: P_frag independent of approach direction
  - Tests if critical transition is first-order or continuous

Protocol:
  - Approach I_c from below: I = 2.00 → 2.03 → 2.05 → 2.07
  - Approach I_c from above: I = 2.50 → 2.20 → 2.10 → 2.07
  - Compare fragmentation rates at overlapping intensities

Usage:
  python benchmarks/u6_sequence_simulator.py --topology ring --num_nodes 200 \
    --degree 20 --num_sequences 15 --seed 99 --intensity 2.12 \
    --export u6_hysteresis_down_i212.jsonl
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_u6_results(jsonl_path: Path) -> List[Dict]:
    """Load JSONL results from U6 simulator."""
    results = []
    if not jsonl_path.exists():
        return results
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_fragmentation_rate(results: List[Dict], sequence_type: str = 'violate') -> float:
    """
    Compute fragmentation rate for given sequence type.
    
    Args:
        results: List of experiment results
        sequence_type: 'valid' or 'violate' (will match both _u6 suffixed and non-suffixed)
    
    Returns:
        Fragmentation rate [0, 1]
    """
    # Match both 'violate' and 'violate_u6' format
    filtered = [r for r in results if sequence_type in r['sequence_type']]
    if not filtered:
        return float('nan')
    
    # Check both 'fragmented' and 'fragmentation' keys for compatibility
    n_fragmented = sum(1 for r in filtered if r.get('fragmented', False) or r.get('fragmentation', False))
    return n_fragmented / len(filtered)


def analyze_hysteresis():
    """
    Analyze hysteresis by comparing fragmentation rates for
    intensity approached from above vs below.
    """
    
    print("\n=== Hysteresis Analysis for TNFR Phase Transition ===\n")
    print("Testing: Does P_frag depend on approach direction?\n")
    print("  - First-order: Hysteresis expected (P↑ ≠ P↓)")
    print("  - Continuous: No hysteresis (P↑ = P↓)\n")
    
    # Define approach sequences
    # UP: increasing intensity
    up_sequence = [
        (1.50, 'u6_i150.jsonl'),
        (2.00, 'u6_i200.jsonl'),
        (2.03, 'u6_fine_i203.jsonl'),
        (2.05, 'u6_i205.jsonl'),
        (2.07, 'u6_fine_i207.jsonl'),
        (2.08, 'u6_fine_i208.jsonl'),
    ]
    
    # DOWN: decreasing intensity (to be collected)
    down_sequence = [
        (2.50, 'u6_i250.jsonl'),
        (2.20, 'u6_i220.jsonl'),
        (2.10, 'u6_i210.jsonl'),
        (2.12, 'u6_hysteresis_down_i212.jsonl'),  # New
        (2.08, 'u6_hysteresis_down_i208.jsonl'),  # New
        (2.07, 'u6_hysteresis_down_i207.jsonl'),  # New
    ]
    
    print("Step 1: Check existing data\n")
    print("UP sequence (I↑):")
    print(f"{'Intensity':>10} | {'P_frag':>10} | {'Status':>15}")
    print("-" * 45)
    
    up_data = {}
    for intensity, filename in up_sequence:
        results = load_u6_results(Path(filename))
        if results:
            p_frag = compute_fragmentation_rate(results, 'violate')
            up_data[intensity] = p_frag
            status = "✓ Available"
        else:
            up_data[intensity] = float('nan')
            status = "Missing"
        
        print(f"{intensity:>10.2f} | {up_data[intensity]:>10.1%} | {status:>15}")
    
    print("\nDOWN sequence (I↓):")
    print(f"{'Intensity':>10} | {'P_frag':>10} | {'Status':>15}")
    print("-" * 45)
    
    down_data = {}
    down_missing = []
    for intensity, filename in down_sequence:
        results = load_u6_results(Path(filename))
        if results:
            p_frag = compute_fragmentation_rate(results, 'violate')
            down_data[intensity] = p_frag
            status = "✓ Available"
        else:
            down_data[intensity] = float('nan')
            status = "⚠ Need to run"
            down_missing.append((intensity, filename))
        
        print(f"{intensity:>10.2f} | {down_data[intensity]:>10.1%} | {status:>15}")
    
    # Check for overlapping intensities
    overlap = set(up_data.keys()) & set(down_data.keys())
    overlap = sorted([I for I in overlap if np.isfinite(up_data[I]) and np.isfinite(down_data[I])])
    
    if overlap:
        print("\n\nStep 2: Hysteresis test at overlapping intensities\n")
        print(f"{'Intensity':>10} | {'P↑ (up)':>10} | {'P↓ (down)':>10} | {'ΔP':>10} | {'Verdict':>15}")
        print("-" * 70)
        
        delta_P_values = []
        for I in overlap:
            p_up = up_data[I]
            p_down = down_data[I]
            delta_p = abs(p_up - p_down)
            delta_P_values.append(delta_p)
            
            if delta_p < 0.05:
                verdict = "No hysteresis"
            elif delta_p < 0.15:
                verdict = "Weak"
            else:
                verdict = "Strong"
            
            print(f"{I:>10.2f} | {p_up:>10.1%} | {p_down:>10.1%} | {delta_p:>10.1%} | {verdict:>15}")
        
        # Statistical test
        print("\n\nStatistical Summary:")
        print(f"  - Mean |ΔP|: {np.mean(delta_P_values):.1%}")
        print(f"  - Max |ΔP|: {np.max(delta_P_values):.1%}")
        print(f"  - N overlaps: {len(overlap)}")
        
        if np.mean(delta_P_values) < 0.05:
            print("\n  Verdict: ✓ CONTINUOUS TRANSITION (no hysteresis)")
            print("  → Consistent with mean-field universality class")
        elif np.mean(delta_P_values) < 0.15:
            print("\n  Verdict: ⚠ WEAK HYSTERESIS")
            print("  → May indicate weak first-order or finite-size effects")
        else:
            print("\n  Verdict: ✗ STRONG HYSTERESIS")
            print("  → First-order transition (discontinuous)")
    
    else:
        print("\n\nStep 2: No overlapping data for hysteresis test")
    
    # Generate commands for missing experiments
    if down_missing:
        print("\n\nStep 3: Commands to collect missing DOWN sequence data\n")
        print("Run these commands to complete hysteresis test:\n")
        
        for intensity, filename in down_missing:
            cmd = (
                f'python benchmarks/u6_sequence_simulator.py '
                f'--topology ring --num_nodes 200 --degree 20 '
                f'--num_sequences 15 --seed 99 --runs 3 '
                f'--intensity {intensity:.2f} '
                f'--export {filename}'
            )
            print(f"# Intensity {intensity:.2f}")
            print(cmd)
            print()
    
    print("="*70)


if __name__ == '__main__':
    analyze_hysteresis()
