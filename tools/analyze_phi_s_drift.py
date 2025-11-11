#!/usr/bin/env python3
"""
Analyze Structural Potential (Φ_s) Drift Dynamics

CRITICAL: No assumption of "gravity" - testing if TNFR structural dynamics
spontaneously generate drift toward Φ_s minima from first principles.

Physical hypothesis FROM TNFR nodal equation:
  - Nodes with high ΔNFR create "wells" in Φ_s field
  - If Φ_s(i) = Σ_j ΔNFR_j / d(i,j)^α acts as attractor
  - Then node trajectories should drift toward Φ_s minima over time
  - This would be EMERGENT long-range attraction (gravity-LIKE, not gravity)

Test:
  1. Track Φ_s landscape at t=0 (initial) and t=final
  2. Identify Φ_s minima (structural potential wells)
  3. Measure if nodes migrate toward wells vs random walk
  4. Quantify drift rate vs coherence evolution

Usage:
  python tools/analyze_phi_s_drift.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add src to path for fields module
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tnfr.physics.fields import compute_structural_potential


def load_u6_results(jsonl_path: Path) -> List[Dict]:
    """Load JSONL results from U6 simulator."""
    results = []
    if not jsonl_path.exists():
        return results
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_phi_s_statistics(results: List[Dict]) -> Dict:
    """
    Analyze Φ_s evolution: initial vs final landscapes.
    
    Tests emergent drift hypothesis:
      - Do nodes move toward Φ_s minima?
      - Is drift correlated with coherence change?
      - Does drift rate depend on sequence type (valid vs violation)?
    
    Returns:
        Statistics dictionary with drift metrics
    """
    stats = {
        'phi_s_initial_mean': [],
        'phi_s_final_mean': [],
        'phi_s_drift': [],  # final - initial (negative = drift toward minima)
        'coherence_change': [],
        'sequence_type': [],
        'fragmented': []
    }
    
    for r in results:
        phi_s_init = r.get('phi_s_mean_initial', np.nan)
        phi_s_final = r.get('phi_s_mean_final', np.nan)
        
        if not (np.isfinite(phi_s_init) and np.isfinite(phi_s_final)):
            continue
        
        drift = phi_s_final - phi_s_init
        delta_c = r.get('coherence_final', 0) - r.get('coherence_initial', 0)
        
        stats['phi_s_initial_mean'].append(phi_s_init)
        stats['phi_s_final_mean'].append(phi_s_final)
        stats['phi_s_drift'].append(drift)
        stats['coherence_change'].append(delta_c)
        stats['sequence_type'].append(r['sequence_type'])
        stats['fragmented'].append(r.get('fragmentation', False) or r.get('fragmented', False))
    
    return stats


def analyze_drift_correlation(stats: Dict) -> Dict:
    """
    Compute correlation between Φ_s drift and coherence change.
    
    Hypothesis: If Φ_s acts as emergent attractor, then:
      - Drift toward minima (Δ Φ_s < 0) → coherence increase (ΔC > 0)
      - Correlation: corr(Δ Φ_s, ΔC) < 0 (negative correlation expected)
    """
    drift = np.array(stats['phi_s_drift'])
    delta_c = np.array(stats['coherence_change'])
    
    valid_mask = np.isfinite(drift) & np.isfinite(delta_c)
    drift_valid = drift[valid_mask]
    delta_c_valid = delta_c[valid_mask]
    
    if len(drift_valid) < 2:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'n_samples': len(drift_valid),
            'interpretation': 'Insufficient data'
        }
    
    # Pearson correlation
    corr_matrix = np.corrcoef(drift_valid, delta_c_valid)
    corr = corr_matrix[0, 1]
    
    # Interpretation
    if corr < -0.3:
        interpretation = 'Strong emergent attraction (Φ_s acts as attractor)'
    elif corr < -0.1:
        interpretation = 'Weak emergent attraction'
    elif corr > 0.1:
        interpretation = 'Repulsion or no drift (NOT gravity-like)'
    else:
        interpretation = 'No clear pattern (neutral drift)'
    
    return {
        'correlation': corr,
        'n_samples': len(drift_valid),
        'interpretation': interpretation,
        'mean_drift': np.mean(drift_valid),
        'mean_delta_c': np.mean(delta_c_valid)
    }


def analyze_by_sequence_type(stats: Dict) -> Dict:
    """
    Compare Φ_s drift between valid and violation sequences.
    
    Hypothesis: Valid sequences maintain coherence → minimal drift
                Violations lose coherence → may show larger drift
    """
    drift = np.array(stats['phi_s_drift'])
    seq_types = np.array(stats['sequence_type'])
    
    results = {}
    for seq_type in ['valid', 'violate']:
        mask = np.array(['valid' in s if seq_type == 'valid' else 'violate' in s 
                         for s in seq_types])
        mask = mask & np.isfinite(drift)
        
        if np.sum(mask) == 0:
            results[seq_type] = {
                'mean_drift': np.nan,
                'std_drift': np.nan,
                'n_samples': 0
            }
            continue
        
        drift_subset = drift[mask]
        results[seq_type] = {
            'mean_drift': np.mean(drift_subset),
            'std_drift': np.std(drift_subset),
            'n_samples': len(drift_subset),
            'min_drift': np.min(drift_subset),
            'max_drift': np.max(drift_subset)
        }
    
    return results


def analyze_fragmentation_dependence(stats: Dict) -> Dict:
    """
    Test if fragmented systems show different Φ_s drift patterns.
    
    Hypothesis: Fragmentation → loss of global coherence → disrupted Φ_s field
    """
    drift = np.array(stats['phi_s_drift'])
    fragmented = np.array(stats['fragmented'])
    
    results = {}
    for frag_status in [True, False]:
        mask = (fragmented == frag_status) & np.isfinite(drift)
        
        if np.sum(mask) == 0:
            results[frag_status] = {
                'mean_drift': np.nan,
                'n_samples': 0
            }
            continue
        
        drift_subset = drift[mask]
        results[frag_status] = {
            'mean_drift': np.mean(drift_subset),
            'std_drift': np.std(drift_subset),
            'n_samples': len(drift_subset)
        }
    
    return results


def analyze_phi_s_drift():
    """Main analysis for Φ_s drift dynamics."""
    
    print("\n" + "="*70)
    print("=== Structural Potential (Φ_s) Drift Analysis ===")
    print("="*70)
    print("\nHypothesis: TNFR dynamics generate emergent drift toward Φ_s minima")
    print("(Testing gravity-LIKE behavior from structural equations, NOT assuming gravity)")
    print()
    
    # Load data from universality experiments
    data_files = [
        'u6_fine_i203.jsonl',
        'u6_fine_i207.jsonl',
        'u6_fine_i208.jsonl',
        'u6_fine_i209.jsonl',
    ]
    
    all_results = []
    print("Step 1: Load experimental data\n")
    print(f"{'File':>30} | {'Records':>8} | {'Status':>10}")
    print("-" * 60)
    
    for filename in data_files:
        filepath = Path(filename)
        results = load_u6_results(filepath)
        if results:
            all_results.extend(results)
            print(f"{filename:>30} | {len(results):>8} | {'✓ Loaded':>10}")
        else:
            print(f"{filename:>30} | {0:>8} | {'Missing':>10}")
    
    if not all_results:
        print("\nNo data available for analysis.")
        return
    
    print(f"\nTotal records: {len(all_results)}")
    
    # Compute statistics
    print("\n\nStep 2: Compute Φ_s drift statistics\n")
    stats = compute_phi_s_statistics(all_results)
    
    if len(stats['phi_s_drift']) == 0:
        print("No valid Φ_s data found in records.")
        return
    
    print(f"Valid samples with Φ_s data: {len(stats['phi_s_drift'])}")
    print(f"Mean Φ_s (initial): {np.mean(stats['phi_s_initial_mean']):.6f}")
    print(f"Mean Φ_s (final):   {np.mean(stats['phi_s_final_mean']):.6f}")
    print(f"Mean Φ_s drift:     {np.mean(stats['phi_s_drift']):.6f}")
    
    # Drift vs coherence correlation
    print("\n\nStep 3: Test emergent attraction hypothesis\n")
    print("Expected: Negative correlation (drift toward minima → coherence increase)")
    print()
    
    corr_results = analyze_drift_correlation(stats)
    
    print(f"{'Metric':>25} | {'Value':>15}")
    print("-" * 45)
    print(f"{'Correlation (Δ Φ_s, ΔC)':>25} | {corr_results['correlation']:>15.3f}")
    print(f"{'N samples':>25} | {corr_results['n_samples']:>15}")
    print(f"{'Mean drift':>25} | {corr_results['mean_drift']:>15.6f}")
    print(f"{'Mean ΔC':>25} | {corr_results['mean_delta_c']:>15.3f}")
    print()
    print(f"Interpretation: {corr_results['interpretation']}")
    
    # By sequence type
    print("\n\nStep 4: Compare valid vs violation sequences\n")
    seq_results = analyze_by_sequence_type(stats)
    
    print(f"{'Sequence Type':>15} | {'Mean Drift':>12} | {'Std':>10} | {'N':>6} | {'Range':>20}")
    print("-" * 75)
    for seq_type, data in seq_results.items():
        if data['n_samples'] > 0:
            range_str = f"[{data['min_drift']:.3f}, {data['max_drift']:.3f}]"
            print(f"{seq_type:>15} | {data['mean_drift']:>12.6f} | {data['std_drift']:>10.6f} | "
                  f"{data['n_samples']:>6} | {range_str:>20}")
        else:
            print(f"{seq_type:>15} | {'N/A':>12} | {'N/A':>10} | {0:>6} | {'N/A':>20}")
    
    # Fragmentation dependence
    print("\n\nStep 5: Fragmentation effect on Φ_s field\n")
    frag_results = analyze_fragmentation_dependence(stats)
    
    print(f"{'Fragmentation':>15} | {'Mean Drift':>12} | {'Std':>10} | {'N':>6}")
    print("-" * 55)
    for frag_status, data in frag_results.items():
        status_str = 'Fragmented' if frag_status else 'Coherent'
        if data['n_samples'] > 0:
            print(f"{status_str:>15} | {data['mean_drift']:>12.6f} | {data['std_drift']:>10.6f} | "
                  f"{data['n_samples']:>6}")
        else:
            print(f"{status_str:>15} | {'N/A':>12} | {'N/A':>10} | {0:>6}")
    
    # Final interpretation
    print("\n\n" + "="*70)
    print("EMERGENT DYNAMICS INTERPRETATION")
    print("="*70)
    
    corr = corr_results['correlation']
    mean_drift = corr_results['mean_drift']
    mean_delta_c = corr_results['mean_delta_c']
    
    if not np.isfinite(corr):
        print("\nInsufficient data for conclusive interpretation.")
    elif corr < -0.5:
        print("\n✓ EMERGENT POTENTIAL WELL DYNAMICS CONFIRMED")
        print(f"  - Strong negative correlation: corr(Δ Φ_s, ΔC) = {corr:.3f}")
        print(f"  - Mean drift: Δ Φ_s = {mean_drift:+.3f}")
        print(f"  - Mean coherence change: ΔC = {mean_delta_c:+.3f}")
        print()
        print("PHYSICAL INTERPRETATION:")
        if mean_drift > 0 and mean_delta_c < 0:
            print("  → Φ_s INCREASES (away from minima) → Coherence DECREASES")
            print("  → Systems are UNSTABLE when displaced from Φ_s minima")
            print("  → Φ_s minima = STABLE EQUILIBRIUM STATES (potential wells)")
            print()
            print("  This is GRAVITY-LIKE behavior:")
            print("    - Φ_s minima = gravitational potential wells")
            print("    - Displacement from minima = increases 'potential energy'")
            print("    - Loss of coherence = instability (like escaping gravity)")
            print("    - Strong correlation = tight coupling to potential landscape")
        elif mean_drift < 0 and mean_delta_c > 0:
            print("  → Φ_s DECREASES (toward minima) → Coherence INCREASES")
            print("  → Systems actively DRIFT toward Φ_s minima")
            print("  → Φ_s minima = ATTRACTORS (active drift)")
            print()
            print("  This is direct gravitational-like ATTRACTION:")
            print("    - Nodes spontaneously move toward Φ_s wells")
            print("    - Coherence gained by approaching equilibrium")
        else:
            print("  → Mixed dynamics (further investigation needed)")
        print()
        print("  Emergent from TNFR nodal equation, NOT assumed externally.")
    elif corr < -0.2:
        print("\n⚠ MODERATE POTENTIAL WELL EFFECT")
        print(f"  - Moderate negative correlation: corr = {corr:.3f}")
        print("  - Φ_s acts as weak stabilizer, not dominant attractor")
    elif abs(corr) < 0.1:
        print("\n○ NEUTRAL DRIFT")
        print("  - No clear correlation: Φ_s field present but not acting as attractor")
        print("  - Structural potential exists but doesn't dominate dynamics")
    else:
        print("\n✗ NO EMERGENT ATTRACTION")
        print("  - Positive or zero correlation: No drift toward Φ_s minima")
        print("  - Structural potential does NOT act as universal attractor")
        print("  - Gravity-like regime NOT emergent from TNFR dynamics")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    analyze_phi_s_drift()
