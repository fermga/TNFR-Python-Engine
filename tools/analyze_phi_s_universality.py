#!/usr/bin/env python3
"""
Analyze Φ_s drift for hierarchical topologies (tree, grid).
Compare with original topologies (ring, scale_free, ws).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path('benchmarks') / 'results'


def load_and_analyze(jsonl_path: Path, label: str):
    """Load and compute Φ_s-coherence correlation."""
    if not jsonl_path.exists():
        return None
    
    data = [json.loads(line) for line in open(jsonl_path)]
    
    phi_s_drift = []
    delta_c = []
    
    for r in data:
        phi_init = r.get('phi_s_mean_initial', np.nan)
        phi_final = r.get('phi_s_mean_final', np.nan)
        c_init = r.get('coherence_initial', np.nan)
        c_final = r.get('coherence_final', np.nan)

        if not (np.isfinite(phi_init) and np.isfinite(phi_final) and np.isfinite(c_init) and np.isfinite(c_final)):
            continue
        
        phi_s_drift.append(phi_final - phi_init)
        delta_c.append(c_final - c_init)
    
    if len(phi_s_drift) < 2:
        return None
    
    phi_arr = np.array(phi_s_drift)
    dc_arr = np.array(delta_c)
    
    corr_matrix = np.corrcoef(phi_arr, dc_arr)
    corr = corr_matrix[0, 1]
    
    return {
        'label': label,
        'n_samples': len(phi_s_drift),
        'corr': corr,
        'mean_phi_drift': np.mean(phi_arr),
        'mean_delta_c': np.mean(dc_arr),
        'std_phi_drift': np.std(phi_arr),
        'std_delta_c': np.std(dc_arr)
    }


print("\n" + "=" * 70)
print("=== Φ_s Universality Test: Hierarchical Topologies ===")
print("=" * 70)

# Original topologies (combined from fine-grained experiments)
print("\nStep 1: Load original topology data (ring, scale_free, ws)")
original_files = [
    'u6_fine_i203.jsonl',
    'u6_fine_i207.jsonl',
    'u6_fine_i208.jsonl',
    'u6_fine_i209.jsonl',
]

all_original = []
for f in original_files:
    path = BASE / f
    if path.exists():
        data = [json.loads(line) for line in open(path)]
        all_original.extend(data)

# Combine and analyze by topology
by_topo_orig = defaultdict(list)
for r in all_original:
    by_topo_orig[r['topology']].append(r)

results = []

for topo, records in by_topo_orig.items():
    phi_s_drift = []
    delta_c = []
    
    for r in records:
        phi_init = r.get('phi_s_mean_initial', np.nan)
        phi_final = r.get('phi_s_mean_final', np.nan)
        c_init = r.get('coherence_initial', np.nan)
        c_final = r.get('coherence_final', np.nan)
        
        if not (np.isfinite(phi_init) and np.isfinite(phi_final) and np.isfinite(c_init) and np.isfinite(c_final)):
            continue
        
        phi_s_drift.append(phi_final - phi_init)
        delta_c.append(c_final - c_init)
    
    if len(phi_s_drift) >= 2:
        phi_arr = np.array(phi_s_drift)
        dc_arr = np.array(delta_c)
        corr_matrix = np.corrcoef(phi_arr, dc_arr)
        corr = corr_matrix[0, 1]
        
        results.append({
            'label': topo,
            'n_samples': len(phi_s_drift),
            'corr': corr,
            'mean_phi_drift': np.mean(phi_arr),
            'mean_delta_c': np.mean(dc_arr)
        })

# Hierarchical topologies
print("\nStep 2: Load hierarchical topology data (tree, grid)")
hier_files = [
    ('u6_hierarchical_i207.jsonl', 'tree+grid (I=2.07)'),
    ('u6_hierarchical_i209.jsonl', 'tree+grid (I=2.09)'),
]

for filepath, label in hier_files:
    path = BASE / filepath
    if path.exists():
        data = [json.loads(line) for line in open(path)]
        by_topo = defaultdict(list)
        for r in data:
            by_topo[r['topology']].append(r)
        
        for topo, records in by_topo.items():
            phi_s_drift = []
            delta_c = []
            
            for r in records:
                phi_init = r.get('phi_s_mean_initial', np.nan)
                phi_final = r.get('phi_s_mean_final', np.nan)
                c_init = r.get('coherence_initial', np.nan)
                c_final = r.get('coherence_final', np.nan)
                
                if not (np.isfinite(phi_init) and np.isfinite(phi_final) and np.isfinite(c_init) and np.isfinite(c_final)):
                    continue
                
                phi_s_drift.append(phi_final - phi_init)
                delta_c.append(c_final - c_init)
            
            if len(phi_s_drift) >= 2:
                phi_arr = np.array(phi_s_drift)
                dc_arr = np.array(delta_c)
                corr_matrix = np.corrcoef(phi_arr, dc_arr)
                corr = corr_matrix[0, 1]
                
                results.append({
                    'label': f'{topo}',
                    'n_samples': len(phi_s_drift),
                    'corr': corr,
                    'mean_phi_drift': np.mean(phi_arr),
                    'mean_delta_c': np.mean(dc_arr)
                })

# Display results
print("\n\nStep 3: Compare correlations across all topologies\n")
print(f"{'Topology':>15} | {'N':>6} | {'corr(Δ Φ_s, ΔC)':>16} | {'Mean Δ Φ_s':>12} | {'Mean ΔC':>10}")
print("-" * 75)

for res in sorted(results, key=lambda x: x['label']):
    print(f"{res['label']:>15} | {res['n_samples']:>6} | {res['corr']:>16.3f} | "
          f"{res['mean_phi_drift']:>12.3f} | {res['mean_delta_c']:>10.3f}")

# Compute universality metric
print("\n\nStep 4: Universality test\n")
corrs = [r['corr'] for r in results if np.isfinite(r['corr'])]
mean_corr = np.mean(corrs)
std_corr = np.std(corrs)
cv = (std_corr / abs(mean_corr)) if mean_corr != 0 else np.nan

print(f"{'Metric':>20} | {'Value':>12}")
print("-" * 40)
print(f"{'Mean correlation':>20} | {mean_corr:>12.3f}")
print(f"{'Std Dev':>20} | {std_corr:>12.3f}")
print(f"{'CV (%)':>20} | {cv * 100:>12.1f}")
print(f"{'N topologies':>20} | {len(corrs):>12}")

if cv < 0.15:
    verdict = "✓ UNIVERSAL (CV < 15%)"
elif cv < 0.30:
    verdict = "⚠ MODERATE UNIVERSALITY"
else:
    verdict = "✗ NON-UNIVERSAL"

print(f"\n{verdict}")

print("\n\nInterpretation:")
if all(c < -0.5 for c in corrs):
    print("  → All topologies show STRONG negative correlation (|corr| > 0.5)")
    print("  → Φ_s potential well dynamics UNIVERSAL across network structures")
    print("  → Hierarchical (tree, grid) behave identically to distributed (ring, scale_free, ws)")
    print("  → Confirms: Φ_s emerges from nodal equation, NOT from topology")
elif cv < 0.15:
    print("  → Low variation (CV < 15%) indicates universal behavior")
    print("  → Φ_s dynamics consistent across diverse topologies")
else:
    print("  → High variation suggests topology-dependent effects")
    print("  → Φ_s may not be universal across all network families")

print("\n" + "=" * 70)
