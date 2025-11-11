"""Analyze critical exponent β by topology.

Tests universality hypothesis: Does β vary with network topology?
If β is universal, all topologies should show similar exponent near critical point.
"""
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
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


def compute_frag_rate_by_topology(rows: List[dict]) -> Dict[str, float]:
    """Compute fragmentation rate per topology."""
    by_topo: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        if r.get('sequence_type') == 'violate_u6':
            topo = r.get('topology', 'unknown')
            by_topo[topo].append(bool(r.get('fragmentation', False)))
    
    rates = {}
    for topo, frags in by_topo.items():
        rates[topo] = sum(frags) / max(len(frags), 1)
    return rates


def estimate_critical_intensity_by_topology() -> Dict[str, float]:
    """Estimate I_c per topology using bisection from sweep data."""
    intensity_files = [
        (1.5, 'u6_threshold_i15.jsonl'),
        (2.0, 'u6_threshold_i20.jsonl'),
        (2.03, 'u6_fine_i203.jsonl'),
        (2.05, 'u6_threshold_i205.jsonl'),
        (2.07, 'u6_fine_i207.jsonl'),
        (2.08, 'u6_fine_i208.jsonl'),
        (2.09, 'u6_fine_i209.jsonl'),
        (2.1, 'u6_threshold_i21.jsonl'),
    ]
    
    # Collect fragmentation rates by topology at each intensity
    topo_intensity_frag: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    for intensity, filepath in intensity_files:
        rows = load_jsonl(Path(filepath))
        if not rows:
            continue
        rates = compute_frag_rate_by_topology(rows)
        for topo, rate in rates.items():
            topo_intensity_frag[topo].append((intensity, rate))
    
    # Find I_c per topology (intensity where P crosses 0.5, or interpolate)
    i_c_by_topo: Dict[str, float] = {}
    for topo, data in topo_intensity_frag.items():
        data = sorted(data, key=lambda x: x[0])
        # Find crossing point
        for i in range(len(data) - 1):
            i_curr, p_curr = data[i]
            i_next, p_next = data[i + 1]
            if p_curr < 0.5 <= p_next:
                # Linear interpolation
                i_c = i_curr + (0.5 - p_curr) / max(p_next - p_curr, 0.01) * (i_next - i_curr)
                i_c_by_topo[topo] = i_c
                break
            elif p_curr == 0.0 and p_next > 0.0:
                # Use midpoint if jump from 0
                i_c_by_topo[topo] = (i_curr + i_next) / 2.0
                break
        else:
            # Fallback: use 2.05 as default
            i_c_by_topo[topo] = 2.05
    
    return i_c_by_topo


def fit_power_law_exponent(i_c: float, data_points: List[Tuple[float, float]]) -> float:
    """Fit P_frag = A * (I - I_c)^β for I > I_c using log-log fit."""
    # Filter points above I_c with non-zero fragmentation
    filtered = [(i, p) for i, p in data_points if i > i_c and 0 < p < 1.0]
    
    if len(filtered) < 2:
        return float('nan')
    
    # Log-log fit: log(P) = log(A) + β * log(I - I_c)
    x = [math.log(i - i_c) for i, p in filtered]
    y = [math.log(p) for i, p in filtered]
    
    # Linear regression
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denom = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if abs(denom) < 1e-10:
        return float('nan')
    
    beta = num / denom
    return beta


def analyze_universality():
    print("\n=== Critical Exponent β by Topology ===\n")
    print("Testing universality hypothesis: Does β vary with topology?\n")
    
    # Get I_c estimates per topology
    i_c_by_topo = estimate_critical_intensity_by_topology()
    
    print("Step 1: Estimate I_c per topology")
    print(f"{'Topology':>12} | {'I_c (estimated)':>16}")
    print("-" * 32)
    for topo in sorted(i_c_by_topo.keys()):
        print(f"{topo:>12} | {i_c_by_topo[topo]:>16.3f}")
    
    # Collect full intensity sweep data per topology
    intensity_files = [
        (1.5, 'u6_threshold_i15.jsonl'),
        (2.0, 'u6_threshold_i20.jsonl'),
        (2.03, 'u6_fine_i203.jsonl'),
        (2.05, 'u6_threshold_i205.jsonl'),
        (2.07, 'u6_fine_i207.jsonl'),
        (2.08, 'u6_fine_i208.jsonl'),
        (2.09, 'u6_fine_i209.jsonl'),
        (2.1, 'u6_threshold_i21.jsonl'),
        (2.2, 'u6_threshold_i22.jsonl'),
        (2.5, 'u6_threshold_i25.jsonl'),
    ]
    
    topo_intensity_frag: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    for intensity, filepath in intensity_files:
        rows = load_jsonl(Path(filepath))
        if not rows:
            continue
        rates = compute_frag_rate_by_topology(rows)
        for topo, rate in rates.items():
            topo_intensity_frag[topo].append((intensity, rate))
    
    # Fit β per topology
    print("\nStep 2: Fit power-law exponent β = d(log P)/d(log(I-I_c))")
    print(f"{'Topology':>12} | {'β (fitted)':>12} | {'Status':>20}")
    print("-" * 50)
    
    beta_values = []
    for topo in sorted(topo_intensity_frag.keys()):
        data = sorted(topo_intensity_frag[topo], key=lambda x: x[0])
        i_c = i_c_by_topo.get(topo, 2.05)
        beta = fit_power_law_exponent(i_c, data)
        
        if math.isnan(beta):
            status = "Insufficient data"
        elif beta < 0:
            status = "Negative (unphysical)"
        elif 0.5 <= beta <= 1.5:
            status = "Mean-field class"
            beta_values.append(beta)
        else:
            status = "Anomalous"
        
        print(f"{topo:>12} | {beta:>12.3f} | {status:>20}")
    
    # Universality test
    if len(beta_values) >= 2:
        beta_mean = sum(beta_values) / len(beta_values)
        beta_std = (sum((b - beta_mean)**2 for b in beta_values) / len(beta_values)) ** 0.5
        beta_cv = beta_std / beta_mean if beta_mean > 0 else float('nan')
        
        print("\nStep 3: Universality Test")
        print(f"{'Metric':>20} | {'Value':>12}")
        print("-" * 35)
        print(f"{'Mean β':>20} | {beta_mean:>12.3f}")
        print(f"{'Std Dev':>20} | {beta_std:>12.3f}")
        print(f"{'Coefficient of Var':>20} | {beta_cv:>12.3f}")
        
        if beta_cv < 0.15:
            verdict = "✓ UNIVERSAL (CV < 15%)"
        elif beta_cv < 0.25:
            verdict = "~ WEAK UNIVERSALITY (CV < 25%)"
        else:
            verdict = "✗ NON-UNIVERSAL (CV ≥ 25%)"
        
        print(f"\n{'Verdict':>20} | {verdict}")
        
        # Interpretation
        print("\nInterpretation:")
        if beta_mean < 0.5:
            print("  - Subcritical: Transition smoother than mean-field")
        elif 0.5 <= beta_mean <= 1.5:
            print("  - Mean-field class: Consistent with long-range interactions")
            print("  - Theoretical β_MF = 0.5 (Ising), 1.0 (Landau)")
        else:
            print("  - Supercritical: Sharper transition than mean-field")
        
        if beta_cv < 0.15:
            print("  - Strong universality: β independent of topology")
            print("  - Suggests common underlying critical dynamics")
    else:
        print("\nInsufficient topologies for universality test.")
    
    # Additional: Show fragmentation curves
    print("\n" + "="*60)
    print("Fragmentation Rate vs Intensity by Topology")
    print("="*60)
    for topo in sorted(topo_intensity_frag.keys()):
        data = sorted(topo_intensity_frag[topo], key=lambda x: x[0])
        print(f"\n{topo}:")
        for intensity, p_frag in data:
            bar = "█" * int(p_frag * 40)
            print(f"  I={intensity:.2f}: {p_frag:5.1%} {bar}")


if __name__ == '__main__':
    analyze_universality()
