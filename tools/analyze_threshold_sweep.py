"""Analyze threshold sweep results across multiple intensity values.

Combines multiple JSONL files to plot fragmentation rate vs intensity.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple


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


def analyze_threshold_sweep(intensity_files: List[Tuple[float, str]]) -> None:
    print("\n=== Fragmentation Threshold Sweep ===\n")
    print(f"{'Intensity':>10} | {'Valid Frag':>12} | {'Violate Frag':>14} | {'Total':>7}")
    print("-" * 60)
    
    for intensity, filepath in sorted(intensity_files):
        rows = load_jsonl(Path(filepath))
        if not rows:
            print(f"{intensity:>10.2f} | {'N/A':>12} | {'N/A':>14} | {'0':>7}")
            continue
        
        valid = [r for r in rows if r.get('sequence_type') == 'valid_u6']
        violate = [r for r in rows if r.get('sequence_type') == 'violate_u6']
        
        valid_frag = sum(1 for r in valid if r.get('fragmentation')) / max(len(valid), 1) * 100
        violate_frag = sum(1 for r in violate if r.get('fragmentation')) / max(len(violate), 1) * 100
        
        print(f"{intensity:>10.2f} | {valid_frag:>10.1f}% | {violate_frag:>12.1f}% | {len(rows):>7}")
    
    # Analyze fields at threshold
    print("\n=== Structural Fields at Critical Intensity (2.05) ===")
    rows_205 = load_jsonl(Path('u6_threshold_i205.jsonl'))
    if rows_205:
        violate_205 = [r for r in rows_205 if r.get('sequence_type') == 'violate_u6']
        frag_205 = [r for r in violate_205 if r.get('fragmentation')]
        no_frag_205 = [r for r in violate_205 if not r.get('fragmentation')]
        
        if frag_205 and no_frag_205:
            # Compare fields between fragmented vs non-fragmented at same intensity
            curv_frag = [abs(float(r.get('curv_phi_max_final', 0))) for r in frag_205]
            curv_no_frag = [abs(float(r.get('curv_phi_max_final', 0))) for r in no_frag_205]
            
            grad_frag = [float(r.get('grad_phi_mean_final', 0)) for r in frag_205]
            grad_no_frag = [float(r.get('grad_phi_mean_final', 0)) for r in no_frag_205]
            
            xi_frag = [float(r.get('xi_c_final', 0)) for r in frag_205 if r.get('xi_c_final')]
            xi_no_frag = [float(r.get('xi_c_final', 0)) for r in no_frag_205 if r.get('xi_c_final')]
            
            print(f"\n{'Metric':>20} | {'Fragmented':>12} | {'Non-Frag':>12}")
            print("-" * 50)
            print(f"{'|K_φ|_max':>20} | {sum(curv_frag)/len(curv_frag):>10.3f} | {sum(curv_no_frag)/len(curv_no_frag):>10.3f}")
            print(f"{'|∇φ|_mean':>20} | {sum(grad_frag)/len(grad_frag):>10.3f} | {sum(grad_no_frag)/len(grad_no_frag):>10.3f}")
            if xi_frag and xi_no_frag:
                print(f"{'ξ_C':>20} | {sum(xi_frag)/len(xi_frag):>10.3f} | {sum(xi_no_frag)/len(xi_no_frag):>10.3f}")


if __name__ == '__main__':
    intensity_files = [
        (1.5, 'u6_threshold_i15.jsonl'),
        (2.0, 'u6_threshold_i20.jsonl'),
        (2.05, 'u6_threshold_i205.jsonl'),
        (2.1, 'u6_threshold_i21.jsonl'),
        (2.2, 'u6_threshold_i22.jsonl'),
        (2.5, 'u6_threshold_i25.jsonl'),
        (3.5, 'u6_extreme_results.jsonl'),
    ]
    
    analyze_threshold_sweep(intensity_files)
