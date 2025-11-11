#!/usr/bin/env python3
"""
Test Φ_s drift in RA-dominated sequences (Resonance-heavy).

Hypothesis: 
  - Violations: PASSIVE drift (pushed away from Φ_s minima by destabilizers)
  - RA-heavy: ACTIVE drift (pulled toward Φ_s minima by resonance amplification)

Protocol:
  1. Generate RA-dominated sequences: [AL, UM, RA, RA, RA, IL, SHA]
  2. Compare Δ Φ_s in RA-heavy vs violation sequences
  3. Test if RA creates NEGATIVE drift (toward minima) vs positive (away)

Expected:
  - Violations: Δ Φ_s > 0 (repulsion), ΔC < 0 (loss)
  - RA-heavy: Δ Φ_s < 0? (attraction), ΔC > 0 (gain)
"""

import json
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("=== Φ_s Drift: RA-Dominated vs Violation Sequences ===")
print("="*70)

print("\nHypothesis:")
print("  - Violations (OZ-heavy): Passive drift AWAY from Φ_s minima")
print("  - RA-dominated: Active drift TOWARD Φ_s minima (if gravity-like)")
print()

# Load violation data
print("Step 1: Analyze violation sequences (destabilizer-heavy)\n")

violation_files = [
    'u6_fine_i203.jsonl',
    'u6_fine_i207.jsonl',
    'u6_fine_i208.jsonl',
    'u6_fine_i209.jsonl',
]

violation_drifts = []
violation_delta_c = []

for filename in violation_files:
    path = Path(filename)
    if not path.exists():
        continue
    
    data = [json.loads(line) for line in open(path)]
    
    for r in data:
        if 'violate' not in r['sequence_type']:
            continue
        
        phi_init = r.get('phi_s_mean_initial', np.nan)
        phi_final = r.get('phi_s_mean_final', np.nan)
        c_init = r.get('coherence_initial', np.nan)
        c_final = r.get('coherence_final', np.nan)
        
        if not (np.isfinite(phi_init) and np.isfinite(phi_final) and 
                np.isfinite(c_init) and np.isfinite(c_final)):
            continue
        
        violation_drifts.append(phi_final - phi_init)
        violation_delta_c.append(c_final - c_init)

print(f"Violation sequences analyzed: {len(violation_drifts)}")
print(f"  Mean Δ Φ_s: {np.mean(violation_drifts):+.3f} (positive = away from minima)")
print(f"  Mean ΔC:    {np.mean(violation_delta_c):+.3f}")
print(f"  Std Δ Φ_s:  {np.std(violation_drifts):.3f}")

# Analyze valid sequences (more RA-balanced, less OZ)
print("\n\nStep 2: Analyze valid sequences (stabilizer-balanced)\n")

valid_drifts = []
valid_delta_c = []

for filename in violation_files:
    path = Path(filename)
    if not path.exists():
        continue
    
    data = [json.loads(line) for line in open(path)]
    
    for r in data:
        if 'valid' not in r['sequence_type']:
            continue
        
        phi_init = r.get('phi_s_mean_initial', np.nan)
        phi_final = r.get('phi_s_mean_final', np.nan)
        c_init = r.get('coherence_initial', np.nan)
        c_final = r.get('coherence_final', np.nan)
        
        if not (np.isfinite(phi_init) and np.isfinite(phi_final) and 
                np.isfinite(c_init) and np.isfinite(c_final)):
            continue
        
        valid_drifts.append(phi_final - phi_init)
        valid_delta_c.append(c_final - c_init)

print(f"Valid sequences analyzed: {len(valid_drifts)}")
print(f"  Mean Δ Φ_s: {np.mean(valid_drifts):+.3f} (negative would = toward minima)")
print(f"  Mean ΔC:    {np.mean(valid_delta_c):+.3f}")
print(f"  Std Δ Φ_s:  {np.std(valid_drifts):.3f}")

# Statistical comparison
print("\n\nStep 3: Statistical comparison\n")

print(f"{'Metric':>25} | {'Violations':>12} | {'Valid':>12} | {'Ratio':>10}")
print("-" * 70)

mean_viol = np.mean(violation_drifts)
mean_valid = np.mean(valid_drifts)
ratio = mean_valid / mean_viol if mean_viol != 0 else np.nan

print(f"{'Mean Δ Φ_s':>25} | {mean_viol:>+12.3f} | {mean_valid:>+12.3f} | {ratio:>10.2f}")

mean_dc_viol = np.mean(violation_delta_c)
mean_dc_valid = np.mean(valid_delta_c)
ratio_dc = mean_dc_valid / mean_dc_viol if mean_dc_viol != 0 else np.nan

print(f"{'Mean ΔC':>25} | {mean_dc_viol:>+12.3f} | {mean_dc_valid:>+12.3f} | {ratio_dc:>10.2f}")

# Interpretation
print("\n\nStep 4: Interpretation\n")

if mean_valid < 0 and mean_viol > 0:
    print("✓ ACTIVE DRIFT CONFIRMED (gravity-like attraction)")
    print("  → Valid sequences: Δ Φ_s < 0 (toward minima)")
    print("  → Violations: Δ Φ_s > 0 (away from minima)")
    print()
    print("  Physical mechanism:")
    print("    - IL/RA stabilizers → reduce ΔNFR → lower Φ_s")
    print("    - OZ destabilizers → increase ΔNFR → raise Φ_s")
    print("    - Φ_s minima = low-ΔNFR configurations")
    print("    - Grammar U6 maintains low Φ_s = stable equilibrium")
elif mean_valid > 0 and mean_valid < mean_viol:
    print("⚠ PASSIVE PROTECTION (no active attraction)")
    print(f"  → Valid sequences: Δ Φ_s = +{mean_valid:.3f} (still away, but less)")
    print(f"  → Violations: Δ Φ_s = +{mean_viol:.3f} (far away)")
    print(f"  → Reduction factor: {ratio:.2f}x")
    print()
    print("  Physical mechanism:")
    print("    - Valid sequences REDUCE drift away from minima")
    print("    - NO active attraction toward minima")
    print("    - Grammar U6 = STABILIZER, not attractor")
    print("    - Φ_s minima = equilibrium, not dynamic sink")
else:
    print("✗ UNEXPECTED PATTERN")
    print("  → Requires further investigation")

# Correlation analysis
print("\n\nStep 5: Correlation by sequence type\n")

corr_viol = np.corrcoef(violation_drifts, violation_delta_c)[0, 1]
corr_valid = np.corrcoef(valid_drifts, valid_delta_c)[0, 1]

print(f"{'Sequence Type':>15} | {'corr(Δ Φ_s, ΔC)':>16} | {'N':>6}")
print("-" * 50)
print(f"{'Violations':>15} | {corr_viol:>16.3f} | {len(violation_drifts):>6}")
print(f"{'Valid':>15} | {corr_valid:>16.3f} | {len(valid_drifts):>6}")

print("\n\nConclusion:")
if abs(corr_viol) > 0.7 and abs(corr_valid) > 0.7:
    print("  → STRONG Φ_s-coherence coupling in BOTH sequence types")
    print("  → Φ_s field universal, regardless of operator content")
    print("  → Gravity-like behavior emergent, not operator-specific")
else:
    print("  → Coupling strength depends on sequence type")
    print("  → May indicate operator-specific Φ_s dynamics")

print("\n" + "="*70)
