"""Post-Exercise Recovery with SHA - Physiological Protocol.

This example demonstrates SHA (Silence) in athletic training as a recovery
mechanism that enables adaptation consolidation.

Protocol: AL → VAL → OZ → IL → SHA
- Emission: Training session begins
- Expansion + Dissonance + Coherence: Stress and acute adaptation
- SHA: Recovery pause (νf → 0, adaptation consolidates)

Key insight: SHA models the essential recovery phase where reduced activity
allows physiological adaptations to consolidate.

References:
- Kellmann, M., et al. (2018). Recovery and Performance in Sport.
- See: docs/source/examples/SHA_CLINICAL_APPLICATIONS.md, Section 4
"""

from tnfr.operators.definitions import Emission, Expansion, Dissonance, Coherence, Silence
from tnfr.structural import create_nfr, run_sequence


def main():
    """Execute post-exercise recovery protocol with SHA."""
    print("=" * 70)
    print("POST-EXERCISE RECOVERY - SHA Adaptation Protocol")
    print("=" * 70)
    print("\nContext: Athlete completing high-intensity training cycle")
    print("Goal: Optimize adaptation through structured recovery\n")
    
    # Create muscle tissue node
    G, muscle = create_nfr("muscle_tissue", epi=0.50, vf=1.00)
    
    # === TRAINING + RECOVERY ===
    print("=" * 70)
    print("TRAINING → RECOVERY CYCLE")
    print("=" * 70)
    print()
    
    print("Protocol: AL → VAL → OZ → IL → SHA")
    print()
    
    print("Step 1: EMISSION (AL) - Training session begins")
    print("  Athlete initiates workout, system activation")
    print()
    
    print("Step 2: EXPANSION (VAL) - Intense muscular activation")
    print("  Activity: High-intensity intervals, heavy resistance")
    print("  Effect: Increased metabolic demand, fiber recruitment")
    print()
    
    print("Step 3: DISSONANCE (OZ) - Metabolic stress")
    print("  Markers: Lactate, ROS, microdamage (adaptive stimulus)")
    print("  Signal: Triggers remodeling response")
    print()
    
    print("Step 4: COHERENCE (IL) - Acute homeostatic response")
    print("  Process: Immediate compensation, metabolite clearance")
    print("  Training complete - adaptive stimulus applied")
    print()
    
    print("Step 5: SILENCE (SHA) - Recovery period [48-72 hours]")
    print("  Activities: Sleep, nutrition, light movement")
    print("  Effect: νf → 0 (minimal activity), adaptation emerges")
    print("  Process:")
    print("    • Day 1: Residual soreness, metabolite clearance")
    print("    • Day 2: Deep adaptation (protein synthesis peaks)")
    print("    • Result: Structural improvement for next training")
    print()
    
    # Execute the complete sequence
    run_sequence(G, muscle, [Emission(), Expansion(), Dissonance(), Coherence(), Silence()])
    
    print("✓ Recovery complete - Adaptation consolidated, ready for next training")
    print()
    
    # Training applications
    print("=" * 70)
    print("TRAINING APPLICATIONS")
    print("=" * 70)
    print()
    print("Periodization Model:")
    print("  • High frequency: SHA 24-48h (maintenance)")
    print("  • Medium frequency: SHA 48-72h (progressive overload)")
    print("  • Low frequency: SHA 72-96h+ (adaptation/taper)")
    print()
    print("Overtraining Prevention:")
    print("  ⚠ Warning: Inadequate SHA leads to accumulated stress")
    print("  ✓ Solution: Extend SHA, reduce training load")
    
    # Physiological correlates
    print()
    print("=" * 70)
    print("PHYSIOLOGICAL CORRELATES")
    print("=" * 70)
    print()
    print("TNFR Metric → Physiological Marker")
    print("-" * 70)
    print("SHA activation   → Recovery mode (HRV, resting HR)")
    print("νf reduction     → Metabolic downregulation (VO2, RMR)")
    print("EPI growth       → Structural adaptation (performance gains)")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT: SHA AS ADAPTATION WINDOW")
    print("=" * 70)
    print("\nSHA enables training adaptation through:")
    print("  1. Reduced activity (νf → 0) = metabolic downregulation")
    print("  2. Structure evolution (EPI increases) = adaptation emerges")
    print("\nWithout adequate SHA (recovery), training stress accumulates")
    print("without adaptation, leading to overtraining and decline.")
    
    print("\n" + "=" * 70)
    print("PROTOCOL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
    print("\n" + "=" * 70)
    print("REFERENCE")
    print("=" * 70)
    print("\nFor detailed protocol documentation, see:")
    print("  docs/source/examples/SHA_CLINICAL_APPLICATIONS.md")
    print("  Section 4: Post-Exercise Recovery Protocol")
    print("\nScientific references:")
    print("  • Kellmann et al. (2018). Recovery and Performance in Sport.")
    print("  • Halson (2014). Monitoring training load.")
