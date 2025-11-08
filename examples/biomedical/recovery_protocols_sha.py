"""Post-Exercise Recovery with SHA - Physiological Protocol.

This example demonstrates SHA (Silence) in athletic training as a recovery
mechanism that enables adaptation consolidation.

Protocol: [Training] VAL → OZ → IL → [Recovery] SHA → [Next Session] NAV → AL
- Training: Expansion + Dissonance + Coherence (stress and adaptation)
- Recovery: Silence (νf → 0, adaptation consolidates)
- Next training: Transition + Emission (return with improved baseline)

Key insight: SHA models the essential recovery phase where reduced activity
allows physiological adaptations to consolidate - muscle repair, metabolic
adjustments, and performance gains require structural pause.

References:
- Kellmann, M., et al. (2018). Recovery and Performance in Sport.
- See: docs/source/examples/SHA_CLINICAL_APPLICATIONS.md, Section 4
"""

from tnfr.operators.definitions import (
    Expansion,
    Dissonance,
    Coherence,
    Silence,
    Transition,
    Emission,
)
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
    
    # === TRAINING PHASE ===
    print("=" * 70)
    print("TRAINING PHASE - Controlled Stress Application")
    print("=" * 70)
    print()
    
    print("Protocol: VAL → OZ → IL (Exercise sequence)")
    print()
    
    print("Step 1: EXPANSION (VAL) - Intense muscular activation")
    print("  Activity: High-intensity intervals, heavy resistance")
    print("  Effect: Increased metabolic demand, fiber recruitment")
    print()
    
    print("Step 2: DISSONANCE (OZ) - Metabolic stress")
    print("  Markers: Lactate, ROS, microdamage (adaptive stimulus)")
    print("  Signal: Triggers remodeling response")
    print()
    
    print("Step 3: COHERENCE (IL) - Acute homeostatic response")
    print("  Process: Immediate compensation, metabolite clearance")
    print()
    
    run_sequence(G, muscle, [Expansion(), Dissonance(), Coherence()])
    
    print("✓ Training complete - Adaptive stimulus applied")
    print("  Markers: Elevated HR, temperature, fatigue, glycogen depletion")
    print()
    
    # === RECOVERY PHASE ===
    print("=" * 70)
    print("RECOVERY PHASE - SHA Adaptation Consolidation")
    print("=" * 70)
    print()
    
    print("Step 4: SILENCE (SHA) - Recovery period")
    print("  Context: 48-72 hour recovery period")
    print("  Activities: Sleep, nutrition, light movement")
    print("  Effect: νf → 0 (minimal activity), adaptation emerges")
    print("  Process:")
    print("    • Day 1: Residual soreness, metabolite clearance")
    print("    • Day 2: Deep adaptation (protein synthesis peaks)")
    print("    • Result: EPI increases (structural improvement)")
    print()
    
    run_sequence(G, muscle, [Silence()])
    
    print("✓ Recovery complete - Adaptation consolidated")
    print("  Changes: Muscle protein synthesis, glycogen supercompensation,")
    print("           mitochondrial biogenesis, neural efficiency")
    print()
    
    # === NEXT TRAINING ===
    print("=" * 70)
    print("NEXT TRAINING SESSION - Adapted Baseline")
    print("=" * 70)
    print()
    
    print("Protocol: NAV → AL (Return to training)")
    print()
    
    print("Step 5: TRANSITION (NAV) - Recovery → Active")
    print("  Process: Return to normal training readiness")
    print()
    
    print("Step 6: EMISSION (AL) - Next training begins")
    print("  Context: Training resumes with adapted system")
    print("  Expected: Improved baseline capacity")
    print()
    
    run_sequence(G, muscle, [Transition(), Emission()])
    
    print("✓ Ready for training - Performance capacity increased")
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
    print("  ⚠ Warning: ΔNFR fails to normalize during SHA")
    print("  → Intervention: Extend SHA, reduce training load")
    print("  ✓ Recovery: νf normalizes, ΔNFR < 0.10")
    print()
    print("Performance Optimization:")
    print("  • SHA timing: Match recovery to adaptation timeline")
    print("  • SHA quality: Lower EPI variance = better recovery")
    print("  • Return criteria: Low ΔNFR + normalized νf")
    
    # Physiological correlates
    print()
    print("=" * 70)
    print("PHYSIOLOGICAL CORRELATES")
    print("=" * 70)
    print()
    print("TNFR Metric → Physiological Marker → Measurement")
    print("-" * 70)
    print("SHA activation   → Recovery mode        → HRV, resting HR")
    print("νf reduction     → Metabolic downreg    → VO2, RMR")
    print("EPI growth       → Structural adaptation → Muscle cross-section")
    print("ΔNFR normal      → Stress clearance     → Cortisol, CK, IL-6")
    print("SHA duration     → Recovery time        → Performance tests")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT: SHA AS ADAPTATION WINDOW")
    print("=" * 70)
    print("\nSHA enables training adaptation through:")
    print("  1. Reduced activity (νf → 0) = metabolic downregulation")
    print("  2. Structure evolution (EPI increases) = adaptation emerges")
    print("  3. Pressure normalization (ΔNFR decreases) = recovery complete")
    print("\nWithout adequate SHA (recovery), training stress accumulates")
    print("without adaptation, leading to overtraining and performance decline.")
    
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
