"""Cardiac Coherence Training with SHA - HRV Consolidation Protocol.

This example demonstrates application of SHA (Silence) in Heart Rate Variability
(HRV) coherence training, showing how structural pause consolidates coherent patterns.

Protocol: AL → IL → SHA
- Emission: Initiate guided breathing
- Coherence: Stabilize HRV pattern
- Silence: Consolidate pattern (create "physiological memory")

References:
- McCraty, R., & Shaffer, F. (2015). Heart rate variability. Glob Adv Health Med, 4(1), 46-61.
- See: docs/source/examples/SHA_CLINICAL_APPLICATIONS.md, Section 1
"""

from tnfr.operators.definitions import Emission, Coherence, Silence
from tnfr.structural import create_nfr, run_sequence


def main():
    """Execute cardiac coherence training protocol with SHA consolidation."""
    print("=" * 70)
    print("CARDIAC COHERENCE TRAINING - SHA Consolidation Protocol")
    print("=" * 70)
    print("\nContext: Patient completing HRV biofeedback session")
    print("Goal: Consolidate coherent pattern before session end\n")
    
    # Create cardiac rhythm node
    G, heart = create_nfr("cardiac_rhythm", epi=0.30, vf=0.85)
    
    print("BASELINE (resting state)")
    print("  Patient at rest, normal heart rate variability")
    print()
    
    # Execute protocol: AL → IL → SHA
    print("EXECUTING PROTOCOL: AL → IL → SHA")
    print("-" * 70)
    print()
    
    print("Step 1: EMISSION (AL) - Guided breathing begins")
    print("  Patient: Breathing at 5-6 breaths/min (resonance frequency)")
    print("  Effect: Initiates coherent cardiac rhythm pattern")
    print()
    
    print("Step 2: COHERENCE (IL) - HRV pattern stabilizes")
    print("  Observation: Sinusoidal HRV at breathing frequency emerges")
    print("  Effect: Baroreceptor sensitivity increases, vagal tone strengthens")
    print()
    
    print("Step 3: SILENCE (SHA) - Pattern consolidation")
    print("  Instruction: 'Continue breathing gently, let the rhythm settle'")
    print("  Duration: 2-3 minutes of minimal-effort breathing")
    print("  Effect: νf → 0 (structural pause), EPI preserved (pattern locked in)")
    print()
    
    # Execute the complete sequence
    run_sequence(G, heart, [Emission(), Coherence(), Silence()])
    
    print("✓ Protocol complete - Coherent pattern consolidated")
    print()
    
    # Expected outcomes
    print("=" * 70)
    print("EXPECTED CLINICAL OUTCOMES")
    print("=" * 70)
    print("\nImmediate (post-session):")
    print("  • Sustained HRV coherence: 10-30 minutes")
    print("  • Reduced sympathetic activation")
    print("  • Subjective calm and centeredness")
    print("\nLong-term (with regular practice):")
    print("  • Increased baseline vagal tone")
    print("  • Faster return to coherence under stress")
    print("  • Improved emotional regulation capacity")
    
    print("\n" + "=" * 70)
    print("PHYSIOLOGICAL CORRELATES")
    print("=" * 70)
    print("\nTNFR Element → Physiological Process")
    print("-" * 70)
    print("SHA activation   → Parasympathetic dominance")
    print("νf → 0           → Sustained vagal activation")
    print("EPI preservation → Baroreceptor adaptation ('cardiac memory')")
    print("ΔNFR containment → Autonomic homeostatic balance")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: SHA AS CONSOLIDATION")
    print("=" * 70)
    print("\nSHA creates 'physiological memory' by:")
    print("  1. Reducing reorganization activity (νf → 0)")
    print("  2. Preserving coherent pattern (EPI intact)")
    print("  3. Allowing structural consolidation without interference")
    print("\nThis models how the autonomic nervous system 'learns' to")
    print("maintain coherence through repeated training + pause cycles.")
    
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
    print("  Section 1: Cardiac Coherence Training")
    print("\nScientific references:")
    print("  • McCraty & Shaffer (2015). Heart rate variability.")
    print("  • Lehrer & Gevirtz (2014). HRV biofeedback.")
